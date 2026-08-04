#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Distill the fixed-K Qwen3-TTS RVQ tail on Seed-TTS-Eval.

The script freezes the speech tokenizer, speaker encoder, Talker, full code
predictor, codec embeddings, and LM heads.  Only
``RVQTailDistillationModel`` receives gradients.  Target audio is encoded to
RVQ codes, but Code2Wav/the speech-tokenizer decoder is never constructed as a
training objective or called.

For every requested locale, rows are shuffled with Python's fixed seed 42 and
the first 50 percent are selected.  The exact manifest is persisted beside the
checkpoint.  The complementary half is therefore available for the A/B
comparison script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vllm_omni.model_executor.models.common.qwen3_code_predictor_tail import (  # noqa: E402
    RVQTailDistillationConfig,
    RVQTailDistillationModel,
)

DATASET_SEED = 42
TRAIN_FRACTION = 0.5
TARGET_SAMPLE_RATE = 24000


@dataclasses.dataclass(frozen=True)
class SeedTTSRow:
    """One resolved Seed-TTS-Eval voice-cloning training row."""

    utterance_id: str
    locale: str
    ref_text: str
    target_text: str
    ref_audio: Path
    target_audio: Path
    meta_line: str


@dataclasses.dataclass(frozen=True)
class PreparedSeedTTSRow:
    """CPU-resident immutable inputs that are reused across all epochs."""

    row: SeedTTSRow
    text_ids: torch.Tensor
    audio_codes: torch.Tensor
    speaker_embedding: torch.Tensor


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for tail distillation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default="tail_distill",
        help="Human-readable prefix for this run's uniquely named artifacts.",
    )
    parser.add_argument("--locales", nargs="+", choices=("en", "zh"), default=("en", "zh"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=8,
        help="Batch size used once for speech-tokenizer preprocessing.",
    )
    parser.add_argument(
        "--audio-loader-workers",
        type=int,
        default=4,
        help="CPU worker threads used to load and resample target/reference WAV files.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--truncation-k", type=int, default=8)
    parser.add_argument("--state-size", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional hard-label CE regularizer. Keep this at 0 for pure teacher "
            "distillation; teacher CE is still reported as a diagnostic baseline."
        ),
    )
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-frames-per-batch",
        type=int,
        default=2048,
        help="Deterministically subsample frame rows above this bound to cap memory.",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--validation-rows-per-locale", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--validation-max-frames-per-batch", type=int, default=1024)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print every N optimizer steps; the first optimizer step is always printed.",
    )
    return parser


def _build_run_id(run_name: str) -> str:
    """Build a filesystem-safe identifier that is unique for every invocation."""
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_name)
    safe_name = safe_name.strip("_-") or "tail_distill"
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    return f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _parse_meta_line(root: Path, locale: str, line: str) -> SeedTTSRow | None:
    """Parse and resolve one Seed-TTS ``meta.lst`` record."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split("|")
    if len(parts) < 4:
        raise ValueError(f"Malformed {locale}/meta.lst row: {stripped[:160]!r}")
    utterance_id, ref_text, ref_rel, target_text = (part.strip() for part in parts[:4])
    ref_audio = root / locale / ref_rel
    target_audio = root / locale / "wavs" / f"{utterance_id}.wav"
    if not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio is missing: {ref_audio}")
    if not target_audio.is_file():
        raise FileNotFoundError(f"Target audio is missing: {target_audio}")
    return SeedTTSRow(
        utterance_id=utterance_id,
        locale=locale,
        ref_text=ref_text,
        target_text=target_text,
        ref_audio=ref_audio.resolve(),
        target_audio=target_audio.resolve(),
        meta_line=stripped,
    )


def _load_training_rows(dataset_root: Path, locales: Sequence[str]) -> list[SeedTTSRow]:
    """Select the seed-42 first half independently for each locale."""
    selected: list[SeedTTSRow] = []
    for locale in locales:
        meta_path = dataset_root / locale / "meta.lst"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Seed-TTS metadata is missing: {meta_path}")
        rows = [
            row
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if (row := _parse_meta_line(dataset_root, locale, line)) is not None
        ]
        random.Random(DATASET_SEED).shuffle(rows)
        split = math.floor(len(rows) * TRAIN_FRACTION)
        if split == 0:
            raise ValueError(f"Not enough Seed-TTS rows in {meta_path}")
        selected.extend(rows[:split])
        print(f"Selected {split}/{len(rows)} {locale} rows with seed={DATASET_SEED}.", flush=True)
    return selected


def _load_validation_rows(
    dataset_root: Path,
    locales: Sequence[str],
    rows_per_locale: int,
) -> list[SeedTTSRow]:
    """Select a fixed subset from the complementary half of each locale."""
    selected: list[SeedTTSRow] = []
    for locale in locales:
        meta_path = dataset_root / locale / "meta.lst"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Seed-TTS metadata is missing: {meta_path}")
        rows = [
            row
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if (row := _parse_meta_line(dataset_root, locale, line)) is not None
        ]
        random.Random(DATASET_SEED).shuffle(rows)
        split = math.floor(len(rows) * TRAIN_FRACTION)
        held_out = rows[split:]
        locale_rows = held_out[: min(rows_per_locale, len(held_out))]
        if not locale_rows:
            raise ValueError(f"No held-out validation rows are available in {meta_path}")
        selected.extend(locale_rows)
        print(
            f"Selected {len(locale_rows)}/{len(held_out)} held-out {locale} rows "
            f"with seed={DATASET_SEED}.",
            flush=True,
        )
    return selected


def _write_manifest(
    rows: Sequence[SeedTTSRow],
    destination: Path,
    *,
    split_name: str,
) -> None:
    """Persist the exact deterministic training split for reproducibility."""
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            record = {
                "utterance_id": row.utterance_id,
                "locale": row.locale,
                "ref_text": row.ref_text,
                "target_text": row.target_text,
                "ref_audio": str(row.ref_audio),
                "target_audio": str(row.target_audio),
                "seed": DATASET_SEED,
                "fraction": TRAIN_FRACTION,
                "split": split_name,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _preprocessing_cache_path(
    output_dir: Path,
    model_path: Path,
    rows: Sequence[SeedTTSRow],
    num_groups: int,
) -> Path:
    """Return a content-sensitive cache path for epoch-invariant inputs."""
    digest = hashlib.sha256()
    digest.update(str(model_path).encode())
    digest.update(str(num_groups).encode())
    model_patterns = (
        "config.json",
        "model*.safetensors",
        "speech_tokenizer/*.json",
        "speech_tokenizer/*.safetensors",
    )
    model_files = (
        [model_path]
        if model_path.is_file()
        else sorted(path for pattern in model_patterns for path in model_path.glob(pattern))
    )
    for model_file in model_files:
        stat = model_file.stat()
        digest.update(str(model_file).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    for row in rows:
        digest.update(row.meta_line.encode())
        for audio_path in (row.target_audio, row.ref_audio):
            stat = audio_path.stat()
            digest.update(str(audio_path).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache_dir = output_dir / "preprocessing_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"seed_tts_inputs_{digest.hexdigest()[:20]}.safetensors"


def _load_audio_24k(path: Path) -> np.ndarray:
    """Load mono float32 audio and resample it to the model's 24 kHz rate."""
    import soundfile as sf

    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=int(sample_rate),
            target_sr=TARGET_SAMPLE_RATE,
        ).astype(np.float32)
    return waveform


def _normalize_audio_codes(audio_codes: Any, num_groups: int) -> list[torch.Tensor]:
    """Normalize tokenizer output to a list of ``[frames, codebooks]`` tensors."""
    if isinstance(audio_codes, torch.Tensor):
        codes = [audio_codes] if audio_codes.ndim == 2 else list(audio_codes)
    else:
        codes = list(audio_codes)
    normalized: list[torch.Tensor] = []
    for code in codes:
        tensor = torch.as_tensor(code, dtype=torch.long)
        if tensor.ndim != 2:
            raise ValueError(f"Expected rank-2 audio codes, got shape={tuple(tensor.shape)}")
        if tensor.shape[-1] != num_groups and tensor.shape[0] == num_groups:
            tensor = tensor.transpose(0, 1)
        if tensor.shape[-1] != num_groups:
            raise ValueError(
                f"Speech tokenizer returned {tensor.shape[-1]} codebooks; expected {num_groups}."
            )
        normalized.append(tensor)
    return normalized


def _load_row_audio(row: SeedTTSRow) -> tuple[np.ndarray, np.ndarray]:
    """Load one target/reference pair on a CPU worker."""
    return _load_audio_24k(row.target_audio), _load_audio_24k(row.ref_audio)


def _prepare_training_rows(
    teacher: torch.nn.Module,
    processor: Any,
    rows: Sequence[SeedTTSRow],
    num_groups: int,
    *,
    batch_size: int,
    audio_loader_workers: int,
    cache_path: Path,
) -> list[PreparedSeedTTSRow]:
    """Encode and cache all epoch-invariant CPU inputs exactly once."""
    if cache_path.is_file():
        cached = load_file(str(cache_path), device="cpu")
        expected_keys = {
            f"row_{row_index:05d}_{field}"
            for row_index in range(len(rows))
            for field in ("audio_codes", "speaker_embedding", "text_ids")
        }
        if set(cached) == expected_keys:
            print(
                json.dumps(
                    {
                        "record_type": "preprocessing_cache_hit",
                        "path": str(cache_path),
                        "rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return [
                PreparedSeedTTSRow(
                    row=row,
                    text_ids=cached[f"row_{row_index:05d}_text_ids"],
                    audio_codes=cached[f"row_{row_index:05d}_audio_codes"],
                    speaker_embedding=cached[f"row_{row_index:05d}_speaker_embedding"],
                )
                for row_index, row in enumerate(rows)
            ]

    prepared: list[PreparedSeedTTSRow] = []
    num_batches = math.ceil(len(rows) / batch_size)
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=audio_loader_workers) as executor:
        for batch_index, batch_rows in enumerate(_batches(rows, batch_size)):
            loaded_audio = list(executor.map(_load_row_audio, batch_rows))
            target_wavs = [pair[0] for pair in loaded_audio]
            ref_wavs = [pair[1] for pair in loaded_audio]
            text_ids = [_tokenize_target_text(processor, row.target_text).cpu() for row in batch_rows]
            with torch.inference_mode():
                encoded = teacher.speech_tokenizer.encode(target_wavs, sr=TARGET_SAMPLE_RATE)
                audio_codes = _normalize_audio_codes(encoded.audio_codes, num_groups)
                speaker_embeddings = torch.stack(
                    [teacher.extract_speaker_embedding(waveform, TARGET_SAMPLE_RATE) for waveform in ref_wavs]
                )
            audio_codes = [codes.detach().cpu().contiguous().clone() for codes in audio_codes]
            speaker_embeddings = speaker_embeddings.detach().float().cpu().contiguous()
            prepared.extend(
                PreparedSeedTTSRow(
                    row=row,
                    text_ids=ids.contiguous().clone(),
                    audio_codes=codes,
                    speaker_embedding=speaker_embedding.clone(),
                )
                for row, ids, codes, speaker_embedding in zip(
                    batch_rows,
                    text_ids,
                    audio_codes,
                    speaker_embeddings,
                )
            )
            if batch_index == 0 or batch_index + 1 == num_batches or (batch_index + 1) % 10 == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    json.dumps(
                        {
                            "record_type": "preprocessing_progress",
                            "batch": batch_index + 1,
                            "batches": num_batches,
                            "rows": len(prepared),
                            "elapsed_seconds": elapsed,
                            "rows_per_second": len(prepared) / max(elapsed, 1e-6),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    cache_state: dict[str, torch.Tensor] = {}
    for row_index, row in enumerate(prepared):
        prefix = f"row_{row_index:05d}_"
        cache_state[f"{prefix}text_ids"] = row.text_ids
        cache_state[f"{prefix}audio_codes"] = row.audio_codes
        cache_state[f"{prefix}speaker_embedding"] = row.speaker_embedding
    temporary_cache_path = cache_path.with_name(
        f".{cache_path.name}.{uuid.uuid4().hex}.tmp"
    )
    save_file(
        cache_state,
        str(temporary_cache_path),
        metadata={"format": "qwen3_tts_tail_preprocessing_v1", "rows": str(len(rows))},
    )
    temporary_cache_path.replace(cache_path)
    print(
        json.dumps(
            {
                "record_type": "preprocessing_cache_saved",
                "path": str(cache_path),
                "rows": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return prepared


def _tokenize_target_text(processor: Any, text: str) -> torch.Tensor:
    """Tokenize target text with the same assistant wrapper as official SFT."""
    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tokenized = processor(text=assistant_text, return_tensors="pt", padding=True)["input_ids"]
    tokenized = tokenized.unsqueeze(0) if tokenized.ndim == 1 else tokenized
    if tokenized.shape[1] <= 5:
        raise ValueError("Target text produced too few tokens for the Qwen3-TTS prompt wrapper.")
    return tokenized[:, :-5]


def _build_teacher_batch(
    *,
    teacher: torch.nn.Module,
    rows: Sequence[PreparedSeedTTSRow],
    device: torch.device,
    num_groups: int,
) -> dict[str, torch.Tensor]:
    """Build the official teacher-forcing layout used by Qwen3-TTS SFT."""
    config = teacher.config
    text_ids = [row.text_ids for row in rows]
    audio_codes = [row.audio_codes for row in rows]
    item_lengths = [ids.shape[1] + codes.shape[0] for ids, codes in zip(text_ids, audio_codes)]
    max_length = max(item_lengths) + 8
    batch_size = len(rows)

    input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long)
    codec_ids = torch.zeros((batch_size, max_length, num_groups), dtype=torch.long)
    text_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    codec_embed_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

    for row_idx, (ids, codes) in enumerate(zip(text_ids, audio_codes)):
        text_len = ids.shape[1]
        codec_len = codes.shape[0]
        codec_start = 8 + text_len - 1

        input_ids[row_idx, :3, 0] = ids[0, :3]
        input_ids[row_idx, 3:7, 0] = config.tts_pad_token_id
        input_ids[row_idx, 7, 0] = config.tts_bos_token_id
        input_ids[row_idx, 8 : 8 + text_len - 3, 0] = ids[0, 3:]
        input_ids[row_idx, 8 + text_len - 3, 0] = config.tts_eos_token_id
        input_ids[row_idx, 8 + text_len - 2 : 8 + text_len + codec_len, 0] = config.tts_pad_token_id
        text_mask[row_idx, : 8 + text_len + codec_len] = True

        talker_config = config.talker_config
        input_ids[row_idx, 3:8, 1] = torch.tensor(
            [
                talker_config.codec_nothink_id,
                talker_config.codec_think_bos_id,
                talker_config.codec_think_eos_id,
                0,
                talker_config.codec_pad_id,
            ],
            dtype=torch.long,
        )
        input_ids[row_idx, 8 : 8 + text_len - 2, 1] = talker_config.codec_pad_id
        input_ids[row_idx, codec_start - 1, 1] = talker_config.codec_bos_id
        input_ids[row_idx, codec_start : codec_start + codec_len, 1] = codes[:, 0]
        input_ids[row_idx, codec_start + codec_len, 1] = talker_config.codec_eos_token_id
        codec_ids[row_idx, codec_start : codec_start + codec_len] = codes

        codec_embed_mask[row_idx, 3 : 8 + text_len + codec_len] = True
        codec_embed_mask[row_idx, 6] = False
        codec_mask[row_idx, codec_start : codec_start + codec_len] = True
        attention_mask[row_idx, : 8 + text_len + codec_len] = True

    return {
        "input_ids": input_ids.to(device=device),
        "codec_ids": codec_ids.to(device=device),
        "text_mask": text_mask.unsqueeze(-1).to(device=device),
        "codec_embed_mask": codec_embed_mask.unsqueeze(-1).to(device=device),
        "codec_mask": codec_mask.to(device=device),
        "attention_mask": attention_mask.to(device=device),
        "speaker_embeddings": torch.stack([row.speaker_embedding for row in rows]).to(device=device),
    }


def _deterministic_frame_subset(
    hidden_states: torch.Tensor,
    codec_ids: torch.Tensor,
    max_frames: int,
    *,
    epoch: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound frame count with a repeatable seed-derived index selection."""
    if max_frames <= 0 or hidden_states.shape[0] <= max_frames:
        return hidden_states, codec_ids
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DATASET_SEED + epoch * 1_000_003 + step)
    indices = torch.randperm(hidden_states.shape[0], generator=generator)[:max_frames].to(hidden_states.device)
    return hidden_states.index_select(0, indices), codec_ids.index_select(0, indices)


def _align_talker_hidden_to_codec_frames(
    hidden_states: torch.Tensor,
    codec_mask: torch.Tensor,
) -> torch.Tensor:
    """Select the Talker state that causally predicts each codec frame.

    ``hidden_states`` is produced from ``inputs_embeds[:, :-1]``.  A codec
    frame at original sequence position ``p`` is predicted by the Talker
    state at ``p - 1``; that is also the state passed to the code predictor
    during autoregressive inference.  Shift the frame mask left by one to
    preserve this alignment and avoid exposing the current frame's codec
    embeddings to its own code-predictor target.
    """
    if codec_mask.ndim != 2 or hidden_states.ndim != 3:
        raise ValueError(
            "Expected hidden_states [batch, sequence-1, hidden] and "
            f"codec_mask [batch, sequence], got {tuple(hidden_states.shape)} "
            f"and {tuple(codec_mask.shape)}."
        )
    expected_shape = (codec_mask.shape[0], codec_mask.shape[1] - 1)
    if hidden_states.shape[:2] != expected_shape:
        raise ValueError(
            "Talker hidden states and shifted codec mask are incompatible: "
            f"hidden prefix={tuple(hidden_states.shape[:2])}, expected={expected_shape}."
        )
    return hidden_states[codec_mask[:, 1:]]


def _extract_code_predictor_teacher_targets(
    *,
    teacher: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    residual_embedding_weight: torch.Tensor,
    residual_codebook_offsets: torch.Tensor,
    truncation_k: int,
    max_frames: int,
    epoch: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run frozen Talker and code predictor to obtain tail distillation targets."""
    talker = teacher.talker
    code_predictor = talker.code_predictor
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]

    with torch.no_grad():
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        text_embeddings = talker.text_projection(talker.get_text_embeddings()(input_text_ids))
        text_embeddings = text_embeddings * batch["text_mask"]
        codec_embeddings = talker.get_input_embeddings()(input_codec_ids)
        codec_embeddings = codec_embeddings * batch["codec_embed_mask"]
        codec_embeddings[:, 6, :] = batch["speaker_embeddings"].to(codec_embeddings.dtype)
        inputs_embeds = text_embeddings + codec_embeddings

        residual_ids = codec_ids[:, :, 1:] + residual_codebook_offsets
        residual_embeddings = F.embedding(residual_ids, residual_embedding_weight)
        inputs_embeds = inputs_embeds + residual_embeddings.sum(dim=2) * batch["codec_mask"].unsqueeze(-1)

        outputs = talker(
            inputs_embeds=inputs_embeds[:, :-1],
            attention_mask=batch["attention_mask"][:, :-1],
            output_hidden_states=True,
            use_cache=False,
        )
        talker_hidden = _align_talker_hidden_to_codec_frames(
            outputs.hidden_states[0][-1],
            batch["codec_mask"],
        )
        frame_codes = codec_ids[batch["codec_mask"]]
        if talker_hidden.shape[0] != frame_codes.shape[0]:
            raise RuntimeError(
                "Talker/code-frame alignment produced different row counts: "
                f"hidden={talker_hidden.shape[0]}, codes={frame_codes.shape[0]}."
            )
        talker_hidden, frame_codes = _deterministic_frame_subset(
            talker_hidden,
            frame_codes,
            max_frames,
            epoch=epoch,
            step=step,
        )

        frame_residual_ids = frame_codes[:, 1:] + residual_codebook_offsets
        frame_residual_embeddings = F.embedding(frame_residual_ids, residual_embedding_weight)
        cp_inputs = torch.cat(
            (
                talker_hidden.unsqueeze(1),
                talker.get_input_embeddings()(frame_codes[:, :1]),
                frame_residual_embeddings[:, :-1],
            ),
            dim=1,
        )
        projected_inputs = code_predictor.small_to_mtp_projection(cp_inputs)
        teacher_hidden = code_predictor.model(
            inputs_embeds=projected_inputs,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state
        tail_previous_embeddings = projected_inputs[:, truncation_k:].detach()

    return (
        teacher_hidden[:, truncation_k - 1].detach().contiguous(),
        tail_previous_embeddings.contiguous(),
        teacher_hidden[:, truncation_k:].detach().contiguous(),
        frame_codes[:, truncation_k:].detach().contiguous(),
    )


def _distillation_loss(
    *,
    student: RVQTailDistillationModel,
    tail_lm_head_weight: torch.Tensor,
    exit_hidden: torch.Tensor,
    previous_embeddings: torch.Tensor,
    teacher_tail_hidden: torch.Tensor,
    tail_codes: torch.Tensor,
    truncation_k: int,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute tail CE, logit KL, and scale-aware hidden matching losses."""
    student_dtype = next(student.parameters()).dtype
    state, anchor = student.initialize(exit_hidden.to(dtype=student_dtype))
    previous_embeddings = previous_embeddings.to(dtype=student_dtype)
    tail_steps = tail_codes.shape[1]
    expected_prefix = (exit_hidden.shape[0], tail_steps)
    if previous_embeddings.shape[:2] != expected_prefix or teacher_tail_hidden.shape[:2] != expected_prefix:
        raise ValueError(
            "Tail distillation tensors have incompatible batch/step dimensions: "
            f"codes={tuple(tail_codes.shape)}, previous={tuple(previous_embeddings.shape)}, "
            f"teacher={tuple(teacher_tail_hidden.shape)}."
        )
    if tail_lm_head_weight.shape[0] != tail_steps:
        raise ValueError(
            f"Expected {tail_steps} tail LM heads, got weight shape={tuple(tail_lm_head_weight.shape)}."
        )
    predicted_tail_hidden = []
    for tail_offset in range(tail_steps):
        predicted_hidden, state = student.predict_next(
            state,
            anchor,
            previous_embeddings[:, tail_offset],
            truncation_k + tail_offset,
        )
        predicted_tail_hidden.append(predicted_hidden)
    predicted_tail_hidden_steps = torch.stack(predicted_tail_hidden, dim=0)

    projection_dtype = tail_lm_head_weight.dtype
    student_logits = torch.bmm(
        predicted_tail_hidden_steps.to(dtype=projection_dtype),
        tail_lm_head_weight,
    ).transpose(0, 1).float()
    with torch.no_grad():
        teacher_logits = torch.bmm(
            teacher_tail_hidden.transpose(0, 1).contiguous().to(dtype=projection_dtype),
            tail_lm_head_weight,
        ).transpose(0, 1).float()

    flat_student_logits = student_logits.flatten(0, 1)
    flat_teacher_logits = teacher_logits.flatten(0, 1)
    flat_labels = tail_codes.flatten()
    if ce_weight > 0.0:
        ce_loss = F.cross_entropy(flat_student_logits, flat_labels)
    else:
        with torch.no_grad():
            ce_loss = F.cross_entropy(flat_student_logits, flat_labels)
    teacher_probs = F.softmax(flat_teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(flat_student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)

    predicted_hidden_float = predicted_tail_hidden_steps.transpose(0, 1).float()
    target_hidden = teacher_tail_hidden.float()
    target_rms = target_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    hidden_loss = F.mse_loss(predicted_hidden_float / target_rms, target_hidden / target_rms)
    loss = ce_weight * ce_loss + kl_weight * kl_loss + hidden_weight * hidden_loss
    with torch.no_grad():
        teacher_ce_loss = F.cross_entropy(flat_teacher_logits, flat_labels)
        hidden_cosine_similarity = F.cosine_similarity(
            predicted_hidden_float,
            target_hidden,
            dim=-1,
        ).mean()
        hidden_cosine_distance = 1.0 - hidden_cosine_similarity
        student_top1 = student_logits.argmax(dim=-1)
        teacher_top1 = teacher_logits.argmax(dim=-1)
        teacher_agreement = student_top1 == teacher_top1
    metrics = {
        "loss": loss.detach(),
        "ce": ce_loss.detach(),
        "teacher_ce": teacher_ce_loss.detach(),
        "ce_excess": (ce_loss - teacher_ce_loss).detach(),
        "kl": kl_loss.detach(),
        "hidden": hidden_loss.detach(),
        # ``hidden_cosine`` is kept as a compatibility alias for the old
        # distance-valued metric.  Prefer the explicitly named fields below.
        "hidden_cosine": hidden_cosine_distance.detach(),
        "hidden_cosine_distance": hidden_cosine_distance.detach(),
        "hidden_cosine_similarity": hidden_cosine_similarity.detach(),
        "student_target_top1": (student_top1 == tail_codes).float().mean(),
        "teacher_target_top1": (teacher_top1 == tail_codes).float().mean(),
        "student_teacher_top1": teacher_agreement.float().mean(),
        "student_teacher_tail_exact": teacher_agreement.all(dim=1).float().mean(),
    }
    for tail_offset in range(tail_steps):
        codebook_index = truncation_k + tail_offset
        metrics[f"student_teacher_top1_codebook_{codebook_index}"] = (
            teacher_agreement[:, tail_offset].float().mean()
        )
    return loss, metrics


def _metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer a group of scalar metrics with a single device synchronization."""
    names = list(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names]).cpu().tolist()
    return dict(zip(names, values))


def _batches(rows: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield contiguous mini-batches from an already shuffled row sequence."""
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def _length_bucketed_batches(
    rows: Sequence[PreparedSeedTTSRow],
    batch_size: int,
    *,
    seed: int,
) -> list[list[PreparedSeedTTSRow]]:
    """Group similar codec lengths while retaining deterministic batch shuffling."""
    ordered = sorted(rows, key=lambda row: row.audio_codes.shape[0])
    batches = list(_batches(ordered, batch_size))
    partial_batch = batches.pop() if batches and len(batches[-1]) < batch_size else None
    generator = random.Random(seed)
    generator.shuffle(batches)
    for batch in batches:
        generator.shuffle(batch)
    if partial_batch is not None:
        generator.shuffle(partial_batch)
        batches.append(partial_batch)
    return batches


def _evaluate_distillation(
    *,
    teacher: torch.nn.Module,
    student: RVQTailDistillationModel,
    rows: Sequence[PreparedSeedTTSRow],
    device: torch.device,
    teacher_dtype: torch.dtype,
    num_groups: int,
    truncation_k: int,
    batch_size: int,
    max_frames: int,
    tail_lm_head_weight: torch.Tensor,
    residual_embedding_weight: torch.Tensor,
    residual_codebook_offsets: torch.Tensor,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> dict[str, Any]:
    """Evaluate deterministic tail similarity on held-out Seed-TTS rows.

    Teacher and student receive the same ground-truth prefix and preceding
    code embeddings.  Their greedy argmax tail codes are then compared.  This
    teacher-forced setup isolates approximation quality from sampling noise
    and autoregressive error propagation.
    """
    batches = _length_bucketed_batches(rows, batch_size, seed=DATASET_SEED)
    metric_totals: dict[str, torch.Tensor] = {}
    total_frames = 0
    started_at = time.perf_counter()
    was_training = student.training
    student.eval()
    try:
        with torch.inference_mode():
            for batch_index, batch_rows in enumerate(batches):
                batch = _build_teacher_batch(
                    teacher=teacher,
                    rows=batch_rows,
                    device=device,
                    num_groups=num_groups,
                )
                with _autocast_context(device, teacher_dtype):
                    targets = _extract_code_predictor_teacher_targets(
                        teacher=teacher,
                        batch=batch,
                        residual_embedding_weight=residual_embedding_weight,
                        residual_codebook_offsets=residual_codebook_offsets,
                        truncation_k=truncation_k,
                        max_frames=max_frames,
                        epoch=0,
                        step=batch_index,
                    )
                _, metrics = _distillation_loss(
                    student=student,
                    tail_lm_head_weight=tail_lm_head_weight,
                    exit_hidden=targets[0],
                    previous_embeddings=targets[1],
                    teacher_tail_hidden=targets[2],
                    tail_codes=targets[3],
                    truncation_k=truncation_k,
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kl_weight=kl_weight,
                    hidden_weight=hidden_weight,
                )
                batch_frames = targets[0].shape[0]
                total_frames += batch_frames
                for name, value in metrics.items():
                    weighted = value * batch_frames
                    if name in metric_totals:
                        metric_totals[name].add_(weighted)
                    else:
                        metric_totals[name] = weighted
        averaged_metrics = _metrics_to_floats(
            {name: total / total_frames for name, total in metric_totals.items()}
        )
    finally:
        student.train(was_training)
    elapsed = time.perf_counter() - started_at
    return {
        "record_type": "validation_summary",
        "validation_mode": "teacher_forced_argmax_tail",
        "held_out_rows": len(rows),
        "batches": len(batches),
        "frames": total_frames,
        "elapsed_seconds": elapsed,
        "frames_per_second": total_frames / max(elapsed, 1e-6),
        **averaged_metrics,
    }


def _accumulation_group_size(
    batch_index: int,
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the actual size of this batch's epoch-local accumulation group."""
    group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
    return min(gradient_accumulation_steps, num_batches - group_start)


def _save_checkpoint(
    student: RVQTailDistillationModel,
    output_dir: Path,
    filename: str,
    metadata: dict[str, str],
) -> Path:
    """Save an atomic safetensors student checkpoint."""
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    state = {name: tensor.detach().float().cpu().contiguous() for name, tensor in student.state_dict().items()}
    save_file(state, str(temporary), metadata=metadata)
    temporary.replace(destination)
    return destination


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a CLI dtype name to a torch dtype."""
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _autocast_context(device: torch.device, dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    """Return a supported autocast context for frozen teacher execution."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prepare_device_backend(device_name: str) -> None:
    """Import the out-of-tree torch backend required by an NPU device.

    Args:
        device_name: Requested torch device string.

    Raises:
        ImportError: If NPU training is requested without ``torch_npu``.
    """
    if not device_name.startswith("npu"):
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise ImportError("NPU distillation requires torch_npu in the active Python environment.") from exc


def main() -> int:
    """Run deterministic Seed-TTS RVQ-tail distillation."""
    args = _build_parser().parse_args()
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.preprocessing_batch_size < 1:
        raise ValueError("Batch sizes and gradient-accumulation-steps must be positive.")
    if args.audio_loader_workers < 1:
        raise ValueError("audio-loader-workers must be positive.")
    if args.validation_rows_per_locale < 1 or args.validation_batch_size < 1:
        raise ValueError("validation-rows-per-locale and validation-batch-size must be positive.")
    if args.validation_max_frames_per_batch < 1:
        raise ValueError("validation-max-frames-per-batch must be positive.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative.")
    if args.max_grad_norm <= 0.0:
        raise ValueError("max-grad-norm must be positive.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if args.log_every < 1:
        raise ValueError("log-every must be positive.")
    loss_weights = (args.ce_weight, args.kl_weight, args.hidden_weight)
    if any(weight < 0.0 for weight in loss_weights) or not any(weight > 0.0 for weight in loss_weights):
        raise ValueError("Loss weights must be non-negative and at least one must be positive.")
    _prepare_device_backend(args.device)

    model_path = args.model_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(args.run_name)
    raw_rows = _load_training_rows(dataset_root, args.locales)
    train_manifest_path = output_dir / f"train_manifest_{run_id}.jsonl"
    _write_manifest(raw_rows, train_manifest_path, split_name="train")

    random.seed(DATASET_SEED)
    np.random.seed(DATASET_SEED)
    torch.manual_seed(DATASET_SEED)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(DATASET_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DATASET_SEED)

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "The training entry requires the official qwen-tts package. "
            "Install the version matching the local Qwen3-TTS checkpoint."
        ) from exc

    teacher_dtype = _resolve_dtype(args.dtype)
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=args.device,
        dtype=teacher_dtype,
        attn_implementation=args.attn_implementation,
    )
    teacher = wrapper.model
    teacher.eval()
    teacher.requires_grad_(False)
    talker = teacher.talker
    code_predictor = talker.code_predictor
    num_groups = int(code_predictor.config.num_code_groups)
    hidden_size = int(code_predictor.config.hidden_size)
    if not 2 <= args.truncation_k < num_groups:
        raise ValueError(f"truncation-k must be in [2, {num_groups - 1}], got {args.truncation_k}.")

    device = next(teacher.parameters()).device
    preprocessing_cache_path = _preprocessing_cache_path(
        output_dir,
        model_path,
        raw_rows,
        num_groups,
    )
    rows = _prepare_training_rows(
        teacher,
        wrapper.processor,
        raw_rows,
        num_groups,
        batch_size=args.preprocessing_batch_size,
        audio_loader_workers=args.audio_loader_workers,
        cache_path=preprocessing_cache_path,
    )
    tail_lm_head_weight = torch.stack(
        [code_predictor.lm_head[index - 1].weight.detach() for index in range(args.truncation_k, num_groups)]
    ).transpose(1, 2).contiguous()
    residual_embedding_modules = list(code_predictor.get_input_embeddings())
    if len(residual_embedding_modules) != num_groups - 1:
        raise ValueError(
            f"Expected {num_groups - 1} residual embedding tables, got {len(residual_embedding_modules)}."
        )
    residual_vocab_size = residual_embedding_modules[0].weight.shape[0]
    if any(module.weight.shape[0] != residual_vocab_size for module in residual_embedding_modules):
        raise ValueError("All residual codebook embeddings must use the same vocabulary size.")
    residual_embedding_weight = torch.cat(
        [module.weight.detach() for module in residual_embedding_modules],
        dim=0,
    ).contiguous()
    residual_codebook_offsets = (
        torch.arange(num_groups - 1, device=device, dtype=torch.long) * residual_vocab_size
    )
    student = RVQTailDistillationModel(
        RVQTailDistillationConfig(
            hidden_size=hidden_size,
            state_size=args.state_size,
            num_code_groups=num_groups,
        )
    ).to(device=device, dtype=torch.float32)
    student.train()
    if args.resume_from is not None:
        state = load_file(str(args.resume_from.expanduser().resolve()), device="cpu")
        student.load_state_dict(state, strict=True)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / f"metrics_{run_id}.jsonl"
    global_step = 0
    optimizer_step = 0
    training_config_record = {
        "record_type": "training_config",
        "run_id": run_id,
        "rows": len(rows),
        "truncation_k": args.truncation_k,
        "num_code_groups": num_groups,
        "hidden_size": hidden_size,
        "state_size": args.state_size,
        "batch_size": args.batch_size,
        "preprocessing_batch_size": args.preprocessing_batch_size,
        "audio_loader_workers": args.audio_loader_workers,
        "preprocessing_cache": str(preprocessing_cache_path),
        "train_manifest": str(train_manifest_path),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "kl_weight": args.kl_weight,
        "hidden_weight": args.hidden_weight,
        "validation_rows_per_locale": args.validation_rows_per_locale,
        "validation_batch_size": args.validation_batch_size,
        "validation_max_frames_per_batch": args.validation_max_frames_per_batch,
        "trainable_parameters": sum(parameter.numel() for parameter in student.parameters()),
    }
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(training_config_record, sort_keys=True) + "\n")
    print(json.dumps(training_config_record, sort_keys=True), flush=True)

    interval_started_at = time.perf_counter()
    interval_frames = 0
    for epoch in range(args.epochs):
        epoch_batches = _length_bucketed_batches(rows, args.batch_size, seed=DATASET_SEED + epoch)
        num_batches = len(epoch_batches)
        epoch_metric_totals: dict[str, torch.Tensor] = {}
        epoch_frames = 0
        for batch_index, batch_rows in enumerate(epoch_batches):
            batch = _build_teacher_batch(
                teacher=teacher,
                rows=batch_rows,
                device=device,
                num_groups=num_groups,
            )
            with _autocast_context(device, teacher_dtype):
                targets = _extract_code_predictor_teacher_targets(
                    teacher=teacher,
                    batch=batch,
                    residual_embedding_weight=residual_embedding_weight,
                    residual_codebook_offsets=residual_codebook_offsets,
                    truncation_k=args.truncation_k,
                    max_frames=args.max_frames_per_batch,
                    epoch=epoch,
                    step=batch_index,
                )
            loss, metrics = _distillation_loss(
                student=student,
                tail_lm_head_weight=tail_lm_head_weight,
                exit_hidden=targets[0],
                previous_embeddings=targets[1],
                teacher_tail_hidden=targets[2],
                tail_codes=targets[3],
                truncation_k=args.truncation_k,
                temperature=args.temperature,
                ce_weight=args.ce_weight,
                kl_weight=args.kl_weight,
                hidden_weight=args.hidden_weight,
            )
            accumulation_group_size = _accumulation_group_size(
                batch_index,
                num_batches,
                args.gradient_accumulation_steps,
            )
            (loss / accumulation_group_size).backward()
            global_step += 1
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == num_batches
            )
            should_log = False
            grad_norm: torch.Tensor | None = None
            output_projection_grad_norm: torch.Tensor | None = None
            update_norm: torch.Tensor | None = None
            if should_step:
                next_optimizer_step = optimizer_step + 1
                should_log = next_optimizer_step == 1 or next_optimizer_step % args.log_every == 0
                update_probe = student.output_projection.weight
                probe_before = update_probe.detach().clone() if should_log else None
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                if should_log and update_probe.grad is not None:
                    output_projection_grad_norm = update_probe.grad.detach().float().norm()
                optimizer.step()
                if probe_before is not None:
                    update_norm = (update_probe.detach() - probe_before).float().norm()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step = next_optimizer_step

            batch_frames = targets[0].shape[0]
            epoch_frames += batch_frames
            interval_frames += batch_frames
            for name, value in metrics.items():
                weighted = value * batch_frames
                if name in epoch_metric_totals:
                    epoch_metric_totals[name].add_(weighted)
                else:
                    epoch_metric_totals[name] = weighted

            if should_log:
                if output_projection_grad_norm is None or update_norm is None or grad_norm is None:
                    raise RuntimeError("Optimizer diagnostics were not collected on a logging step.")
                scalar_metrics = _metrics_to_floats(
                    {
                        **metrics,
                        "grad_norm": grad_norm,
                        "output_projection_grad_norm": output_projection_grad_norm,
                        "update_norm": update_norm,
                    }
                )
                if optimizer_step == 1 and any(
                    not math.isfinite(scalar_metrics[name]) or scalar_metrics[name] <= 0.0
                    for name in ("grad_norm", "output_projection_grad_norm", "update_norm")
                ):
                    raise FloatingPointError(
                        "The first optimizer step did not produce finite non-zero gradients and an update: "
                        f"{scalar_metrics}"
                    )
                interval_elapsed = time.perf_counter() - interval_started_at
                record = {
                    "record_type": "optimizer_step",
                    "run_id": run_id,
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "accumulation_group_size": accumulation_group_size,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "frames_per_second": interval_frames / max(interval_elapsed, 1e-6),
                    **scalar_metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                interval_started_at = time.perf_counter()
                interval_frames = 0

        epoch_metrics = _metrics_to_floats(
            {name: total / epoch_frames for name, total in epoch_metric_totals.items()}
        )
        epoch_record = {
            "record_type": "epoch_summary",
            "run_id": run_id,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "frames": epoch_frames,
            **epoch_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

        metadata = {
            "format": "qwen3_tts_rvq_tail_distillation_v2",
            "run_id": run_id,
            "seed": str(DATASET_SEED),
            "train_fraction": str(TRAIN_FRACTION),
            "truncation_k": str(args.truncation_k),
            "num_code_groups": str(num_groups),
            "hidden_size": str(hidden_size),
            "state_size": str(args.state_size),
            "epoch": str(epoch),
            "ce_weight": str(args.ce_weight),
            "kl_weight": str(args.kl_weight),
            "hidden_weight": str(args.hidden_weight),
        }
        _save_checkpoint(
            student,
            output_dir,
            f"qwen3_tts_rvq_tail_{run_id}_epoch_{epoch + 1}.safetensors",
            metadata,
        )

    final_path = _save_checkpoint(
        student,
        output_dir,
        f"qwen3_tts_rvq_tail_{run_id}.safetensors",
        {
            "format": "qwen3_tts_rvq_tail_distillation_v2",
            "run_id": run_id,
            "seed": str(DATASET_SEED),
            "train_fraction": str(TRAIN_FRACTION),
            "truncation_k": str(args.truncation_k),
            "num_code_groups": str(num_groups),
            "hidden_size": str(hidden_size),
            "state_size": str(args.state_size),
            "epochs": str(args.epochs),
            "ce_weight": str(args.ce_weight),
            "kl_weight": str(args.kl_weight),
            "hidden_weight": str(args.hidden_weight),
        },
    )

    validation_raw_rows = _load_validation_rows(
        dataset_root,
        args.locales,
        args.validation_rows_per_locale,
    )
    validation_manifest_path = output_dir / f"validation_manifest_{run_id}.jsonl"
    _write_manifest(validation_raw_rows, validation_manifest_path, split_name="validation")
    validation_cache_path = _preprocessing_cache_path(
        output_dir,
        model_path,
        validation_raw_rows,
        num_groups,
    )
    validation_rows = _prepare_training_rows(
        teacher,
        wrapper.processor,
        validation_raw_rows,
        num_groups,
        batch_size=args.preprocessing_batch_size,
        audio_loader_workers=args.audio_loader_workers,
        cache_path=validation_cache_path,
    )
    validation_summary = _evaluate_distillation(
        teacher=teacher,
        student=student,
        rows=validation_rows,
        device=device,
        teacher_dtype=teacher_dtype,
        num_groups=num_groups,
        truncation_k=args.truncation_k,
        batch_size=args.validation_batch_size,
        max_frames=args.validation_max_frames_per_batch,
        tail_lm_head_weight=tail_lm_head_weight,
        residual_embedding_weight=residual_embedding_weight,
        residual_codebook_offsets=residual_codebook_offsets,
        temperature=args.temperature,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        hidden_weight=args.hidden_weight,
    )
    validation_summary.update(
        {
            "run_id": run_id,
            "checkpoint": str(final_path),
            "manifest": str(validation_manifest_path),
            "preprocessing_cache": str(validation_cache_path),
            "metric_directions": {
                "kl": "lower_is_better",
                "hidden_cosine_distance": "lower_is_better",
                "hidden_cosine_similarity": "higher_is_better",
                "student_teacher_top1": "higher_is_better",
                "student_teacher_tail_exact": "higher_is_better",
            },
        }
    )
    validation_summary_path = output_dir / f"validation_summary_{run_id}.json"
    validation_summary_path.write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(validation_summary, sort_keys=True) + "\n")
    print(json.dumps(validation_summary, sort_keys=True), flush=True)

    config = {
        "run_id": run_id,
        "code_predictor_truncation_mode": "fixed",
        "code_predictor_truncation_k": args.truncation_k,
        "code_predictor_early_exit_fill_strategy": "distillation",
        "code_predictor_distillation_state_size": args.state_size,
        "code_predictor_distillation_weights": str(final_path),
        "dataset_seed": DATASET_SEED,
        "train_fraction": TRAIN_FRACTION,
        "locales": list(args.locales),
        "metrics": str(metrics_path),
        "train_manifest": str(train_manifest_path),
        "validation_manifest": str(validation_manifest_path),
        "validation_summary": str(validation_summary_path),
        "validation_mode": validation_summary["validation_mode"],
        "validation": {
            "kl": validation_summary["kl"],
            "hidden_cosine_similarity": validation_summary["hidden_cosine_similarity"],
            "hidden_cosine_distance": validation_summary["hidden_cosine_distance"],
            "student_teacher_top1": validation_summary["student_teacher_top1"],
            "student_teacher_tail_exact": validation_summary["student_teacher_tail_exact"],
        },
    }
    rendered_config = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    config_path = output_dir / f"distillation_config_{run_id}.json"
    config_path.write_text(rendered_config, encoding="utf-8")
    latest_config_path = output_dir / "distillation_config.json"
    latest_config_path.write_text(rendered_config, encoding="utf-8")
    print(
        f"Saved run={run_id} checkpoint={final_path} validation={validation_summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
