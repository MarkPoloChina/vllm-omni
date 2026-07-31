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
import contextlib
import dataclasses
import json
import math
import random
import sys
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for tail distillation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locales", nargs="+", choices=("en", "zh"), default=("en", "zh"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--truncation-k", type=int, default=8)
    parser.add_argument("--state-size", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-frames-per-batch",
        type=int,
        default=512,
        help="Deterministically subsample frame rows above this bound to cap memory.",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


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


def _write_manifest(rows: Sequence[SeedTTSRow], output_dir: Path) -> None:
    """Persist the exact deterministic training split for reproducibility."""
    manifest_path = output_dir / "train_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
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
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def _encode_batch_audio(
    teacher: torch.nn.Module,
    rows: Sequence[SeedTTSRow],
    num_groups: int,
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """Encode target audio and load reference waveforms without decoding audio."""
    target_wavs = [_load_audio_24k(row.target_audio) for row in rows]
    ref_wavs = [_load_audio_24k(row.ref_audio) for row in rows]
    with torch.inference_mode():
        encoded = teacher.speech_tokenizer.encode(target_wavs, sr=TARGET_SAMPLE_RATE)
    return _normalize_audio_codes(encoded.audio_codes, num_groups), ref_wavs


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
    processor: Any,
    rows: Sequence[SeedTTSRow],
    audio_codes: Sequence[torch.Tensor],
    ref_wavs: Sequence[np.ndarray],
    device: torch.device,
    num_groups: int,
) -> dict[str, torch.Tensor]:
    """Build the official teacher-forcing layout used by Qwen3-TTS SFT."""
    config = teacher.config
    text_ids = [_tokenize_target_text(processor, row.target_text) for row in rows]
    item_lengths = [ids.shape[1] + codes.shape[0] for ids, codes in zip(text_ids, audio_codes)]
    max_length = max(item_lengths) + 8
    batch_size = len(rows)

    input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long, device=device)
    codec_ids = torch.zeros((batch_size, max_length, num_groups), dtype=torch.long, device=device)
    text_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    codec_embed_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)

    for row_idx, (ids, codes) in enumerate(zip(text_ids, audio_codes)):
        ids = ids.to(device=device)
        codes = codes.to(device=device)
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
            device=device,
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

    speaker_embeddings = []
    for waveform in ref_wavs:
        embedding = teacher.extract_speaker_embedding(waveform, TARGET_SAMPLE_RATE)
        speaker_embeddings.append(embedding.detach())

    return {
        "input_ids": input_ids,
        "codec_ids": codec_ids,
        "text_mask": text_mask.unsqueeze(-1),
        "codec_embed_mask": codec_embed_mask.unsqueeze(-1),
        "codec_mask": codec_mask,
        "attention_mask": attention_mask,
        "speaker_embeddings": torch.stack(speaker_embeddings).to(device=device),
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


def _extract_code_predictor_teacher_targets(
    *,
    teacher: torch.nn.Module,
    batch: dict[str, torch.Tensor],
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

        for codebook_idx in range(1, code_predictor.config.num_code_groups):
            residual = code_predictor.get_input_embeddings()[codebook_idx - 1](codec_ids[:, :, codebook_idx])
            inputs_embeds = inputs_embeds + residual * batch["codec_mask"].unsqueeze(-1)

        outputs = talker(
            inputs_embeds=inputs_embeds[:, :-1],
            attention_mask=batch["attention_mask"][:, :-1],
            output_hidden_states=True,
            use_cache=False,
        )
        talker_hidden = outputs.hidden_states[0][-1][batch["codec_mask"][:, :-1]]
        frame_codes = codec_ids[batch["codec_mask"]]
        talker_hidden, frame_codes = _deterministic_frame_subset(
            talker_hidden,
            frame_codes,
            max_frames,
            epoch=epoch,
            step=step,
        )

        cp_inputs = [talker_hidden.unsqueeze(1), talker.get_input_embeddings()(frame_codes[:, :1])]
        for codebook_idx in range(1, code_predictor.config.num_code_groups - 1):
            cp_inputs.append(
                code_predictor.get_input_embeddings()[codebook_idx - 1](
                    frame_codes[:, codebook_idx : codebook_idx + 1]
                )
            )
        projected_inputs = code_predictor.small_to_mtp_projection(torch.cat(cp_inputs, dim=1))
        teacher_hidden = code_predictor.model(
            inputs_embeds=projected_inputs,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state
        previous_embedding = code_predictor.small_to_mtp_projection(
            code_predictor.get_input_embeddings()[truncation_k - 2](
                frame_codes[:, truncation_k - 1 : truncation_k]
            )
        ).squeeze(1)

    return (
        teacher_hidden[:, truncation_k - 1].detach(),
        previous_embedding.detach(),
        teacher_hidden.detach(),
        frame_codes.detach(),
    )


def _distillation_loss(
    *,
    student: RVQTailDistillationModel,
    code_predictor: torch.nn.Module,
    exit_hidden: torch.Tensor,
    previous_embedding: torch.Tensor,
    teacher_hidden: torch.Tensor,
    frame_codes: torch.Tensor,
    truncation_k: int,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute tail CE, logit KL, and hidden-state matching losses."""
    student_dtype = next(student.parameters()).dtype
    state, anchor = student.initialize(exit_hidden.to(dtype=student_dtype))
    previous = previous_embedding.to(dtype=student_dtype)
    ce_total = torch.zeros((), dtype=torch.float32, device=exit_hidden.device)
    kl_total = torch.zeros_like(ce_total)
    hidden_total = torch.zeros_like(ce_total)
    tail_steps = frame_codes.shape[1] - truncation_k

    for codebook_idx in range(truncation_k, frame_codes.shape[1]):
        predicted_hidden, state = student.predict_next(state, anchor, previous, codebook_idx)
        target_hidden = teacher_hidden[:, codebook_idx].float()
        lm_head = code_predictor.lm_head[codebook_idx - 1]
        lm_head_dtype = next(lm_head.parameters()).dtype
        student_logits = lm_head(predicted_hidden.to(dtype=lm_head_dtype)).float()
        with torch.no_grad():
            teacher_logits = lm_head(teacher_hidden[:, codebook_idx]).float()

        ce_total = ce_total + F.cross_entropy(student_logits, frame_codes[:, codebook_idx])
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl_total = kl_total + F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            temperature**2
        )
        hidden_total = hidden_total + (1.0 - F.cosine_similarity(predicted_hidden.float(), target_hidden).mean())

        if codebook_idx < frame_codes.shape[1] - 1:
            previous = code_predictor.small_to_mtp_projection(
                code_predictor.get_input_embeddings()[codebook_idx - 1](
                    frame_codes[:, codebook_idx : codebook_idx + 1]
                )
            ).squeeze(1)
            previous = previous.detach().to(dtype=student_dtype)

    ce_loss = ce_total / tail_steps
    kl_loss = kl_total / tail_steps
    hidden_loss = hidden_total / tail_steps
    loss = ce_weight * ce_loss + kl_weight * kl_loss + hidden_weight * hidden_loss
    metrics = {
        "loss": float(loss.detach()),
        "ce": float(ce_loss.detach()),
        "kl": float(kl_loss.detach()),
        "hidden": float(hidden_loss.detach()),
        "frames": float(frame_codes.shape[0]),
    }
    return loss, metrics


def _batches(rows: Sequence[SeedTTSRow], batch_size: int) -> Iterator[list[SeedTTSRow]]:
    """Yield contiguous mini-batches from an already shuffled row sequence."""
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


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
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch-size and gradient-accumulation-steps must be positive.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    _prepare_device_backend(args.device)

    model_path = args.model_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_training_rows(dataset_root, args.locales)
    _write_manifest(rows, output_dir)

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
    student = RVQTailDistillationModel(
        RVQTailDistillationConfig(
            hidden_size=hidden_size,
            state_size=args.state_size,
            num_code_groups=num_groups,
        )
    ).to(device=device, dtype=torch.float32)
    if args.resume_from is not None:
        state = load_file(str(args.resume_from.expanduser().resolve()), device="cpu")
        student.load_state_dict(state, strict=True)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output_dir / "metrics.jsonl"
    global_step = 0
    optimizer_step = 0
    print(
        f"Training only RVQTailDistillationModel: rows={len(rows)} K={args.truncation_k} "
        f"Q={num_groups} hidden={hidden_size} state={args.state_size}",
        flush=True,
    )

    for epoch in range(args.epochs):
        epoch_rows = list(rows)
        random.Random(DATASET_SEED + epoch).shuffle(epoch_rows)
        num_batches = math.ceil(len(epoch_rows) / args.batch_size)
        for batch_index, batch_rows in enumerate(_batches(epoch_rows, args.batch_size)):
            audio_codes, ref_wavs = _encode_batch_audio(teacher, batch_rows, num_groups)
            batch = _build_teacher_batch(
                teacher=teacher,
                processor=wrapper.processor,
                rows=batch_rows,
                audio_codes=audio_codes,
                ref_wavs=ref_wavs,
                device=device,
                num_groups=num_groups,
            )
            with _autocast_context(device, teacher_dtype):
                targets = _extract_code_predictor_teacher_targets(
                    teacher=teacher,
                    batch=batch,
                    truncation_k=args.truncation_k,
                    max_frames=args.max_frames_per_batch,
                    epoch=epoch,
                    step=batch_index,
                )
            loss, metrics = _distillation_loss(
                student=student,
                code_predictor=code_predictor,
                exit_hidden=targets[0],
                previous_embedding=targets[1],
                teacher_hidden=targets[2],
                frame_codes=targets[3],
                truncation_k=args.truncation_k,
                temperature=args.temperature,
                ce_weight=args.ce_weight,
                kl_weight=args.kl_weight,
                hidden_weight=args.hidden_weight,
            )
            (loss / args.gradient_accumulation_steps).backward()
            global_step += 1
            should_step = (
                global_step % args.gradient_accumulation_steps == 0 or batch_index + 1 == num_batches
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            record = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if batch_index % args.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)

        metadata = {
            "format": "qwen3_tts_rvq_tail_distillation_v1",
            "seed": str(DATASET_SEED),
            "train_fraction": str(TRAIN_FRACTION),
            "truncation_k": str(args.truncation_k),
            "num_code_groups": str(num_groups),
            "hidden_size": str(hidden_size),
            "state_size": str(args.state_size),
            "epoch": str(epoch),
        }
        _save_checkpoint(student, output_dir, f"qwen3_tts_rvq_tail_epoch_{epoch + 1}.safetensors", metadata)

    final_path = _save_checkpoint(
        student,
        output_dir,
        "qwen3_tts_rvq_tail.safetensors",
        {
            "format": "qwen3_tts_rvq_tail_distillation_v1",
            "seed": str(DATASET_SEED),
            "train_fraction": str(TRAIN_FRACTION),
            "truncation_k": str(args.truncation_k),
            "num_code_groups": str(num_groups),
            "hidden_size": str(hidden_size),
            "state_size": str(args.state_size),
            "epochs": str(args.epochs),
        },
    )
    config_path = output_dir / "distillation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "weights": str(final_path),
                "code_predictor_truncation_mode": "fixed",
                "code_predictor_truncation_k": args.truncation_k,
                "code_predictor_early_exit_fill_strategy": "distillation",
                "code_predictor_distillation_state_size": args.state_size,
                "dataset_seed": DATASET_SEED,
                "train_fraction": TRAIN_FRACTION,
                "locales": list(args.locales),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved final RVQ-tail checkpoint to {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
