#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Distill the Qwen3-TTS RVQ tail from Teacher-generated rollouts.

Seed-TTS-Eval supplies voice-cloning conditions only: reference audio,
reference text, and synthesis text. The frozen Teacher generates every output
codec frame. No target waveform is resolved, loaded, or encoded.

At each Student tail step, the preceding codec is the Teacher-emitted codec
from the same rollout step. Teacher traces are built once and cached, so no
Teacher forward occurs inside the epoch loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta
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
TRACE_FORMAT = "qwen3_tts_teacher_rollout_trace_v1"
LOCALE_TO_LANGUAGE = {"en": "English", "zh": "Chinese"}


@dataclasses.dataclass(frozen=True)
class SeedTTSRow:
    """One Seed-TTS voice-cloning condition."""

    utterance_id: str
    locale: str
    ref_text: str
    target_text: str
    ref_audio: Path
    meta_line: str


@dataclasses.dataclass(frozen=True)
class TeacherTrace:
    """CPU-resident frame-level Teacher trajectory."""

    exit_hidden: torch.Tensor
    teacher_tail_hidden: torch.Tensor
    teacher_codes: torch.Tensor
    row_offsets: torch.Tensor

    @property
    def frames(self) -> int:
        """Return the generated frame count."""
        return int(self.teacher_codes.shape[0])


@dataclasses.dataclass(frozen=True)
class TailTeacherWeights:
    """Frozen CP tensors retained after releasing the full Teacher."""

    previous_embedding_weight: torch.Tensor
    projection_weight: torch.Tensor | None
    projection_bias: torch.Tensor | None
    tail_lm_head_weight: torch.Tensor


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locales", nargs="+", choices=("en", "zh"), default=("en", "zh"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--trace-cache-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--trace-extraction-frame-batch-size", type=int, default=4096)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Student frame batch size, not an utterance batch size.",
    )
    parser.add_argument("--validation-batch-size", type=int, default=4096)
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
        default=1.0,
        help="CE against Teacher-emitted tail codecs.",
    )
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=0.1)
    parser.add_argument("--validation-rows-per-locale", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--rollout-do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rollout-top-k", type=int, default=50)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-temperature", type=float, default=0.9)
    parser.add_argument("--rollout-repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--rollout-subtalker-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Greedy by default, making Teacher-emitted residuals Teacher top-1.",
    )
    parser.add_argument("--rollout-subtalker-top-k", type=int, default=50)
    parser.add_argument("--rollout-subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--rollout-non-streaming-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def _normalize_config_value(value: Any) -> Any:
    """Convert values into stable JSON data."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_config_value(item) for key, item in sorted(value.items())}
    return value


def _build_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return the canonical run configuration."""
    arguments = {
        name: _normalize_config_value(value)
        for name, value in sorted(vars(args).items())
        if name != "output_dir"
    }
    return {
        "schema_version": 2,
        "distillation_paradigm": "teacher_rollout_teacher_conditioned_tail",
        "dataset_seed": DATASET_SEED,
        "train_fraction": TRAIN_FRACTION,
        "arguments": arguments,
    }


def _configuration_hash(configuration: dict[str, Any]) -> str:
    """Return a stable short SHA-256 digest."""
    canonical = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _create_run_directory(
    output_root: Path,
    config_hash: str,
    *,
    started_at: datetime | None = None,
) -> Path:
    """Create a unique timestamp-and-configuration run directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = started_at or datetime.now().astimezone()
    for collision_offset in range(24 * 60 * 60):
        candidate_time = timestamp + timedelta(seconds=collision_offset)
        candidate = output_root / (
            f"rvq_{candidate_time.strftime('%Y%m%d-%H%M%S')}_{config_hash}"
        )
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not allocate a run directory below {output_root}.")


def _parse_meta_line(root: Path, locale: str, line: str) -> SeedTTSRow | None:
    """Parse a metadata row without resolving target audio."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split("|")
    if len(parts) < 4:
        raise ValueError(f"Malformed {locale}/meta.lst row: {stripped[:160]!r}")
    utterance_id, ref_text, ref_rel, target_text = (part.strip() for part in parts[:4])
    ref_audio = Path(ref_rel)
    if not ref_audio.is_absolute():
        ref_audio = root / locale / ref_audio
    if not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio is missing: {ref_audio}")
    return SeedTTSRow(
        utterance_id=utterance_id,
        locale=locale,
        ref_text=ref_text,
        target_text=target_text,
        ref_audio=ref_audio.resolve(),
        meta_line=stripped,
    )


def _load_locale_rows(dataset_root: Path, locale: str) -> list[SeedTTSRow]:
    """Load and seed-42 shuffle one locale."""
    meta_path = dataset_root / locale / "meta.lst"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Seed-TTS metadata is missing: {meta_path}")
    rows = [
        row
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if (row := _parse_meta_line(dataset_root, locale, line)) is not None
    ]
    random.Random(DATASET_SEED).shuffle(rows)
    if len(rows) < 2:
        raise ValueError(f"Not enough rows in {meta_path}.")
    return rows


def _load_training_rows(dataset_root: Path, locales: Sequence[str]) -> list[SeedTTSRow]:
    """Select the deterministic first half per locale."""
    selected: list[SeedTTSRow] = []
    for locale in locales:
        rows = _load_locale_rows(dataset_root, locale)
        split = math.floor(len(rows) * TRAIN_FRACTION)
        selected.extend(rows[:split])
        print(f"Selected {split}/{len(rows)} {locale} Teacher-rollout rows.", flush=True)
    return selected


def _load_validation_rows(
    dataset_root: Path,
    locales: Sequence[str],
    rows_per_locale: int,
) -> list[SeedTTSRow]:
    """Select rows from the deterministic complementary half."""
    selected: list[SeedTTSRow] = []
    for locale in locales:
        rows = _load_locale_rows(dataset_root, locale)
        split = math.floor(len(rows) * TRAIN_FRACTION)
        locale_rows = rows[split : split + rows_per_locale]
        if not locale_rows:
            raise ValueError(f"No validation rows for {locale}.")
        selected.extend(locale_rows)
        print(f"Selected {len(locale_rows)} held-out {locale} rollout rows.", flush=True)
    return selected


def _write_manifest(
    rows: Sequence[SeedTTSRow],
    destination: Path,
    *,
    split_name: str,
) -> None:
    """Persist exact Teacher-rollout conditions."""
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        "utterance_id": row.utterance_id,
                        "locale": row.locale,
                        "language": LOCALE_TO_LANGUAGE[row.locale],
                        "ref_text": row.ref_text,
                        "target_text": row.target_text,
                        "ref_audio": str(row.ref_audio),
                        "split": split_name,
                        "seed": DATASET_SEED,
                        "supervision_source": "teacher_rollout",
                        "target_audio_used": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _batches(values: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield contiguous batches."""
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def _trace_cache_path(output_dir: Path, split_name: str) -> Path:
    """Return the run-local trace cache path."""
    trace_dir = output_dir / "teacher_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{split_name}.safetensors"


def _trace_descriptor_path(cache_path: Path) -> Path:
    """Return a trace descriptor path."""
    return cache_path.with_suffix(".json")


def _write_trace_descriptor(
    cache_path: Path,
    fingerprint: str,
    *,
    rows: int,
    frames: int,
) -> None:
    """Write a completed trace descriptor atomically."""
    destination = _trace_descriptor_path(cache_path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "format": TRACE_FORMAT,
                "fingerprint": fingerprint,
                "file": cache_path.name,
                "rows": rows,
                "frames": frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _model_fingerprint_files(model_path: Path) -> list[Path]:
    """Return checkpoint files that influence rollouts."""
    if model_path.is_file():
        return [model_path]
    patterns = (
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "speech_tokenizer/*.json",
        "speech_tokenizer/*.safetensors",
    )
    return sorted({path for pattern in patterns for path in model_path.glob(pattern)})


def _trace_fingerprint(
    model_path: Path,
    rows: Sequence[SeedTTSRow],
    *,
    split_seed: int,
    num_groups: int,
    args: argparse.Namespace,
) -> str:
    """Hash Teacher files, rollout conditions, and sampling configuration."""
    rollout_config = {
        "format": TRACE_FORMAT,
        "split_seed": split_seed,
        "num_groups": num_groups,
        "truncation_k": args.truncation_k,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_do_sample": args.rollout_do_sample,
        "rollout_top_k": args.rollout_top_k,
        "rollout_top_p": args.rollout_top_p,
        "rollout_temperature": args.rollout_temperature,
        "rollout_repetition_penalty": args.rollout_repetition_penalty,
        "rollout_subtalker_do_sample": args.rollout_subtalker_do_sample,
        "rollout_subtalker_top_k": args.rollout_subtalker_top_k,
        "rollout_subtalker_top_p": args.rollout_subtalker_top_p,
        "rollout_subtalker_temperature": args.rollout_subtalker_temperature,
        "rollout_max_new_tokens": args.rollout_max_new_tokens,
        "rollout_non_streaming_mode": args.rollout_non_streaming_mode,
        "trace_cache_dtype": args.trace_cache_dtype,
    }
    digest = hashlib.sha256(json.dumps(rollout_config, sort_keys=True).encode("utf-8"))
    for path in _model_fingerprint_files(model_path):
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    for row in rows:
        digest.update(row.meta_line.encode("utf-8"))
        stat = row.ref_audio.stat()
        digest.update(str(row.ref_audio).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _reuse_prior_trace(
    output_root: Path,
    run_directory: Path,
    cache_path: Path,
    fingerprint: str,
) -> None:
    """Reuse a content-identical trace from an earlier run."""
    pattern = f"rvq_*/teacher_trace/{cache_path.with_suffix('.json').name}"
    for descriptor_path in sorted(output_root.glob(pattern), reverse=True):
        if run_directory in descriptor_path.parents:
            continue
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = descriptor_path.with_suffix(".safetensors")
        if (
            descriptor.get("format") != TRACE_FORMAT
            or descriptor.get("fingerprint") != fingerprint
            or not source.is_file()
        ):
            continue
        try:
            os.link(source, cache_path)
            method = "hardlink"
        except OSError:
            temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(cache_path)
            method = "copy"
        _write_trace_descriptor(
            cache_path,
            fingerprint,
            rows=int(descriptor["rows"]),
            frames=int(descriptor["frames"]),
        )
        print(
            json.dumps(
                {
                    "record_type": "teacher_trace_reused",
                    "source": str(source),
                    "path": str(cache_path),
                    "method": method,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return


def _validate_trace(
    trace: TeacherTrace,
    *,
    rows: int,
    num_groups: int,
    truncation_k: int,
) -> None:
    """Validate trace tensor dimensions."""
    frames = trace.frames
    tail_steps = num_groups - truncation_k
    hidden_size = int(trace.exit_hidden.shape[-1])
    if trace.exit_hidden.ndim != 2:
        raise ValueError("exit_hidden must be rank two.")
    if trace.teacher_tail_hidden.shape != (frames, tail_steps, hidden_size):
        raise ValueError(f"Invalid tail hidden shape {tuple(trace.teacher_tail_hidden.shape)}.")
    if trace.teacher_codes.shape != (frames, num_groups):
        raise ValueError(f"Invalid Teacher code shape {tuple(trace.teacher_codes.shape)}.")
    if trace.row_offsets.shape != (rows + 1,):
        raise ValueError("Invalid row offset count.")
    if int(trace.row_offsets[0]) != 0 or int(trace.row_offsets[-1]) != frames:
        raise ValueError("Row offsets do not span trace frames.")
    if bool((trace.row_offsets[1:] < trace.row_offsets[:-1]).any()):
        raise ValueError("Row offsets must be non-decreasing.")


def _load_trace(
    cache_path: Path,
    fingerprint: str,
    *,
    rows: int,
    num_groups: int,
    truncation_k: int,
) -> TeacherTrace | None:
    """Load a matching trace or return None."""
    descriptor_path = _trace_descriptor_path(cache_path)
    if not cache_path.is_file() or not descriptor_path.is_file():
        return None
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if descriptor.get("format") != TRACE_FORMAT or descriptor.get("fingerprint") != fingerprint:
        return None
    tensors = load_file(str(cache_path), device="cpu")
    expected = {"exit_hidden", "teacher_tail_hidden", "teacher_codes", "row_offsets"}
    if set(tensors) != expected:
        return None
    trace = TeacherTrace(**tensors)
    _validate_trace(trace, rows=rows, num_groups=num_groups, truncation_k=truncation_k)
    print(
        json.dumps(
            {
                "record_type": "teacher_trace_hit",
                "path": str(cache_path),
                "rows": rows,
                "frames": trace.frames,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return trace


def _wrapper_tokenize(wrapper: Any, texts: list[str]) -> list[torch.Tensor]:
    """Tokenize through the official wrapper with a compatibility fallback."""
    method = getattr(wrapper, "_tokenize_texts", None)
    if callable(method):
        return method(texts)
    device = next(wrapper.model.parameters()).device
    result: list[torch.Tensor] = []
    for text_value in texts:
        ids = wrapper.processor(text=text_value, return_tensors="pt", padding=True)["input_ids"]
        result.append((ids.unsqueeze(0) if ids.ndim == 1 else ids).to(device))
    return result


def _assistant_text(wrapper: Any, value: str) -> str:
    """Build the official synthesis text prompt."""
    method = getattr(wrapper, "_build_assistant_text", None)
    if callable(method):
        return method(value)
    return f"<|im_start|>assistant\n{value}<|im_end|>\n<|im_start|>assistant\n"


def _reference_text(wrapper: Any, value: str) -> str:
    """Build the official reference text prompt."""
    method = getattr(wrapper, "_build_ref_text", None)
    if callable(method):
        return method(value)
    return f"<|im_start|>assistant\n{value}<|im_end|>\n"


def _prompt_items_to_dict(wrapper: Any, items: list[Any]) -> dict[str, Any]:
    """Convert official prompt items for the low-level generate API."""
    method = getattr(wrapper, "_prompt_items_to_voice_clone_prompt", None)
    if callable(method):
        return method(items)
    return {
        "ref_code": [item.ref_code for item in items],
        "ref_spk_embedding": [item.ref_spk_embedding for item in items],
        "x_vector_only_mode": [item.x_vector_only_mode for item in items],
        "icl_mode": [item.icl_mode for item in items],
    }


def _rollout_generate_kwargs(wrapper: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Build Teacher sampling arguments."""
    values = {
        "do_sample": args.rollout_do_sample,
        "top_k": args.rollout_top_k,
        "top_p": args.rollout_top_p,
        "temperature": args.rollout_temperature,
        "repetition_penalty": args.rollout_repetition_penalty,
        "subtalker_dosample": args.rollout_subtalker_do_sample,
        "subtalker_top_k": args.rollout_subtalker_top_k,
        "subtalker_top_p": args.rollout_subtalker_top_p,
        "subtalker_temperature": args.rollout_subtalker_temperature,
        "max_new_tokens": args.rollout_max_new_tokens,
    }
    method = getattr(wrapper, "_merge_generate_kwargs", None)
    return method(**values) if callable(method) else values


def _generate_teacher_rollout(
    wrapper: Any,
    rows: Sequence[SeedTTSRow],
    args: argparse.Namespace,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate full Teacher codecs and causal Talker hidden states."""
    prompt_items = wrapper.create_voice_clone_prompt(
        ref_audio=[str(row.ref_audio) for row in rows],
        ref_text=[row.ref_text for row in rows],
        x_vector_only_mode=False,
    )
    voice_prompt = _prompt_items_to_dict(wrapper, prompt_items)
    input_ids = _wrapper_tokenize(
        wrapper,
        [_assistant_text(wrapper, row.target_text) for row in rows],
    )
    ref_ids = _wrapper_tokenize(
        wrapper,
        [_reference_text(wrapper, row.ref_text) for row in rows],
    )
    with torch.inference_mode():
        codes, talker_hidden = wrapper.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_prompt,
            languages=[LOCALE_TO_LANGUAGE[row.locale] for row in rows],
            non_streaming_mode=args.rollout_non_streaming_mode,
            **_rollout_generate_kwargs(wrapper, args),
        )
    if len(codes) != len(rows) or len(talker_hidden) != len(rows):
        raise RuntimeError("Teacher generation returned an unexpected batch size.")
    return list(codes), list(talker_hidden)


def _normalize_rollout_pair(
    codes: torch.Tensor,
    hidden: torch.Tensor,
    *,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one rollout to frame-major tensors."""
    codes = torch.as_tensor(codes, dtype=torch.long, device=hidden.device)
    if codes.ndim != 2:
        raise ValueError(f"Teacher codes must be rank two, got {tuple(codes.shape)}.")
    if codes.shape[-1] != num_groups and codes.shape[0] == num_groups:
        codes = codes.transpose(0, 1)
    if codes.shape[-1] != num_groups:
        raise ValueError(f"Teacher generated {codes.shape[-1]} codebooks, expected {num_groups}.")
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2:
        raise ValueError(f"Talker hidden must be rank two, got {tuple(hidden.shape)}.")
    frames = min(int(codes.shape[0]), int(hidden.shape[0]))
    if frames < 1:
        raise ValueError("Teacher generated no usable frames.")
    return codes[:frames].contiguous(), hidden[:frames].contiguous()


def _extract_teacher_trace_chunk(
    talker: torch.nn.Module,
    codes: torch.Tensor,
    talker_hidden: torch.Tensor,
    *,
    truncation_k: int,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay CP once on Teacher-generated codes and extract tail targets."""
    code_predictor = talker.code_predictor
    num_groups = int(code_predictor.config.num_code_groups)
    inputs = [talker_hidden.unsqueeze(1), talker.get_input_embeddings()(codes[:, :1])]
    inputs.extend(
        code_predictor.get_input_embeddings()[codebook - 1](
            codes[:, codebook : codebook + 1]
        )
        for codebook in range(1, num_groups - 1)
    )
    projected = code_predictor.small_to_mtp_projection(torch.cat(inputs, dim=1))
    with torch.inference_mode():
        hidden = code_predictor.model(
            inputs_embeds=projected,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state
    if hidden.shape[1] != num_groups:
        raise RuntimeError(f"CP replay sequence is {hidden.shape[1]}, expected {num_groups}.")
    return (
        hidden[:, truncation_k - 1].to(dtype=cache_dtype).cpu().contiguous(),
        hidden[:, truncation_k:].to(dtype=cache_dtype).cpu().contiguous(),
        codes.cpu().contiguous(),
    )


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a CLI dtype."""
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _materialize_teacher_trace(
    wrapper: Any,
    rows: Sequence[SeedTTSRow],
    *,
    cache_path: Path,
    fingerprint: str,
    split_seed: int,
    args: argparse.Namespace,
) -> TeacherTrace:
    """Generate and atomically cache a split's Teacher trajectory."""
    talker = wrapper.model.talker
    num_groups = int(talker.code_predictor.config.num_code_groups)
    cache_dtype = _resolve_dtype(args.trace_cache_dtype)
    torch.manual_seed(split_seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    exit_parts: list[torch.Tensor] = []
    hidden_parts: list[torch.Tensor] = []
    code_parts: list[torch.Tensor] = []
    row_offsets = [0]
    rollout_batches = list(_batches(rows, args.rollout_batch_size))
    started_at = time.perf_counter()
    for rollout_index, batch_rows in enumerate(rollout_batches):
        generated_codes, generated_hidden = _generate_teacher_rollout(wrapper, batch_rows, args)
        normalized = [
            _normalize_rollout_pair(codes, hidden, num_groups=num_groups)
            for codes, hidden in zip(generated_codes, generated_hidden)
        ]
        for codes, _ in normalized:
            row_offsets.append(row_offsets[-1] + int(codes.shape[0]))
        batch_codes = torch.cat([pair[0] for pair in normalized], dim=0)
        batch_hidden = torch.cat([pair[1] for pair in normalized], dim=0)
        frame_batch = args.trace_extraction_frame_batch_size
        for start in range(0, batch_codes.shape[0], frame_batch):
            stop = min(start + frame_batch, batch_codes.shape[0])
            exit_hidden, tail_hidden, codes = _extract_teacher_trace_chunk(
                talker,
                batch_codes[start:stop],
                batch_hidden[start:stop],
                truncation_k=args.truncation_k,
                cache_dtype=cache_dtype,
            )
            exit_parts.append(exit_hidden)
            hidden_parts.append(tail_hidden)
            code_parts.append(codes)
        elapsed = time.perf_counter() - started_at
        print(
            json.dumps(
                {
                    "record_type": "teacher_rollout_progress",
                    "batch": rollout_index + 1,
                    "batches": len(rollout_batches),
                    "rows": len(row_offsets) - 1,
                    "frames": row_offsets[-1],
                    "elapsed_seconds": elapsed,
                    "frames_per_second": row_offsets[-1] / max(elapsed, 1e-6),
                    "target_audio_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    trace = TeacherTrace(
        exit_hidden=torch.cat(exit_parts, dim=0),
        teacher_tail_hidden=torch.cat(hidden_parts, dim=0),
        teacher_codes=torch.cat(code_parts, dim=0),
        row_offsets=torch.tensor(row_offsets, dtype=torch.long),
    )
    _validate_trace(
        trace,
        rows=len(rows),
        num_groups=num_groups,
        truncation_k=args.truncation_k,
    )
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    save_file(
        {
            "exit_hidden": trace.exit_hidden,
            "teacher_tail_hidden": trace.teacher_tail_hidden,
            "teacher_codes": trace.teacher_codes,
            "row_offsets": trace.row_offsets,
        },
        str(temporary),
        metadata={
            "format": TRACE_FORMAT,
            "rows": str(len(rows)),
            "frames": str(trace.frames),
            "supervision": "teacher_rollout",
            "target_audio_used": "false",
        },
    )
    temporary.replace(cache_path)
    _write_trace_descriptor(
        cache_path,
        fingerprint,
        rows=len(rows),
        frames=trace.frames,
    )
    print(
        json.dumps(
            {
                "record_type": "teacher_trace_saved",
                "path": str(cache_path),
                "rows": len(rows),
                "frames": trace.frames,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return trace


def _prepare_trace(
    wrapper: Any,
    rows: Sequence[SeedTTSRow],
    *,
    output_root: Path,
    run_directory: Path,
    model_path: Path,
    split_name: str,
    split_seed: int,
    num_groups: int,
    args: argparse.Namespace,
) -> tuple[TeacherTrace, Path]:
    """Load, reuse, or generate one split's Teacher trace."""
    cache_path = _trace_cache_path(run_directory, split_name)
    fingerprint = _trace_fingerprint(
        model_path,
        rows,
        split_seed=split_seed,
        num_groups=num_groups,
        args=args,
    )
    _reuse_prior_trace(output_root, run_directory, cache_path, fingerprint)
    trace = _load_trace(
        cache_path,
        fingerprint,
        rows=len(rows),
        num_groups=num_groups,
        truncation_k=args.truncation_k,
    )
    if trace is None:
        trace = _materialize_teacher_trace(
            wrapper,
            rows,
            cache_path=cache_path,
            fingerprint=fingerprint,
            split_seed=split_seed,
            args=args,
        )
    return trace, cache_path


def _clone_tail_teacher_weights(
    code_predictor: torch.nn.Module,
    *,
    truncation_k: int,
    num_groups: int,
) -> TailTeacherWeights:
    """Clone the small frozen CP interface used during Student training."""
    embeddings = code_predictor.get_input_embeddings()
    previous_embedding_weight = torch.stack(
        [
            embeddings[codebook - 1].weight.detach().clone()
            for codebook in range(truncation_k - 1, num_groups - 1)
        ]
    ).contiguous()
    projection = code_predictor.small_to_mtp_projection
    projection_weight = getattr(projection, "weight", None)
    projection_bias = getattr(projection, "bias", None)
    tail_lm_head_weight = torch.stack(
        [
            code_predictor.lm_head[codebook - 1].weight.detach().clone()
            for codebook in range(truncation_k, num_groups)
        ]
    ).transpose(1, 2).contiguous()
    return TailTeacherWeights(
        previous_embedding_weight=previous_embedding_weight,
        projection_weight=(
            projection_weight.detach().clone().contiguous()
            if projection_weight is not None
            else None
        ),
        projection_bias=(
            projection_bias.detach().clone().contiguous()
            if projection_bias is not None
            else None
        ),
        tail_lm_head_weight=tail_lm_head_weight,
    )


def _project_previous_codes(
    previous_codes: torch.Tensor,
    weights: TailTeacherWeights,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Embed Teacher preceding codecs for all tail steps."""
    if previous_codes.shape[1] != weights.previous_embedding_weight.shape[0]:
        raise ValueError("Teacher previous-code count does not match tail embedding tables.")
    embeddings = torch.stack(
        [
            F.embedding(previous_codes[:, step], weights.previous_embedding_weight[step])
            for step in range(previous_codes.shape[1])
        ],
        dim=1,
    )
    if weights.projection_weight is not None:
        embeddings = F.linear(embeddings, weights.projection_weight, weights.projection_bias)
    return embeddings.to(dtype=dtype)


def _project_one_code(
    code: torch.Tensor,
    weights: TailTeacherWeights,
    *,
    tail_offset: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Embed one preceding codec for Student free-running validation."""
    embedding = F.embedding(code, weights.previous_embedding_weight[tail_offset])
    if weights.projection_weight is not None:
        embedding = F.linear(embedding, weights.projection_weight, weights.projection_bias)
    return embedding.to(dtype=dtype)


def _student_teacher_conditioned_logits(
    student: RVQTailDistillationModel,
    weights: TailTeacherWeights,
    *,
    exit_hidden: torch.Tensor,
    previous_embeddings: torch.Tensor,
    truncation_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Student AR while conditioning each step on Teacher history."""
    student_dtype = next(student.parameters()).dtype
    state, anchor = student.initialize(exit_hidden.to(dtype=student_dtype))
    predicted_hidden: list[torch.Tensor] = []
    for tail_offset in range(previous_embeddings.shape[1]):
        hidden, state = student.predict_next(
            state,
            anchor,
            previous_embeddings[:, tail_offset].to(dtype=student_dtype),
            truncation_k + tail_offset,
        )
        predicted_hidden.append(hidden)
    stacked = torch.stack(predicted_hidden, dim=1)
    logits = torch.bmm(
        stacked.transpose(0, 1).to(dtype=weights.tail_lm_head_weight.dtype),
        weights.tail_lm_head_weight,
    ).transpose(0, 1).float()
    return logits, stacked.float()


def _distillation_loss(
    *,
    student: RVQTailDistillationModel,
    weights: TailTeacherWeights,
    exit_hidden: torch.Tensor,
    teacher_tail_hidden: torch.Tensor,
    teacher_codes: torch.Tensor,
    truncation_k: int,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Teacher-code CE, Teacher-distribution KL, and hidden loss."""
    tail_codes = teacher_codes[:, truncation_k:]
    previous_codes = teacher_codes[:, truncation_k - 1 : -1]
    student_dtype = next(student.parameters()).dtype
    previous_embeddings = _project_previous_codes(
        previous_codes,
        weights,
        dtype=student_dtype,
    )
    student_logits, student_hidden = _student_teacher_conditioned_logits(
        student,
        weights,
        exit_hidden=exit_hidden,
        previous_embeddings=previous_embeddings,
        truncation_k=truncation_k,
    )
    with torch.no_grad():
        teacher_logits = torch.bmm(
            teacher_tail_hidden.transpose(0, 1).to(
                dtype=weights.tail_lm_head_weight.dtype
            ),
            weights.tail_lm_head_weight,
        ).transpose(0, 1).float()

    flat_student = student_logits.flatten(0, 1)
    flat_teacher = teacher_logits.flatten(0, 1)
    flat_codes = tail_codes.flatten()
    code_ce = F.cross_entropy(flat_student, flat_codes)
    teacher_code_ce = F.cross_entropy(flat_teacher, flat_codes)
    kl = F.kl_div(
        F.log_softmax(flat_student / temperature, dim=-1),
        F.softmax(flat_teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    teacher_hidden = teacher_tail_hidden.float()
    hidden_rms = teacher_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    hidden = F.mse_loss(student_hidden / hidden_rms, teacher_hidden / hidden_rms)
    loss = ce_weight * code_ce + kl_weight * kl + hidden_weight * hidden

    with torch.no_grad():
        student_top1 = student_logits.argmax(dim=-1)
        teacher_top1 = teacher_logits.argmax(dim=-1)
        code_agreement = student_top1 == tail_codes
        distribution_agreement = student_top1 == teacher_top1
        teacher_code_top1 = teacher_top1 == tail_codes
        hidden_similarity = F.cosine_similarity(
            student_hidden,
            teacher_hidden,
            dim=-1,
        ).mean()
    metrics = {
        "loss": loss.detach(),
        "ce_teacher_code": code_ce.detach(),
        "teacher_code_ce": teacher_code_ce.detach(),
        "kl_teacher_distribution": kl.detach(),
        "hidden": hidden.detach(),
        "hidden_cosine_similarity": hidden_similarity.detach(),
        "hidden_cosine_distance": (1.0 - hidden_similarity).detach(),
        "student_teacher_top1": distribution_agreement.float().mean(),
        "student_teacher_code_top1": code_agreement.float().mean(),
        "student_teacher_tail_exact": code_agreement.all(dim=1).float().mean(),
        "teacher_code_top1": teacher_code_top1.float().mean(),
    }
    for tail_offset in range(tail_codes.shape[1]):
        codebook = truncation_k + tail_offset
        metrics[f"student_teacher_code_top1_codebook_{codebook}"] = (
            code_agreement[:, tail_offset].float().mean()
        )
    return loss, metrics


def _free_running_metrics(
    *,
    student: RVQTailDistillationModel,
    weights: TailTeacherWeights,
    exit_hidden: torch.Tensor,
    teacher_codes: torch.Tensor,
    truncation_k: int,
) -> dict[str, torch.Tensor]:
    """Measure Student self-conditioning against the Teacher trajectory."""
    student_dtype = next(student.parameters()).dtype
    state, anchor = student.initialize(exit_hidden.to(dtype=student_dtype))
    previous_code = teacher_codes[:, truncation_k - 1]
    generated: list[torch.Tensor] = []
    tail_steps = teacher_codes.shape[1] - truncation_k
    for tail_offset in range(tail_steps):
        previous_embedding = _project_one_code(
            previous_code,
            weights,
            tail_offset=tail_offset,
            dtype=student_dtype,
        )
        hidden, state = student.predict_next(
            state,
            anchor,
            previous_embedding,
            truncation_k + tail_offset,
        )
        logits = F.linear(
            hidden.to(dtype=weights.tail_lm_head_weight.dtype),
            weights.tail_lm_head_weight[tail_offset].transpose(0, 1),
        )
        previous_code = logits.argmax(dim=-1)
        generated.append(previous_code)
    generated_codes = torch.stack(generated, dim=1)
    expected = teacher_codes[:, truncation_k:]
    agreement = generated_codes == expected
    metrics = {
        "free_running_student_teacher_code_top1": agreement.float().mean(),
        "free_running_student_teacher_tail_exact": agreement.all(dim=1).float().mean(),
    }
    for tail_offset in range(tail_steps):
        codebook = truncation_k + tail_offset
        metrics[f"free_running_student_teacher_code_top1_codebook_{codebook}"] = (
            agreement[:, tail_offset].float().mean()
        )
    return metrics


def _metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer scalar metrics with one device synchronization."""
    names = list(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names]).cpu().tolist()
    return dict(zip(names, values))


def _teacher_trace_sanity(
    trace: TeacherTrace,
    weights: TailTeacherWeights,
    *,
    device: torch.device,
    batch_size: int,
    truncation_k: int,
) -> dict[str, float]:
    """Check that replayed Teacher logits reproduce Teacher-emitted codecs."""
    tail_steps = trace.teacher_codes.shape[1] - truncation_k
    matched = torch.zeros(tail_steps, dtype=torch.long, device=device)
    exact = torch.zeros((), dtype=torch.long, device=device)
    for start in range(0, trace.frames, batch_size):
        stop = min(start + batch_size, trace.frames)
        indices = torch.arange(start, stop, dtype=torch.long)
        tail_hidden = trace.teacher_tail_hidden.index_select(0, indices).to(device)
        codes = trace.teacher_codes.index_select(0, indices).to(device)
        logits = torch.bmm(
            tail_hidden.transpose(0, 1).to(dtype=weights.tail_lm_head_weight.dtype),
            weights.tail_lm_head_weight,
        ).transpose(0, 1)
        agreement = logits.argmax(dim=-1) == codes[:, truncation_k:]
        matched.add_(agreement.sum(dim=0))
        exact.add_(agreement.all(dim=1).sum())
    per_codebook = matched.float() / trace.frames
    result = {
        "teacher_code_top1": float(per_codebook.mean().cpu()),
        "teacher_tail_exact": float((exact.float() / trace.frames).cpu()),
    }
    for tail_offset, value in enumerate(per_codebook.cpu().tolist()):
        result[f"teacher_code_top1_codebook_{truncation_k + tail_offset}"] = value
    return result


def _frame_indices(frames: int, batch_size: int, *, seed: int) -> list[torch.Tensor]:
    """Build deterministic shuffled CPU frame batches."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(frames, generator=generator)
    return [
        permutation[start : start + batch_size]
        for start in range(0, frames, batch_size)
    ]


def _trace_batch(
    trace: TeacherTrace,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather a CPU trace batch and transfer it to the accelerator."""
    return (
        trace.exit_hidden.index_select(0, indices).to(device=device, non_blocking=True),
        trace.teacher_tail_hidden.index_select(0, indices).to(
            device=device,
            non_blocking=True,
        ),
        trace.teacher_codes.index_select(0, indices).to(device=device, non_blocking=True),
    )


def _accumulation_group_size(
    batch_index: int,
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the actual accumulation group size, including a partial tail."""
    group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
    return min(gradient_accumulation_steps, num_batches - group_start)


def _evaluate_distillation(
    *,
    student: RVQTailDistillationModel,
    weights: TailTeacherWeights,
    trace: TeacherTrace,
    device: torch.device,
    batch_size: int,
    truncation_k: int,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
    held_out_rows: int,
) -> dict[str, Any]:
    """Evaluate Teacher-conditioned fit and deployment-like self-conditioning."""
    index_batches = _frame_indices(trace.frames, batch_size, seed=DATASET_SEED)
    totals: dict[str, torch.Tensor] = {}
    started_at = time.perf_counter()
    was_training = student.training
    student.eval()
    try:
        with torch.inference_mode():
            for indices in index_batches:
                exit_hidden, tail_hidden, codes = _trace_batch(
                    trace,
                    indices,
                    device=device,
                )
                _, metrics = _distillation_loss(
                    student=student,
                    weights=weights,
                    exit_hidden=exit_hidden,
                    teacher_tail_hidden=tail_hidden,
                    teacher_codes=codes,
                    truncation_k=truncation_k,
                    temperature=temperature,
                    ce_weight=ce_weight,
                    kl_weight=kl_weight,
                    hidden_weight=hidden_weight,
                )
                metrics.update(
                    _free_running_metrics(
                        student=student,
                        weights=weights,
                        exit_hidden=exit_hidden,
                        teacher_codes=codes,
                        truncation_k=truncation_k,
                    )
                )
                frame_count = int(indices.numel())
                for name, value in metrics.items():
                    weighted = value * frame_count
                    totals[name] = totals[name] + weighted if name in totals else weighted
    finally:
        student.train(was_training)
    averaged = _metrics_to_floats(
        {name: total / trace.frames for name, total in totals.items()}
    )
    elapsed = time.perf_counter() - started_at
    return {
        "record_type": "validation_summary",
        "validation_mode": "teacher_rollout_teacher_conditioned_and_student_free_running",
        "supervision_source": "teacher_rollout",
        "target_audio_used": False,
        "held_out_rows": held_out_rows,
        "batches": len(index_batches),
        "frames": trace.frames,
        "elapsed_seconds": elapsed,
        "frames_per_second": trace.frames / max(elapsed, 1e-6),
        **averaged,
    }


def _save_checkpoint(
    student: RVQTailDistillationModel,
    output_dir: Path,
    filename: str,
    metadata: dict[str, str],
) -> Path:
    """Save an atomic Student checkpoint."""
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    save_file(
        {
            name: tensor.detach().float().cpu().contiguous()
            for name, tensor in student.state_dict().items()
        },
        str(temporary),
        metadata=metadata,
    )
    temporary.replace(destination)
    return destination


def _prepare_device_backend(device_name: str) -> None:
    """Import torch_npu when an NPU device is requested."""
    if not device_name.startswith("npu"):
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise ImportError("NPU distillation requires torch_npu.") from exc


def _empty_device_cache(device: torch.device) -> None:
    """Release cached allocations after deleting the frozen Teacher."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI values."""
    positive_ints = (
        "rollout_batch_size",
        "trace_extraction_frame_batch_size",
        "batch_size",
        "validation_batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "validation_rows_per_locale",
        "log_every",
        "rollout_max_new_tokens",
    )
    for name in positive_ints:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0 or args.temperature <= 0.0:
        raise ValueError("Learning rate, max grad norm, and KD temperature must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("Weight decay must be non-negative.")
    loss_weights = (args.ce_weight, args.kl_weight, args.hidden_weight)
    if any(value < 0.0 for value in loss_weights) or not any(
        value > 0.0 for value in loss_weights
    ):
        raise ValueError("Loss weights must be non-negative with at least one positive.")
    if not 0.0 < args.rollout_top_p <= 1.0:
        raise ValueError("rollout-top-p must be in (0, 1].")
    if not 0.0 < args.rollout_subtalker_top_p <= 1.0:
        raise ValueError("rollout-subtalker-top-p must be in (0, 1].")
    if args.rollout_temperature <= 0.0 or args.rollout_subtalker_temperature <= 0.0:
        raise ValueError("Rollout temperatures must be positive.")


def main() -> int:
    """Run Teacher-rollout, Teacher-conditioned tail distillation."""
    args = _build_parser().parse_args()
    _validate_args(args)
    _prepare_device_backend(args.device)

    model_path = args.model_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    configuration = _build_run_configuration(args)
    config_hash = _configuration_hash(configuration)
    output_dir = _create_run_directory(output_root, config_hash)
    run_id = output_dir.name
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "hash_algorithm": "sha256_first_12_hex",
                "configuration": configuration,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "record_type": "run_created",
                "run_id": run_id,
                "config_hash": config_hash,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_rows = _load_training_rows(dataset_root, args.locales)
    validation_rows = _load_validation_rows(
        dataset_root,
        args.locales,
        args.validation_rows_per_locale,
    )
    train_manifest_path = output_dir / "train_manifest.jsonl"
    validation_manifest_path = output_dir / "validation_manifest.jsonl"
    _write_manifest(train_rows, train_manifest_path, split_name="train")
    _write_manifest(validation_rows, validation_manifest_path, split_name="validation")

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
        raise ImportError("Install the official qwen-tts package for this checkpoint.") from exc

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
    code_predictor = teacher.talker.code_predictor
    num_groups = int(code_predictor.config.num_code_groups)
    hidden_size = int(code_predictor.config.hidden_size)
    if not 2 <= args.truncation_k < num_groups:
        raise ValueError(
            f"truncation-k must be in [2, {num_groups - 1}], got {args.truncation_k}."
        )
    device = next(teacher.parameters()).device

    train_trace, train_trace_path = _prepare_trace(
        wrapper,
        train_rows,
        output_root=output_root,
        run_directory=output_dir,
        model_path=model_path,
        split_name="train",
        split_seed=DATASET_SEED,
        num_groups=num_groups,
        args=args,
    )
    validation_trace, validation_trace_path = _prepare_trace(
        wrapper,
        validation_rows,
        output_root=output_root,
        run_directory=output_dir,
        model_path=model_path,
        split_name="validation",
        split_seed=DATASET_SEED + 10_000,
        num_groups=num_groups,
        args=args,
    )
    tail_weights = _clone_tail_teacher_weights(
        code_predictor,
        truncation_k=args.truncation_k,
        num_groups=num_groups,
    )
    trace_sanity = _teacher_trace_sanity(
        train_trace,
        tail_weights,
        device=device,
        batch_size=args.validation_batch_size,
        truncation_k=args.truncation_k,
    )
    print(
        json.dumps(
            {
                "record_type": "teacher_trace_sanity",
                "rollout_subtalker_do_sample": args.rollout_subtalker_do_sample,
                **trace_sanity,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if (
        not args.rollout_subtalker_do_sample
        and trace_sanity["teacher_code_top1"] < 0.99
    ):
        raise RuntimeError(
            "Greedy Teacher trace replay failed to reproduce Teacher-emitted tail "
            f"codecs (top1={trace_sanity['teacher_code_top1']:.6f}). This indicates "
            "a Talker/Code-Predictor alignment or qwen-tts version mismatch."
        )
    del code_predictor, teacher, wrapper
    gc.collect()
    _empty_device_cache(device)
    print(
        json.dumps(
            {
                "record_type": "teacher_released",
                "train_frames": train_trace.frames,
                "validation_frames": validation_trace.frames,
                "training_loop_teacher_forwards": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    student_dtype = _resolve_dtype(args.student_dtype)
    student = RVQTailDistillationModel(
        RVQTailDistillationConfig(
            hidden_size=hidden_size,
            state_size=args.state_size,
            num_code_groups=num_groups,
        )
    ).to(device=device, dtype=student_dtype)
    if args.resume_from is not None:
        state = load_file(str(args.resume_from.expanduser().resolve()), device="cpu")
        student.load_state_dict(state, strict=True)
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    metrics_path = output_dir / "metrics.jsonl"
    training_record = {
        "record_type": "training_config",
        "run_id": run_id,
        "config_hash": config_hash,
        "distillation_paradigm": "teacher_rollout_teacher_conditioned_tail",
        "supervision_source": "teacher_rollout",
        "target_audio_used": False,
        "teacher_previous_code_conditioning": True,
        "training_loop_teacher_forwards": 0,
        "teacher_trace_sanity": trace_sanity,
        "rows": len(train_rows),
        "frames": train_trace.frames,
        "truncation_k": args.truncation_k,
        "num_code_groups": num_groups,
        "hidden_size": hidden_size,
        "state_size": args.state_size,
        "batch_size_frames": args.batch_size,
        "rollout_batch_size_rows": args.rollout_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "kl_weight": args.kl_weight,
        "hidden_weight": args.hidden_weight,
        "teacher_trace": str(train_trace_path),
        "train_manifest": str(train_manifest_path),
        "trainable_parameters": sum(parameter.numel() for parameter in student.parameters()),
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(training_record, sort_keys=True) + "\n")
    print(json.dumps(training_record, sort_keys=True), flush=True)

    global_step = 0
    optimizer_step = 0
    interval_started = time.perf_counter()
    interval_frames = 0
    for epoch in range(args.epochs):
        index_batches = _frame_indices(
            train_trace.frames,
            args.batch_size,
            seed=DATASET_SEED + epoch,
        )
        epoch_totals: dict[str, torch.Tensor] = {}
        epoch_frames = 0
        for batch_index, indices in enumerate(index_batches):
            exit_hidden, tail_hidden, codes = _trace_batch(
                train_trace,
                indices,
                device=device,
            )
            loss, metrics = _distillation_loss(
                student=student,
                weights=tail_weights,
                exit_hidden=exit_hidden,
                teacher_tail_hidden=tail_hidden,
                teacher_codes=codes,
                truncation_k=args.truncation_k,
                temperature=args.temperature,
                ce_weight=args.ce_weight,
                kl_weight=args.kl_weight,
                hidden_weight=args.hidden_weight,
            )
            group_size = _accumulation_group_size(
                batch_index,
                len(index_batches),
                args.gradient_accumulation_steps,
            )
            (loss / group_size).backward()
            global_step += 1
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(index_batches)
            )
            should_log = False
            diagnostics: dict[str, torch.Tensor] = {}
            if should_step:
                next_optimizer_step = optimizer_step + 1
                should_log = next_optimizer_step == 1 or next_optimizer_step % args.log_every == 0
                probe = student.output_projection.weight
                probe_before = probe.detach().clone() if should_log else None
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                if should_log:
                    if probe.grad is None:
                        raise RuntimeError("Student output projection has no gradient.")
                    diagnostics["grad_norm"] = torch.as_tensor(grad_norm, device=device)
                    diagnostics["output_projection_grad_norm"] = probe.grad.detach().float().norm()
                optimizer.step()
                if probe_before is not None:
                    diagnostics["update_norm"] = (probe.detach() - probe_before).float().norm()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step = next_optimizer_step

            frame_count = int(indices.numel())
            epoch_frames += frame_count
            interval_frames += frame_count
            for name, value in metrics.items():
                weighted = value * frame_count
                epoch_totals[name] = (
                    epoch_totals[name] + weighted if name in epoch_totals else weighted
                )

            if should_log:
                scalar = _metrics_to_floats({**metrics, **diagnostics})
                if optimizer_step == 1 and any(
                    not math.isfinite(scalar[name]) or scalar[name] <= 0.0
                    for name in ("grad_norm", "output_projection_grad_norm", "update_norm")
                ):
                    raise FloatingPointError(
                        f"First optimizer step had no finite non-zero update: {scalar}"
                    )
                elapsed = time.perf_counter() - interval_started
                record = {
                    "record_type": "optimizer_step",
                    "run_id": run_id,
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "frames_per_second": interval_frames / max(elapsed, 1e-6),
                    **scalar,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                interval_started = time.perf_counter()
                interval_frames = 0

        epoch_metrics = _metrics_to_floats(
            {name: total / epoch_frames for name, total in epoch_totals.items()}
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
        _save_checkpoint(
            student,
            output_dir,
            f"qwen3_tts_rvq_tail_epoch_{epoch + 1}.safetensors",
            {
                "format": "qwen3_tts_rvq_tail_distillation_v3",
                "paradigm": "teacher_rollout_teacher_conditioned_tail",
                "run_id": run_id,
                "config_hash": config_hash,
                "truncation_k": str(args.truncation_k),
                "num_code_groups": str(num_groups),
                "hidden_size": str(hidden_size),
                "state_size": str(args.state_size),
                "epoch": str(epoch + 1),
            },
        )

    final_path = _save_checkpoint(
        student,
        output_dir,
        "qwen3_tts_rvq_tail.safetensors",
        {
            "format": "qwen3_tts_rvq_tail_distillation_v3",
            "paradigm": "teacher_rollout_teacher_conditioned_tail",
            "run_id": run_id,
            "config_hash": config_hash,
            "truncation_k": str(args.truncation_k),
            "num_code_groups": str(num_groups),
            "hidden_size": str(hidden_size),
            "state_size": str(args.state_size),
            "epochs": str(args.epochs),
        },
    )

    validation_summary = _evaluate_distillation(
        student=student,
        weights=tail_weights,
        trace=validation_trace,
        device=device,
        batch_size=args.validation_batch_size,
        truncation_k=args.truncation_k,
        temperature=args.temperature,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        hidden_weight=args.hidden_weight,
        held_out_rows=len(validation_rows),
    )
    validation_summary.update(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "checkpoint": str(final_path),
            "manifest": str(validation_manifest_path),
            "teacher_trace": str(validation_trace_path),
            "metric_directions": {
                "kl_teacher_distribution": "lower_is_better",
                "hidden_cosine_distance": "lower_is_better",
                "hidden_cosine_similarity": "higher_is_better",
                "student_teacher_code_top1": "higher_is_better",
                "student_teacher_tail_exact": "higher_is_better",
                "free_running_student_teacher_code_top1": "higher_is_better",
                "free_running_student_teacher_tail_exact": "higher_is_better",
            },
        }
    )
    validation_summary_path = output_dir / "validation_summary.json"
    validation_summary_path.write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(validation_summary, sort_keys=True) + "\n")
    print(json.dumps(validation_summary, sort_keys=True), flush=True)

    deployment_config = {
        "run_id": run_id,
        "config_hash": config_hash,
        "run_config": str(run_config_path),
        "code_predictor_truncation_mode": "fixed",
        "code_predictor_truncation_k": args.truncation_k,
        "code_predictor_early_exit_fill_strategy": "distillation",
        "code_predictor_distillation_state_size": args.state_size,
        "code_predictor_distillation_weights": str(final_path),
        "distillation_paradigm": "teacher_rollout_teacher_conditioned_tail",
        "target_audio_used": False,
        "metrics": str(metrics_path),
        "train_manifest": str(train_manifest_path),
        "validation_manifest": str(validation_manifest_path),
        "validation_summary": str(validation_summary_path),
        "validation": {
            "kl_teacher_distribution": validation_summary["kl_teacher_distribution"],
            "hidden_cosine_similarity": validation_summary["hidden_cosine_similarity"],
            "student_teacher_code_top1": validation_summary["student_teacher_code_top1"],
            "student_teacher_tail_exact": validation_summary["student_teacher_tail_exact"],
            "free_running_student_teacher_code_top1": validation_summary[
                "free_running_student_teacher_code_top1"
            ],
            "free_running_student_teacher_tail_exact": validation_summary[
                "free_running_student_teacher_tail_exact"
            ],
        },
    }
    config_path = output_dir / "distillation_config.json"
    config_path.write_text(
        json.dumps(deployment_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved run={run_id} directory={output_dir} checkpoint={final_path} "
        f"validation={validation_summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
