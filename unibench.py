#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run an audio-level Qwen3-TTS early-exit experiment against a live server.

This root-level entry point runs the existing Seed-TTS accuracy benchmark
against the OpenAI speech endpoint. Start one server per configuration first,
for example a baseline server with code predictor early exit disabled and a
variant server whose stage config enables ``code_predictor_early_exit_*``
options. Then run this script once per server with a different ``--label``.

Example:

    python run_qwen3_tts_early_exit_audio_experiment.py \\
        --label entropy_090_prior \\
        --host 127.0.0.1 \\
        --port 8000 \\
        --model Qwen/Qwen3-TTS \\
        --seed-tts-dataset-path /path/to/seed-tts-eval \\
        --num-prompts 200 \\
        --summary-jsonl results/qwen3_tts_early_exit_audio.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from datetime import datetime
from pathlib import Path
from typing import Any

from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    find_vllm_cli,
    load_benchmark_result,
    run_vllm_bench_subprocess,
)

DEFAULT_SEED_TTS_DATASET = "./seed-tts-eval"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Experiment label, e.g. baseline or entropy_090_prior.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True, help="Model id/name exposed by the running server.")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset shuffle seed forwarded to vllm bench serve. Keep fixed for A/B comparisons.",
    )
    parser.add_argument("--seed-tts-dataset-path", default=None)
    parser.add_argument("--seed-tts-root", type=Path, default=None)
    parser.add_argument("--seed-tts-locale", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--seed-tts-wer-eval",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set SEED_TTS_WER_EVAL for the benchmark subprocess.",
    )
    parser.add_argument(
        "--seed-tts-sim-eval",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set SEED_TTS_SIM_EVAL for the benchmark subprocess.",
    )
    parser.add_argument(
        "--seed-tts-utmos-eval",
        type=int,
        choices=(0, 1),
        default=0,
        help="Set SEED_TTS_UTMOS_EVAL for the benchmark subprocess.",
    )
    parser.add_argument("--seed-tts-eval-device", default=None)
    parser.add_argument(
        "--seed-tts-whisper-model",
        default=None,
        help="HF id or local directory for English WER ASR. Sets SEED_TTS_HF_WHISPER_MODEL.",
    )
    parser.add_argument(
            "--seed-tts-paraformer-model",
            default=None,
            help="HF id or local directory for Chinese WER ASR. Sets SEED_TTS_PARAFORMER_MODEL.",
        )
    parser.add_argument(
        "--seed-tts-sim-device",
        default=None,
        help="Device for WavLM SIM scoring, e.g. cpu, cuda:0, npu:0. Sets SEED_TTS_SIM_DEVICE.",
    )
    parser.add_argument(
        "--seed-tts-wavlm-model",
        default=None,
        help="HF id or local directory for WavLM SIM embeddings. Sets SEED_TTS_WAVLM_MODEL.",
    )
    parser.add_argument(
        "--seed-tts-utmos-device",
        default=None,
        help="Device for UTMOS scoring, e.g. cpu, cuda:0, npu:0. Sets SEED_TTS_UTMOS_DEVICE.",
    )
    parser.add_argument(
        "--seed-tts-utmos-model",
        default=None,
        help=(
            "UTMOS model source. Accepts a local .jit file path, a local HF-style repo directory, "
            "or a Hugging Face repo id. Sets SEED_TTS_UTMOS_JIT_PATH or SEED_TTS_UTMOS_HF_REPO."
        ),
    )
    parser.add_argument("--seed-tts-wer-save-items", action="store_true")
    parser.add_argument("--seed-tts-file-ref-audio", action="store_true")
    parser.add_argument(
        "--seed-extra-body-json",
        default="{}",
        help="Forwarded to vllm bench serve as --extra-body.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/qwen3_tts_early_exit_audio"),
        help="Directory for full benchmark result JSON files.",
    )
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=Path("results/qwen3_tts_early_exit_audio/summary.jsonl"),
        help="Append one compact result row here.",
    )
    parser.add_argument(
        "--max-seed-tts-mean-wer",
        type=float,
        default=100.0,
        help="Forwarded validation ceiling. Defaults high so exploratory sweeps do not fail early.",
    )
    parser.add_argument("--min-seed-tts-mean-sim", type=float, default=None)
    parser.add_argument("--min-seed-tts-mean-utmos", type=float, default=None)
    return parser


def _latest_seed_result(result_dir: Path, before: set[Path]) -> Path:
    candidates = sorted(
        (path for path in result_dir.glob("qwen_omni_acc_seed_tts_*.json") if path not in before),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No new Seed-TTS result JSON found in {result_dir}")
    return candidates[0]


def _summary_row(label: str, result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "timestamp": int(time.time()),
        "datatime": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "result_path": str(result_path),
        "seed_tts_content_evaluated": result.get("seed_tts_content_evaluated"),
        "seed_tts_content_error_mean": result.get("seed_tts_content_error_mean"),
        "seed_tts_content_error_median": result.get("seed_tts_content_error_median"),
        "seed_tts_sim_evaluated": result.get("seed_tts_sim_evaluated"),
        "seed_tts_sim_mean": result.get("seed_tts_sim_mean"),
        "seed_tts_sim_median": result.get("seed_tts_sim_median"),
        "seed_tts_sim_failed": result.get("seed_tts_sim_failed"),
        "seed_tts_sim_skipped_no_ref": result.get("seed_tts_sim_skipped_no_ref"),
        "seed_tts_utmos_mean": result.get("seed_tts_utmos_mean"),
        "seed_tts_utmos_median": result.get("seed_tts_utmos_median"),
        "mean_e2el_ms": result.get("mean_e2el_ms"),
        "mean_audio_ttfp_ms": result.get("mean_audio_ttfp_ms"),
        "mean_audio_rtf": result.get("mean_audio_rtf"),
        "request_throughput": result.get("request_throughput"),
    }


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def main() -> int:
    args = _build_parser().parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    before = set(args.result_dir.glob("qwen_omni_acc_seed_tts_*.json"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"qwen_omni_acc_seed_tts_{_safe_filename_token(args.label)}_{timestamp}.json"
    dataset_path = args.seed_tts_dataset_path or DEFAULT_SEED_TTS_DATASET

    bench_argv = [
        "bench",
        "serve",
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--endpoint",
        "/v1/audio/speech",
        "--backend",
        "openai-audio-speech",
        "--request-rate",
        "inf",
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.max_concurrency),
        "--no-oversample",
        "--num-warmups",
        str(args.num_warmups),
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "e2el,audio_ttfp,audio_rtf,audio_duration",
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        result_filename,
        "--dataset-name",
        "seed-tts",
        "--dataset-path",
        dataset_path,
        "--seed-tts-locale",
        args.seed_tts_locale,
        "--extra-body",
        args.seed_extra_body_json,
    ]
    if args.seed_tts_wer_eval:
        bench_argv.append("--seed-tts-wer-eval")
    if args.seed_tts_root:
        bench_argv.extend(["--seed-tts-root", str(args.seed_tts_root)])
    if args.seed_tts_wer_save_items:
        bench_argv.append("--seed-tts-wer-save-items")
    if args.seed_tts_file_ref_audio:
        bench_argv.append("--seed-tts-file-ref-audio")

    vllm = find_vllm_cli()
    print("\n$", vllm, *bench_argv, "\n", flush=True)
    extra_env = {
        "SEED_TTS_WER_EVAL": str(args.seed_tts_wer_eval),
        "SEED_TTS_SIM_EVAL": str(args.seed_tts_sim_eval),
        "SEED_TTS_UTMOS_EVAL": str(args.seed_tts_utmos_eval),
    }
    ascend_rt_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if ascend_rt_visible_devices is not None:
        extra_env["ASCEND_RT_VISIBLE_DEVICES"] = ascend_rt_visible_devices
    if args.seed_tts_eval_device:
        extra_env["SEED_TTS_EVAL_DEVICE"] = args.seed_tts_eval_device
    if args.seed_tts_whisper_model:
        extra_env["SEED_TTS_HF_WHISPER_MODEL"] = args.seed_tts_whisper_model
    if args.seed_tts_paraformer_model:
        extra_env["SEED_TTS_PARAFORMER_MODEL"] = args.seed_tts_paraformer_model
    if args.seed_tts_sim_device:
        extra_env["SEED_TTS_SIM_DEVICE"] = args.seed_tts_sim_device
    if args.seed_tts_wavlm_model:
        extra_env["SEED_TTS_WAVLM_MODEL"] = args.seed_tts_wavlm_model
    if args.seed_tts_utmos_device:
        extra_env["SEED_TTS_UTMOS_DEVICE"] = args.seed_tts_utmos_device
    if args.seed_tts_utmos_model:
        extra_env["SEED_TTS_UTMOS_HF_REPO"] = args.seed_tts_utmos_model
    run_vllm_bench_subprocess(vllm, bench_argv, extra_env=extra_env)
    result_path = _latest_seed_result(args.result_dir, before)
    result = load_benchmark_result(result_path)
    row = _summary_row(args.label, result_path, result)
    with args.summary_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("\nEarly-exit audio experiment summary:")
    print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
    failed: list[str] = []
    mean_wer = result.get("seed_tts_content_error_mean")
    if mean_wer is not None and float(mean_wer) > float(args.max_seed_tts_mean_wer):
        failed.append(f"WER {float(mean_wer):.6f} > {args.max_seed_tts_mean_wer:.6f}")
    sim = result.get("seed_tts_sim_mean")
    if args.min_seed_tts_mean_sim is not None and sim is not None:
        if float(sim) < float(args.min_seed_tts_mean_sim):
            failed.append(f"SIM {float(sim):.6f} < {args.min_seed_tts_mean_sim:.6f}")
    utmos = result.get("seed_tts_utmos_mean")
    if args.min_seed_tts_mean_utmos is not None and utmos is not None:
        if float(utmos) < float(args.min_seed_tts_mean_utmos):
            failed.append(f"UTMOS {float(utmos):.6f} < {args.min_seed_tts_mean_utmos:.6f}")
    if failed:
        for item in failed:
            print(f"ACCURACY CHECK FAILED: {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
