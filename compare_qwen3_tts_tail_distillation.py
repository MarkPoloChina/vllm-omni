#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare full and distilled Qwen3-TTS services on held-out Seed-TTS rows.

Start two vLLM-Omni services before running this script:

* baseline: ``code_predictor_truncation_mode: none``
* distilled: ``code_predictor_truncation_mode: fixed`` with the trained sidecar

For the selected locale, this script repeats the training split procedure
(seed 42, 50 percent) and evaluates only the complementary second half.  It
invokes the existing ``abse.py`` benchmark once per endpoint, then reports WER
and timing deltas from the two aligned runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_SEED = 42
TRAIN_FRACTION = 0.5


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI options for a two-endpoint A/B evaluation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name exposed by both running services.")
    parser.add_argument("--seed-tts-root", type=Path, required=True)
    parser.add_argument("--seed-tts-locale", choices=("en", "zh"), default="zh")
    parser.add_argument("--baseline-host", default="127.0.0.1")
    parser.add_argument("--baseline-port", type=int, required=True)
    parser.add_argument("--distilled-host", default="127.0.0.1")
    parser.add_argument("--distilled-port", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_tts_tail_distillation_ab"))
    parser.add_argument("--seed-tts-eval-device", default=None)
    parser.add_argument("--seed-tts-whisper-model", default=None)
    parser.add_argument("--seed-tts-sim-device", default=None)
    parser.add_argument("--seed-tts-wavlm-model", default=None)
    parser.add_argument("--seed-tts-file-ref-audio", action="store_true")
    parser.add_argument("--seed-extra-body-json", default="{}")
    return parser


def _valid_meta_lines(meta_path: Path) -> list[str]:
    """Load non-empty, well-formed Seed-TTS metadata rows."""
    lines: list[str] = []
    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if len(line.split("|")) < 4:
            raise ValueError(f"Malformed Seed-TTS metadata row: {line[:160]!r}")
        lines.append(line)
    if not lines:
        raise ValueError(f"No usable Seed-TTS rows found in {meta_path}")
    return lines


def _ensure_dataset_link(source: Path, destination: Path) -> None:
    """Create and validate a holdout-dataset directory symlink."""
    if not source.is_dir():
        raise FileNotFoundError(f"Seed-TTS audio directory is missing: {source}")
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise FileExistsError(f"Holdout path already exists with a different target: {destination}")
        return
    destination.symlink_to(source.resolve(), target_is_directory=True)


def _build_holdout_root(dataset_root: Path, locale: str, output_dir: Path) -> tuple[Path, int]:
    """Materialize the seed-42 complementary 50% split used for evaluation."""
    source_locale = dataset_root / locale
    meta_path = source_locale / "meta.lst"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Seed-TTS metadata is missing: {meta_path}")

    rows = _valid_meta_lines(meta_path)
    random.Random(DATASET_SEED).shuffle(rows)
    split = math.floor(len(rows) * TRAIN_FRACTION)
    holdout_rows = rows[split:]
    holdout_root = output_dir / "seed_tts_holdout_seed42"
    holdout_locale = holdout_root / locale
    holdout_locale.mkdir(parents=True, exist_ok=True)
    (holdout_locale / "meta.lst").write_text("\n".join(holdout_rows) + "\n", encoding="utf-8")
    _ensure_dataset_link(source_locale / "prompt-wavs", holdout_locale / "prompt-wavs")
    target_wavs = source_locale / "wavs"
    if target_wavs.is_dir():
        _ensure_dataset_link(target_wavs, holdout_locale / "wavs")
    return holdout_root, len(holdout_rows)


def _append_optional_argument(command: list[str], name: str, value: str | None) -> None:
    """Append a CLI option only when its value is provided."""
    if value:
        command.extend([name, value])


def _run_abse(
    *,
    label: str,
    host: str,
    port: int,
    args: argparse.Namespace,
    holdout_root: Path,
    num_prompts: int,
    result_dir: Path,
    summary_path: Path,
) -> None:
    """Run the existing Seed-TTS benchmark against one live service."""
    command = [
        sys.executable,
        str(SCRIPT_DIR / "abse.py"),
        "--label",
        label,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        args.model,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(args.max_concurrency),
        "--num-warmups",
        str(args.num_warmups),
        "--seed",
        str(DATASET_SEED),
        "--seed-tts-dataset-path",
        str(holdout_root),
        "--seed-tts-root",
        str(holdout_root),
        "--seed-tts-locale",
        args.seed_tts_locale,
        "--seed-extra-body-json",
        args.seed_extra_body_json,
        "--result-dir",
        str(result_dir),
        "--summary-jsonl",
        str(summary_path),
    ]
    _append_optional_argument(command, "--seed-tts-eval-device", args.seed_tts_eval_device)
    _append_optional_argument(command, "--seed-tts-whisper-model", args.seed_tts_whisper_model)
    _append_optional_argument(command, "--seed-tts-sim-device", args.seed_tts_sim_device)
    _append_optional_argument(command, "--seed-tts-wavlm-model", args.seed_tts_wavlm_model)
    if args.seed_tts_file_ref_audio:
        command.append("--seed-tts-file-ref-audio")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def _load_summary_row(summary_path: Path, label: str) -> dict[str, Any]:
    """Return the most recently appended summary row for an experiment label."""
    matches: list[dict[str, Any]] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") == label:
            matches.append(row)
    if not matches:
        raise ValueError(f"No summary row found for label={label!r} in {summary_path}")
    return matches[-1]


def _numeric(value: Any) -> float | None:
    """Convert a benchmark metric to float while preserving missing values."""
    return None if value is None else float(value)


def _metric_comparison(
    baseline: dict[str, Any],
    distilled: dict[str, Any],
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, float | None]:
    """Calculate aligned absolute and multiplicative metric changes."""
    baseline_value = _numeric(baseline.get(metric))
    distilled_value = _numeric(distilled.get(metric))
    if baseline_value is None or distilled_value is None:
        return {"baseline": baseline_value, "distilled": distilled_value, "delta": None, "factor": None}
    if higher_is_better:
        factor = distilled_value / baseline_value if baseline_value != 0 else None
    else:
        factor = baseline_value / distilled_value if distilled_value != 0 else None
    return {
        "baseline": baseline_value,
        "distilled": distilled_value,
        "delta": distilled_value - baseline_value,
        "factor": factor,
    }


def _build_comparison(
    baseline: dict[str, Any],
    distilled: dict[str, Any],
    *,
    locale: str,
    num_prompts: int,
) -> dict[str, Any]:
    """Build the final WER and timing comparison record."""
    lower_better = (
        "seed_tts_content_error_mean",
        "mean_e2el_ms",
        "mean_audio_ttfp_ms",
        "mean_audio_rtf",
    )
    result: dict[str, Any] = {
        "seed": DATASET_SEED,
        "partition": "held_out_second_50_percent",
        "locale": locale,
        "num_prompts": num_prompts,
        "baseline_result_path": baseline.get("result_path"),
        "distilled_result_path": distilled.get("result_path"),
        "metrics": {},
    }
    for metric in lower_better:
        result["metrics"][metric] = _metric_comparison(
            baseline,
            distilled,
            metric,
            higher_is_better=False,
        )
    result["metrics"]["request_throughput"] = _metric_comparison(
        baseline,
        distilled,
        "request_throughput",
        higher_is_better=True,
    )
    return result


def main() -> int:
    """Run both held-out benchmarks and save their comparison."""
    args = _build_parser().parse_args()
    if (args.baseline_host, args.baseline_port) == (args.distilled_host, args.distilled_port):
        raise ValueError("Baseline and distilled services must use different live endpoints.")
    if args.num_prompts < 1:
        raise ValueError("num-prompts must be positive.")

    dataset_root = args.seed_tts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_root, holdout_size = _build_holdout_root(
        dataset_root,
        args.seed_tts_locale,
        output_dir,
    )
    num_prompts = min(args.num_prompts, holdout_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_label = f"full_k16_{args.seed_tts_locale}_{timestamp}"
    distilled_label = f"distilled_k8_{args.seed_tts_locale}_{timestamp}"
    summary_path = output_dir / "summary.jsonl"
    result_dir = output_dir / "raw"

    _run_abse(
        label=baseline_label,
        host=args.baseline_host,
        port=args.baseline_port,
        args=args,
        holdout_root=holdout_root,
        num_prompts=num_prompts,
        result_dir=result_dir,
        summary_path=summary_path,
    )
    _run_abse(
        label=distilled_label,
        host=args.distilled_host,
        port=args.distilled_port,
        args=args,
        holdout_root=holdout_root,
        num_prompts=num_prompts,
        result_dir=result_dir,
        summary_path=summary_path,
    )

    baseline = _load_summary_row(summary_path, baseline_label)
    distilled = _load_summary_row(summary_path, distilled_label)
    comparison = _build_comparison(
        baseline,
        distilled,
        locale=args.seed_tts_locale,
        num_prompts=num_prompts,
    )
    comparison_path = output_dir / f"comparison_{args.seed_tts_locale}_{timestamp}.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved comparison to {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
