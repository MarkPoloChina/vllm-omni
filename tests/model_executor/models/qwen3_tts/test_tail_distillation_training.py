# SPDX-License-Identifier: Apache-2.0
"""Static regression coverage for Teacher-rollout tail distillation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import torch

from train_qwen3_tts_tail_distillation import (
    SeedTTSRow,
    TailTeacherWeights,
    TeacherTrace,
    _accumulation_group_size,
    _build_run_configuration,
    _configuration_hash,
    _create_run_directory,
    _distillation_loss,
    _load_training_rows,
    _project_previous_codes,
    _validate_trace,
    _write_manifest,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor_tail import (
    RVQTailDistillationConfig,
    RVQTailDistillationModel,
)


def test_seed_tts_training_rows_do_not_require_target_wav(tmp_path) -> None:
    """Only reference audio and text conditions are required for a rollout."""
    locale_root = tmp_path / "en"
    prompt_root = locale_root / "prompt-wavs"
    prompt_root.mkdir(parents=True)
    lines = []
    for index in range(4):
        reference = prompt_root / f"ref_{index}.wav"
        reference.touch()
        lines.append(
            f"target_{index}|reference {index}|prompt-wavs/{reference.name}|synthesis {index}"
        )
    (locale_root / "meta.lst").write_text("\n".join(lines), encoding="utf-8")

    rows = _load_training_rows(tmp_path, ["en"])

    assert len(rows) == 2
    assert all(row.ref_audio.is_file() for row in rows)
    assert not (locale_root / "wavs").exists()


def test_manifest_declares_teacher_supervision_and_no_target_audio(tmp_path) -> None:
    """The persisted manifest must make the supervision source unambiguous."""
    reference = tmp_path / "reference.wav"
    reference.touch()
    row = SeedTTSRow(
        utterance_id="target",
        locale="en",
        ref_text="reference",
        target_text="synthesis",
        ref_audio=reference,
        meta_line="target|reference|reference.wav|synthesis",
    )
    destination = tmp_path / "manifest.jsonl"

    _write_manifest([row], destination, split_name="train")

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["supervision_source"] == "teacher_rollout"
    assert record["target_audio_used"] is False
    assert "target_audio" not in record


def test_teacher_previous_code_embeddings_are_step_aligned() -> None:
    """Tail q must consume the Teacher code from q minus one."""
    embedding_weight = torch.zeros(2, 5, 3)
    embedding_weight[0, 2] = torch.tensor([2.0, 20.0, 200.0])
    embedding_weight[1, 4] = torch.tensor([4.0, 40.0, 400.0])
    weights = TailTeacherWeights(
        previous_embedding_weight=embedding_weight,
        projection_weight=None,
        projection_bias=None,
        tail_lm_head_weight=torch.empty(2, 3, 5),
    )

    result = _project_previous_codes(
        torch.tensor([[2, 4]]),
        weights,
        dtype=torch.float32,
    )

    assert torch.equal(result[0, 0], embedding_weight[0, 2])
    assert torch.equal(result[0, 1], embedding_weight[1, 4])


def test_distillation_loss_backpropagates_from_teacher_rollout_codes() -> None:
    """Teacher-conditioned CE, KL, and hidden loss must train the Student."""
    torch.manual_seed(7)
    batch_size = 6
    hidden_size = 12
    state_size = 8
    vocab_size = 17
    num_groups = 5
    truncation_k = 2
    tail_steps = num_groups - truncation_k
    student = RVQTailDistillationModel(
        RVQTailDistillationConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            num_code_groups=num_groups,
        )
    )
    weights = TailTeacherWeights(
        previous_embedding_weight=torch.randn(tail_steps, vocab_size, hidden_size),
        projection_weight=None,
        projection_bias=None,
        tail_lm_head_weight=torch.randn(tail_steps, hidden_size, vocab_size),
    )
    codes = torch.randint(vocab_size, (batch_size, num_groups))
    exit_hidden = torch.randn(batch_size, hidden_size)
    teacher_hidden = torch.randn(batch_size, tail_steps, hidden_size)

    loss, metrics = _distillation_loss(
        student=student,
        weights=weights,
        exit_hidden=exit_hidden,
        teacher_tail_hidden=teacher_hidden,
        teacher_codes=codes,
        truncation_k=truncation_k,
        temperature=2.0,
        ce_weight=1.0,
        kl_weight=1.0,
        hidden_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student.output_projection.weight.grad is not None
    assert float(student.output_projection.weight.grad.norm()) > 0.0
    assert torch.isfinite(metrics["ce_teacher_code"])
    assert torch.isfinite(metrics["kl_teacher_distribution"])
    assert 0.0 <= float(metrics["student_teacher_code_top1"]) <= 1.0


def test_trace_shape_validation_uses_rows_and_tail_steps() -> None:
    """Trace offsets and tail dimensions must agree with the configured K."""
    trace = TeacherTrace(
        exit_hidden=torch.empty(5, 4),
        teacher_tail_hidden=torch.empty(5, 2, 4),
        teacher_codes=torch.empty(5, 4, dtype=torch.long),
        row_offsets=torch.tensor([0, 2, 5]),
    )

    _validate_trace(trace, rows=2, num_groups=4, truncation_k=2)


def test_partial_accumulation_group_uses_actual_tail_size() -> None:
    """A partial final group must not be divided by the configured full size."""
    values = [
        _accumulation_group_size(index, 10, 4)
        for index in range(10)
    ]

    assert values == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_configuration_hash_and_run_directory_format(tmp_path) -> None:
    """Output location is excluded while training arguments affect the hash."""
    args = SimpleNamespace(
        output_dir=tmp_path / "first",
        model_path=tmp_path / "model",
        dataset_root=tmp_path / "dataset",
        epochs=20,
        batch_size=4096,
        locales=("en", "zh"),
        resume_from=None,
    )
    first = _configuration_hash(_build_run_configuration(args))
    args.output_dir = tmp_path / "second"
    same = _configuration_hash(_build_run_configuration(args))
    args.epochs = 21
    changed = _configuration_hash(_build_run_configuration(args))
    started = datetime(2026, 8, 5, 13, 26, 50, tzinfo=timezone.utc)

    run = _create_run_directory(tmp_path, first, started_at=started)

    assert first == same
    assert first != changed
    assert run.name == f"rvq_20260805-132650_{first}"

