# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-TTS RVQ-tail distillation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from train_qwen3_tts_tail_distillation import (
    _accumulation_group_size,
    _align_talker_hidden_to_codec_frames,
    _build_run_configuration,
    _configuration_hash,
    _create_run_directory,
    _distillation_loss,
    _length_bucketed_batches,
    _load_training_rows,
    _load_validation_rows,
    _reuse_prior_preprocessing_cache,
    _write_preprocessing_cache_descriptor,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor_tail import (
    RVQTailDistillationConfig,
    RVQTailDistillationModel,
)


def test_talker_hidden_is_shifted_to_the_state_before_each_codec_frame() -> None:
    """A frame must use the hidden state before its own embeddings are consumed."""
    hidden_states = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)
    codec_mask = torch.tensor(
        [
            [False, False, True, True, True, False],
            [False, True, True, False, False, False],
        ]
    )

    aligned = _align_talker_hidden_to_codec_frames(hidden_states, codec_mask)

    # Row 0 codec positions 2, 3, 4 use hidden positions 1, 2, 3;
    # row 1 codec positions 1, 2 use hidden positions 0, 1.
    assert torch.equal(aligned.squeeze(-1), torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0]))


def test_partial_gradient_accumulation_group_uses_its_actual_size() -> None:
    """The last partial group is normalized independently in every epoch."""
    sizes = [
        _accumulation_group_size(index, num_batches=10, gradient_accumulation_steps=4)
        for index in range(10)
    ]

    assert sizes == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_configuration_hash_is_stable_and_ignores_output_location(tmp_path) -> None:
    arguments = SimpleNamespace(
        output_dir=tmp_path / "output-a",
        model_path=tmp_path / "model",
        dataset_root=tmp_path / "dataset",
        epochs=20,
        batch_size=4,
        locales=("en", "zh"),
        resume_from=None,
    )
    first = _configuration_hash(_build_run_configuration(arguments))
    arguments.output_dir = tmp_path / "output-b"
    second = _configuration_hash(_build_run_configuration(arguments))
    arguments.epochs = 21
    changed = _configuration_hash(_build_run_configuration(arguments))

    assert first == second
    assert first != changed
    assert len(first) == 12


def test_run_directory_uses_timestamp_and_config_hash_without_overwrite(tmp_path) -> None:
    started_at = datetime(2026, 8, 5, 13, 26, 50, tzinfo=timezone.utc)

    first = _create_run_directory(tmp_path, "0123456789ab", started_at=started_at)
    second = _create_run_directory(tmp_path, "0123456789ab", started_at=started_at)

    assert first.name == "rvq_20260805-132650_0123456789ab"
    assert second.name == "rvq_20260805-132651_0123456789ab"


def test_preprocessing_cache_is_reused_inside_the_new_run_directory(tmp_path) -> None:
    previous_cache = (
        tmp_path
        / "rvq_20260805-120000_0123456789ab"
        / "preprocessing_cache"
        / "train.safetensors"
    )
    previous_cache.parent.mkdir(parents=True)
    previous_cache.write_bytes(b"cached tensors")
    _write_preprocessing_cache_descriptor(previous_cache, "content-fingerprint")
    current_run = tmp_path / "rvq_20260805-130000_0123456789ab"
    current_cache = current_run / "preprocessing_cache" / "train.safetensors"
    current_cache.parent.mkdir(parents=True)

    _reuse_prior_preprocessing_cache(
        tmp_path,
        current_run,
        current_cache,
        "content-fingerprint",
    )

    assert current_cache.read_bytes() == b"cached tensors"
    assert current_cache.with_suffix(".json").is_file()


def test_validation_rows_are_disjoint_from_training_half(tmp_path) -> None:
    locale_root = tmp_path / "en"
    wav_root = locale_root / "wavs"
    wav_root.mkdir(parents=True)
    meta_lines = []
    for index in range(8):
        utterance_id = f"utterance_{index}"
        reference_name = f"reference_{index}.wav"
        (wav_root / f"{utterance_id}.wav").touch()
        (wav_root / reference_name).touch()
        meta_lines.append(
            f"{utterance_id}|reference text|wavs/{reference_name}|target text {index}"
        )
    (locale_root / "meta.lst").write_text("\n".join(meta_lines), encoding="utf-8")

    training = _load_training_rows(tmp_path, ["en"])
    validation = _load_validation_rows(tmp_path, ["en"], rows_per_locale=3)

    assert len(training) == 4
    assert len(validation) == 3
    assert {row.utterance_id for row in training}.isdisjoint(
        row.utterance_id for row in validation
    )


def test_length_bucketed_batches_keep_similar_sequences_together() -> None:
    rows = [SimpleNamespace(index=index, audio_codes=torch.empty(index + 1, 1)) for index in range(10)]

    batches = _length_bucketed_batches(rows, batch_size=4, seed=42)

    full_batch_indices = {frozenset(row.index for row in batch) for batch in batches[:-1]}
    assert full_batch_indices == {frozenset(range(4)), frozenset(range(4, 8))}
    assert {row.index for row in batches[-1]} == {8, 9}


class _TinyCodePredictor(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_groups: int) -> None:
        super().__init__()
        self.lm_head = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_groups - 1)]
        )
        self._embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(vocab_size, hidden_size) for _ in range(num_groups - 1)]
        )
        self.small_to_mtp_projection = torch.nn.Identity()

    def get_input_embeddings(self) -> torch.nn.ModuleList:
        return self._embeddings


def test_distillation_loss_backpropagates_and_reports_teacher_baseline() -> None:
    """The pure distillation objective must produce finite student gradients."""
    torch.manual_seed(7)
    batch_size = 6
    hidden_size = 12
    num_groups = 5
    truncation_k = 2
    vocab_size = 17
    student = RVQTailDistillationModel(
        RVQTailDistillationConfig(
            hidden_size=hidden_size,
            state_size=8,
            num_code_groups=num_groups,
        )
    )
    code_predictor = _TinyCodePredictor(hidden_size, vocab_size, num_groups)
    code_predictor.requires_grad_(False)
    teacher_hidden = torch.randn(batch_size, num_groups, hidden_size)
    frame_codes = torch.randint(vocab_size, (batch_size, num_groups))
    previous_embeddings = torch.randn(batch_size, num_groups - truncation_k, hidden_size)

    loss, metrics = _distillation_loss(
        student=student,
        tail_lm_head_weight=torch.stack(
            [code_predictor.lm_head[index - 1].weight for index in range(truncation_k, num_groups)]
        ).transpose(1, 2).contiguous(),
        exit_hidden=teacher_hidden[:, truncation_k - 1],
        previous_embeddings=previous_embeddings,
        teacher_tail_hidden=teacher_hidden[:, truncation_k:],
        tail_codes=frame_codes[:, truncation_k:],
        truncation_k=truncation_k,
        temperature=2.0,
        ce_weight=0.0,
        kl_weight=1.0,
        hidden_weight=0.1,
    )
    with torch.no_grad():
        state, anchor = student.initialize(teacher_hidden[:, truncation_k - 1])
        predicted_hidden = []
        for offset in range(num_groups - truncation_k):
            hidden, state = student.predict_next(
                state,
                anchor,
                previous_embeddings[:, offset],
                truncation_k + offset,
            )
            predicted_hidden.append(hidden)
        manual_student_logits = torch.stack(
            [
                code_predictor.lm_head[truncation_k + offset - 1](hidden)
                for offset, hidden in enumerate(predicted_hidden)
            ],
            dim=1,
        )
        manual_teacher_logits = torch.stack(
            [
                code_predictor.lm_head[truncation_k + offset - 1](
                    teacher_hidden[:, truncation_k + offset]
                )
                for offset in range(num_groups - truncation_k)
            ],
            dim=1,
        )
        expected_kl = F.kl_div(
            F.log_softmax(manual_student_logits.flatten(0, 1) / 2.0, dim=-1),
            F.softmax(manual_teacher_logits.flatten(0, 1) / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)
    probe_before = student.output_projection.weight.detach().clone()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    output_projection_grad = student.output_projection.weight.grad
    optimizer.step()
    update_norm = (student.output_projection.weight.detach() - probe_before).norm()
    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)
    assert float(grad_norm) > 0.0
    assert output_projection_grad is not None
    assert float(output_projection_grad.norm()) > 0.0
    assert float(update_norm) > 0.0
    assert float(metrics["teacher_ce"]) > 0.0
    assert torch.allclose(metrics["kl"], expected_kl)
    assert float(metrics["loss"]) == pytest.approx(float(metrics["kl"] + 0.1 * metrics["hidden"]))
    assert float(metrics["hidden_cosine_similarity"]) == pytest.approx(
        1.0 - float(metrics["hidden_cosine_distance"])
    )
    assert torch.equal(metrics["hidden_cosine"], metrics["hidden_cosine_distance"])
    assert 0.0 <= float(metrics["student_teacher_top1"]) <= 1.0
    assert 0.0 <= float(metrics["student_teacher_tail_exact"]) <= 1.0
    for codebook_index in range(truncation_k, num_groups):
        assert 0.0 <= float(metrics[f"student_teacher_top1_codebook_{codebook_index}"]) <= 1.0
