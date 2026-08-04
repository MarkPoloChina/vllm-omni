# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-TTS RVQ-tail distillation helpers."""

from __future__ import annotations

import torch

from train_qwen3_tts_tail_distillation import (
    _accumulation_group_size,
    _align_talker_hidden_to_codec_frames,
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
