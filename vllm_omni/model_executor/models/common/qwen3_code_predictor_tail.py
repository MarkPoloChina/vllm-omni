"""Small recurrent student used to complete a truncated RVQ code sequence.

The full Qwen3-TTS code predictor repeatedly evaluates a transformer while
growing a short sequence by one RVQ codebook at a time.  This module replaces
the expensive tail of that loop with a compact recurrent transition.  The
student predicts hidden states in the original code-predictor hidden space, so
the frozen codec embeddings and per-codebook LM heads can be reused.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class RVQTailDistillationConfig:
    """Configuration for :class:`RVQTailDistillationModel`.

    Args:
        hidden_size: Hidden size of the full code predictor.
        state_size: Hidden size of the recurrent student.
        num_code_groups: Total number of RVQ codebooks.
    """

    hidden_size: int
    state_size: int
    num_code_groups: int


class RVQTailDistillationModel(nn.Module):
    """Predict code-predictor tail hidden states with a small GRU transition."""

    def __init__(self, config: RVQTailDistillationConfig) -> None:
        """Initialize the recurrent RVQ-tail student.

        Args:
            config: Student dimensions and number of RVQ codebooks.
        """
        super().__init__()
        self.config = config
        self.state_projection = nn.Linear(config.hidden_size, config.state_size)
        self.code_projection = nn.Linear(config.hidden_size, config.state_size, bias=False)
        self.step_embedding = nn.Embedding(config.num_code_groups, config.state_size)
        self.transition = nn.GRUCell(config.state_size, config.state_size)
        self.output_projection = nn.Linear(config.state_size, config.hidden_size)
        self.output_gate = nn.Linear(config.state_size, config.hidden_size)

    def initialize(self, exit_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create recurrent state and the residual anchor at the truncation point.

        Args:
            exit_hidden: Full-model hidden state with shape ``[batch, hidden]``.

        Returns:
            A ``(state, anchor)`` tuple used by :meth:`predict_next`.
        """
        anchor = exit_hidden
        state = torch.tanh(self.state_projection(exit_hidden))
        return state, anchor

    def predict_next(
        self,
        state: torch.Tensor,
        anchor: torch.Tensor,
        previous_code_embedding: torch.Tensor,
        codebook_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the hidden state used by one residual-codebook LM head.

        Args:
            state: Previous recurrent state with shape ``[batch, state_size]``.
            anchor: Full-model hidden state captured at truncation.
            previous_code_embedding: Projected embedding of the preceding code.
            codebook_index: Absolute RVQ codebook index being predicted.

        Returns:
            ``(predicted_hidden, next_state)``.
        """
        if isinstance(codebook_index, int):
            step_ids = torch.full(
                (state.shape[0],),
                codebook_index,
                dtype=torch.long,
                device=state.device,
            )
        else:
            step_ids = codebook_index.to(device=state.device, dtype=torch.long).reshape(-1)
        transition_input = self.code_projection(previous_code_embedding)
        transition_input = F.silu(transition_input + self.step_embedding(step_ids))
        next_state = self.transition(transition_input, state)
        gate = torch.sigmoid(self.output_gate(next_state))
        predicted_hidden = anchor + gate * self.output_projection(next_state)
        return predicted_hidden, next_state
