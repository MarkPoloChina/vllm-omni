"""Qwen3 Code Predictor -- optimized re-prefill, no KV cache.

Shared by Qwen3-Omni and Qwen3-TTS talker models.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* Per-call embedding buffer to avoid cross-request aliasing
* Pre-allocated position_ids (read-only, safe to persist)
* torch.compile (epilogue_fusion=False) on inner transformer by default
* Optional manual CUDA graph capture per batch-size bucket
* Inline sampling (top-k + top-p) -- no custom op overhead
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.common.qwen3_code_predictor_tail import (
    RVQTailDistillationConfig,
    RVQTailDistillationModel,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# ===================================================================
# HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference. vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the autoregressive steps of the code predictor, causing severe
# audio quality degradation.
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's implementation exactly.

    Computes variance in float32 to avoid bfloat16 precision loss.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's implementation exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (matching HF)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ===================================================================
#  Attention
# ===================================================================


class CodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    No KV cache -- the code predictor always re-prefills the full (short)
    sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.is_gqa = self.num_kv_heads != self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.max_seq = int(config.num_code_groups) + 1

        # Separate q/k/v projections matching HF (no fused packing)
        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if current_omni_platform.is_npu():
            if self.max_seq > 2048:
                raise ValueError(
                    "Qwen3-TTS code predictor NPU fusion attention uses a fixed 2048x2048 "
                    f"causal mask, but max_seq={self.max_seq} exceeds the mask size."
                )
            # Ascend SDPA is_causal migration example uses a fixed 2048x2048
            # compressed causal mask with sparse_mode=2.
            fusion_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("_fusion_causal_mask", fusion_mask, persistent=False)

    def _forward_npu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        import torch_npu

        q_f, k_f, v_f = q, k, v
        if self.is_gqa:
            k_f = (
                k[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )
            v_f = (
                v[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )

        mask = self._fusion_causal_mask
        mask = mask.contiguous()
        q_f = q_f.contiguous()
        k_f = k_f.contiguous()
        v_f = v_f.contiguous()
        return torch_npu.npu_fusion_attention(
            q_f,
            k_f,
            v_f,
            self.num_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=float(self.scaling),
            keep_prob=1.0,
            # Keep torch_npu's API spelling.
            pre_tockens=2147483647,
            next_tockens=2147483647,
            inner_precise=0,
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            # Ascend SDPA is_causal migration example uses sparse_mode=2.
            sparse_mode=2,
            gen_mask_parallel=True,
            # Keep sync=True for the NPU fused attention path.
            sync=True,
        )[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        # cos/sin are [batch, seq_len, head_dim], need unsqueeze at dim=1 for heads
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        if not current_omni_platform.is_npu():
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                is_causal=True,
                enable_gqa=self.is_gqa,
            )
        else:
            attn_out = self._forward_npu_attention(q, k, v, bsz, seq_len)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out)


# ===================================================================
#  MLP
# ===================================================================


class CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ===================================================================
#  Decoder Layer
# ===================================================================


class CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Base Transformer Model (re-prefill, no KV cache)
# ===================================================================


class CodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    Signature: ``forward(inputs_embeds, position_ids) -> hidden_states``
    """

    def __init__(
        self,
        config,
        *,
        embedding_dim: int | None = None,
        use_parallel_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        if use_parallel_embedding:
            self.codec_embedding = nn.ModuleList(
                [VocabParallelEmbedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )
        else:
            self.codec_embedding = nn.ModuleList(
                [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Run the transformer body in float32 when the model is in fp16.
        # fp16 lacks the dynamic range for stable attention scores and
        # SiLU-gated MLP intermediates, producing NaN on GPUs without
        # native bf16 support (Turing, Volta).  The RMSNorm and RoPE
        # layers already upcast internally; this extends the same
        # treatment to attention and MLP.
        # autocast to float32 is unsupported on CPU; skip fp32 upcast there
        # (CPU uses full-precision intermediates internally).
        input_dtype = inputs_embeds.dtype
        use_fp32 = input_dtype == torch.float16 and inputs_embeds.device.type != "cpu"
        if use_fp32:
            inputs_embeds = inputs_embeds.float()
        hidden_states = inputs_embeds
        with torch.amp.autocast(inputs_embeds.device.type, enabled=use_fp32, dtype=torch.float32):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings)
            hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===================================================================
#  Wrapper Configuration
# ===================================================================


@dataclasses.dataclass
class CodePredictorWrapperConfig:
    """Controls behavioral differences between model-specific code predictors."""

    use_cuda_graphs: bool = False
    use_parallel_embedding: bool = False
    use_projection: bool = False
    return_proj_buf: bool = False
    sampling_mode: str = "stored"
    early_exit_enabled: bool = False
    early_exit_metric: str = "entropy"
    early_exit_min_keep: int = 8
    early_exit_entropy_threshold: float = 0.9
    early_exit_confidence_threshold: float = 0.45
    early_exit_margin_threshold: float = 0.1
    early_exit_patience: int = 1
    early_exit_fill_strategy: str = "pad"
    early_exit_pad_token: int = 0
    early_exit_prior_tokens: Sequence[int] | None = None
    truncation_mode: str = "none"
    truncation_k: int = 8
    distillation_state_size: int = 384
    distillation_weights: str | None = None


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill, persistent buffers)
# ===================================================================


class CodePredictorWrapper(nn.Module):
    """Optimized code predictor -- re-prefill approach, no KV cache.

    Each AR step forwards the full growing sequence (len 2 -> num_code_groups+1)
    through the transformer.  The extra O(T^2) FLOPs are negligible for
    short sequences, and this avoids all KV-cache management overhead.

    Optimizations:
      1. Per-call embedding buffer -- avoids cross-request aliasing.
      2. Pre-allocated position_ids -- no torch.arange per step.
      3. Cached module references -- bypass ModuleList indexing.
      4. torch.compile on inner transformer.
      5. Inline sampling (top-k + top-p) -- no custom op overhead.
      6. Optional manual CUDA graph capture per batch-size bucket.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cp_config,
        wrapper_config: CodePredictorWrapperConfig,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = cp_config
        self._wrapper_config = wrapper_config
        self.prefix = prefix
        extra_cfg = self._stage_connector_extra_config(vllm_config)

        self._num_groups = int(cp_config.num_code_groups)
        self._cp_hidden = int(cp_config.hidden_size)
        self._vocab_size = int(cp_config.vocab_size)

        # For Omni backward compat (accessed by the talker)
        self.num_code_groups = self._num_groups

        # Determine embedding dimension
        _talker_hidden = int(talker_hidden_size) if talker_hidden_size is not None else self._cp_hidden

        self.model = CodePredictorBaseModel(
            cp_config,
            embedding_dim=_talker_hidden,
            use_parallel_embedding=wrapper_config.use_parallel_embedding,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(cp_config.hidden_size, cp_config.vocab_size, bias=False) for _ in range(self._num_groups - 1)]
        )

        # Projection: Identity when hidden sizes match or not needed
        if wrapper_config.use_projection and _talker_hidden != self._cp_hidden:
            self.small_to_mtp_projection = nn.Linear(_talker_hidden, self._cp_hidden, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling defaults for "stored" mode
        self._top_k: int = 50
        self._top_p: float = 0.8

        # Keep the original perceptual early-exit configuration available, but
        # select it explicitly through truncation_mode.  If the new option is
        # absent, preserve compatibility with early_exit_enabled.
        legacy_early_exit_enabled = self._parse_bool_config(
            extra_cfg.get("code_predictor_early_exit_enabled", wrapper_config.early_exit_enabled)
        )
        configured_truncation_mode = self._parse_string_config(
            extra_cfg.get("code_predictor_truncation_mode", wrapper_config.truncation_mode)
        ).lower()
        if not configured_truncation_mode or (
            "code_predictor_truncation_mode" not in extra_cfg and legacy_early_exit_enabled
        ):
            configured_truncation_mode = "perceptual" if legacy_early_exit_enabled else "none"
        self._truncation_mode = configured_truncation_mode
        self._fixed_truncation_k = self._parse_int_config(
            extra_cfg.get("code_predictor_truncation_k", wrapper_config.truncation_k),
            default=wrapper_config.truncation_k,
            minimum=2,
        )
        self._early_exit_enabled = self._truncation_mode == "perceptual"
        self._early_exit_metric = self._parse_string_config(
            extra_cfg.get("code_predictor_early_exit_metric", wrapper_config.early_exit_metric)
        ).lower()
        self._early_exit_min_keep = self._parse_int_config(
            extra_cfg.get("code_predictor_early_exit_min_keep", wrapper_config.early_exit_min_keep),
            default=wrapper_config.early_exit_min_keep,
            minimum=1,
        )
        self._early_exit_entropy_threshold = self._parse_float_config(
            extra_cfg.get(
                "code_predictor_early_exit_entropy_threshold",
                wrapper_config.early_exit_entropy_threshold,
            ),
            default=wrapper_config.early_exit_entropy_threshold,
        )
        self._early_exit_confidence_threshold = self._parse_float_config(
            extra_cfg.get(
                "code_predictor_early_exit_confidence_threshold",
                wrapper_config.early_exit_confidence_threshold,
            ),
            default=wrapper_config.early_exit_confidence_threshold,
        )
        self._early_exit_margin_threshold = self._parse_float_config(
            extra_cfg.get(
                "code_predictor_early_exit_margin_threshold",
                wrapper_config.early_exit_margin_threshold,
            ),
            default=wrapper_config.early_exit_margin_threshold,
        )
        self._early_exit_patience = self._parse_int_config(
            extra_cfg.get("code_predictor_early_exit_patience", wrapper_config.early_exit_patience),
            default=wrapper_config.early_exit_patience,
            minimum=1,
        )
        self._early_exit_fill_strategy = self._parse_string_config(
            extra_cfg.get("code_predictor_early_exit_fill_strategy", wrapper_config.early_exit_fill_strategy)
        ).lower()
        self._early_exit_pad_token = self._normalize_code_token(
            self._parse_int_config(
                extra_cfg.get("code_predictor_early_exit_pad_token", wrapper_config.early_exit_pad_token),
                default=wrapper_config.early_exit_pad_token,
                minimum=0,
            )
        )
        self._early_exit_prior_tokens = self._parse_int_list_config(
            extra_cfg.get("code_predictor_early_exit_prior_tokens", wrapper_config.early_exit_prior_tokens)
        )
        self._distillation_state_size = self._parse_int_config(
            extra_cfg.get(
                "code_predictor_distillation_state_size",
                wrapper_config.distillation_state_size,
            ),
            default=wrapper_config.distillation_state_size,
            minimum=1,
        )
        self._distillation_weights = self._parse_string_config(
            extra_cfg.get(
                "code_predictor_distillation_weights",
                wrapper_config.distillation_weights,
            )
        )
        self._validate_truncation_config()

        self.tail_distillation: RVQTailDistillationModel | None = None
        if self._early_exit_fill_strategy == "distillation" and self._truncation_mode != "none":
            self.tail_distillation = RVQTailDistillationModel(
                RVQTailDistillationConfig(
                    hidden_size=self._cp_hidden,
                    state_size=self._distillation_state_size,
                    num_code_groups=self._num_groups,
                )
            )

        if self._truncation_mode == "perceptual":
            logger.info_once(
                "code_predictor: perceptual truncation enabled metric=%s min_keep=%d fill=%s",
                self._early_exit_metric,
                self._early_exit_min_keep,
                self._early_exit_fill_strategy,
            )
        elif self._truncation_mode == "fixed":
            logger.info_once(
                "code_predictor: fixed truncation enabled K=%d fill=%s",
                self._fixed_truncation_k,
                self._early_exit_fill_strategy,
            )

        # Lazily initialised state
        self._proj_buf: torch.Tensor | None = None
        self._model_dtype: torch.dtype | None = None
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int | tuple[int, int], torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None
        self._device_graphs: dict[int | tuple[int, int], tuple] = {}  # (graph, static_output) per bucket
        prefix_graphs_requested = self._parse_bool_config(extra_cfg.get("code_predictor_prefix_graphs"))
        self._prefix_graphs_enabled = prefix_graphs_requested and wrapper_config.use_cuda_graphs
        if prefix_graphs_requested and not self._prefix_graphs_enabled:
            logger.info_once(
                "code_predictor: prefix graphs requested but disabled because use_cuda_graphs=%s",
                wrapper_config.use_cuda_graphs,
            )
        self._prefix_graph_buckets = self._parse_positive_int_set(
            extra_cfg.get("code_predictor_prefix_graph_buckets")
        )
        self._prefix_graph_seq_lens = self._parse_positive_int_set(
            extra_cfg.get("code_predictor_prefix_graph_seq_lens")
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8) -> None:
        """Configure sampling parameters to maintain consistency with previous implementation."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug("Sampling parameters updated: top_k=%d, top_p=%.2f", top_k, top_p)

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype, bsz: int) -> None:
        """Ensure the projection buffer can hold at least *bsz* rows."""
        max_seq = self._num_groups + 1
        if (
            self._proj_buf is not None
            and self._proj_buf.device == device
            and self._proj_buf.dtype == dtype
            and self._proj_buf.shape[0] >= bsz
        ):
            return
        self._proj_buf = torch.zeros(bsz, max_seq, self._cp_hidden, dtype=dtype, device=device)

    def _setup_compile(self) -> None:
        """Lazily set up torch.compile with optional device graph capture."""
        if self._compiled_model_fwd is not None:
            return

        # Cache model parameter dtype so forward() doesn't need to query it
        # on every call.  Also ensures warmup buffers match model precision
        # even when upstream modules produce a different dtype (#2385).
        self._model_dtype = next(self.model.parameters()).dtype
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)

        if not current_omni_platform.supports_torch_inductor():
            # NPU or other platforms without Inductor support
            self._compiled_model_fwd = self.model.forward

            if current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
                # For NPU, use eager + NPU graphs (no torch.compile)
                self._warmup_buckets()
                self._capture_npu_graphs()
                logger.info("code_predictor: eager mode + NPU graphs")
            else:
                logger.warning_once("code_predictor: torch.compile disabled")
            return

        # torch.compile fuses RMSNorm/RoPE in ways that lose float32
        # precision, compounding across AR steps. Use epilogue_fusion=False
        # to disable the problematic fusions while still getting kernel
        # fusion benefits for the linear layers and SDPA.
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            dynamic=False,
            options={"epilogue_fusion": False},
        )
        self._warmup_buckets()

        if self._wrapper_config.use_cuda_graphs:
            self._capture_cuda_graphs()
            logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
        else:
            logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")

    def _padded_bsz(self, bsz: int) -> int:
        """Round batch size up to nearest power-of-2 bucket."""
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    @staticmethod
    def _stage_connector_extra_config(vllm_config: VllmConfig) -> dict:
        model_cfg = getattr(vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        return extra_cfg if isinstance(extra_cfg, dict) else {}

    @staticmethod
    def _parse_bool_config(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, int):
            return bool(value)
        return False

    @staticmethod
    def _parse_string_config(value: object) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _parse_int_config(value: object, *, default: int, minimum: int | None = None) -> int:
        if value is None:
            parsed = int(default)
        else:
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid int config value {value!r}") from exc
        if minimum is not None and parsed < minimum:
            parsed = minimum
        return parsed

    @staticmethod
    def _parse_float_config(value: object, *, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid float config value {value!r}") from exc

    @staticmethod
    def _parse_int_list_config(value: object) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        elif isinstance(value, int):
            raw_values = [value]
        else:
            try:
                raw_values = list(value)
            except TypeError as exc:
                raise ValueError(f"Invalid int list config value {value!r}") from exc
        out: list[int] = []
        for item in raw_values:
            try:
                out.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid int list config value {item!r}") from exc
        return out

    @staticmethod
    def _parse_positive_int_set(value: object) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        elif isinstance(value, int):
            raw_values = [value]
        else:
            try:
                raw_values = list(value)
            except TypeError as exc:
                raise ValueError(f"Invalid positive int config value {value!r}") from exc
        values: set[int] = set()
        for item in raw_values:
            try:
                parsed = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid positive int config value {item!r}") from exc
            if parsed > 0:
                values.add(parsed)
        return values

    def _normalize_code_token(self, token: int) -> int:
        return min(max(int(token), 0), self._vocab_size - 1)

    def _validate_truncation_config(self) -> None:
        """Validate fixed/perceptual truncation and tail-fill configuration."""
        valid_modes = {"none", "fixed", "perceptual"}
        if self._truncation_mode not in valid_modes:
            raise ValueError(
                "Invalid code_predictor_truncation_mode="
                f"{self._truncation_mode!r}; expected one of {sorted(valid_modes)}."
            )
        if self._fixed_truncation_k > self._num_groups:
            raise ValueError(
                "code_predictor_truncation_k must not exceed num_code_groups: "
                f"{self._fixed_truncation_k} > {self._num_groups}."
            )

        valid_metrics = {
            "entropy",
            "confidence",
            "entropy_or_confidence",
            "entropy_and_confidence",
        }
        if self._early_exit_metric not in valid_metrics:
            raise ValueError(
                "Invalid code_predictor_early_exit_metric="
                f"{self._early_exit_metric!r}; expected one of {sorted(valid_metrics)}."
            )
        valid_fill = {"distillation", "pad", "prior"}
        if self._early_exit_fill_strategy not in valid_fill:
            raise ValueError(
                "Invalid code_predictor_early_exit_fill_strategy="
                f"{self._early_exit_fill_strategy!r}; expected one of {sorted(valid_fill)}."
            )
        if (
            self._truncation_mode != "none"
            and self._early_exit_fill_strategy == "distillation"
            and not self._distillation_weights
        ):
            raise ValueError(
                "code_predictor_distillation_weights is required when "
                "code_predictor_early_exit_fill_strategy=distillation."
            )

    def _early_exit_trigger(self, probs: torch.Tensor) -> torch.Tensor:
        """Return a per-row mask indicating whether this step is low-value."""
        top2 = probs.topk(min(2, probs.shape[-1]), dim=-1).values
        confidence = top2[:, 0]
        if top2.shape[-1] > 1:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones_like(confidence)

        entropy = -(probs * (probs.clamp_min(1e-20).log())).sum(dim=-1)
        support = (probs > 0).sum(dim=-1).to(dtype=torch.float32).clamp_min(2.0)
        norm_entropy = entropy / support.log()
        entropy_trigger = norm_entropy >= self._early_exit_entropy_threshold
        confidence_trigger = (confidence <= self._early_exit_confidence_threshold) | (
            margin <= self._early_exit_margin_threshold
        )

        if self._early_exit_metric == "entropy":
            return entropy_trigger
        if self._early_exit_metric == "confidence":
            return confidence_trigger
        if self._early_exit_metric == "entropy_and_confidence":
            return entropy_trigger & confidence_trigger
        return entropy_trigger | confidence_trigger

    def _early_exit_fill_token(self, codebook_idx: int) -> int:
        if self._early_exit_fill_strategy == "pad" or not self._early_exit_prior_tokens:
            return self._early_exit_pad_token

        tokens = self._early_exit_prior_tokens
        if len(tokens) >= self._num_groups:
            token = tokens[codebook_idx]
        elif len(tokens) == self._num_groups - 1 and codebook_idx > 0:
            token = tokens[codebook_idx - 1]
        else:
            token = tokens[min(codebook_idx, len(tokens) - 1)]
        return self._normalize_code_token(token)

    def _fill_remaining_codebooks(
        self,
        *,
        all_codes: torch.Tensor,
        proj_buf: torch.Tensor,
        start_codebook: int,
        bsz: int,
        dtype: torch.dtype,
        projection: nn.Module,
        codec_embeds: list[nn.Module],
    ) -> None:
        """Fill codebooks start_codebook..G-1 after an early exit."""
        if start_codebook >= self._num_groups:
            return
        device = all_codes.device
        for codebook_idx in range(start_codebook, self._num_groups):
            fill_token = self._early_exit_fill_token(codebook_idx)
            code = torch.full((bsz, 1), fill_token, dtype=torch.long, device=device)
            if self._wrapper_config.return_proj_buf:
                all_codes[:, codebook_idx] = code
            else:
                all_codes[:, codebook_idx] = code.reshape(bsz)

            if self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[codebook_idx - 1](code)
                proj_buf[:bsz, codebook_idx + 1, :] = projection(new_embed.reshape(bsz, 1, -1).to(dtype)).reshape(
                    bsz, -1
                )

    def _complete_remaining_codebooks(
        self,
        *,
        all_codes: torch.Tensor,
        proj_buf: torch.Tensor,
        start_codebook: int,
        exit_hidden: torch.Tensor,
        previous_code_embedding: torch.Tensor,
        bsz: int,
        dtype: torch.dtype,
        projection: nn.Module,
        codec_embeds: list[nn.Module],
        lm_heads: list[nn.Module],
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator | None,
        generators: Sequence[torch.Generator | None] | None,
    ) -> None:
        """Complete truncated codebooks with either static or distilled fill.

        Args:
            all_codes: Output code tensor being populated.
            proj_buf: Projected autoregressive input buffer.
            start_codebook: First omitted codebook index.
            exit_hidden: Last hidden state computed by the full predictor.
            previous_code_embedding: Projected embedding of the last kept code.
            bsz: Active batch size.
            dtype: Full code-predictor dtype.
            projection: Talker-to-code-predictor projection.
            codec_embeds: Frozen residual-codebook embeddings.
            lm_heads: Frozen per-codebook output heads.
            do_sample: Whether per-call sampling is enabled.
            temperature: Per-call sampling temperature.
            top_k: Per-call top-k value.
            top_p: Per-call top-p value.
            generator: Optional shared random generator.
            generators: Optional per-row random generators.
        """
        if self._early_exit_fill_strategy != "distillation":
            self._fill_remaining_codebooks(
                all_codes=all_codes,
                proj_buf=proj_buf,
                start_codebook=start_codebook,
                bsz=bsz,
                dtype=dtype,
                projection=projection,
                codec_embeds=codec_embeds,
            )
            return

        student = self.tail_distillation
        if student is None:
            raise RuntimeError("Distillation fill requested without an initialized RVQ-tail student.")
        student_dtype = next(student.parameters()).dtype
        state, anchor = student.initialize(exit_hidden.to(dtype=student_dtype))
        previous_embedding = previous_code_embedding.to(dtype=student_dtype)

        for codebook_idx in range(start_codebook, self._num_groups):
            predicted_hidden, state = student.predict_next(
                state,
                anchor,
                previous_embedding,
                codebook_idx,
            )
            logits = lm_heads[codebook_idx - 1](predicted_hidden.to(dtype=dtype))
            code, _ = self._sample_logits(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                generators=generators,
            )
            if self._wrapper_config.return_proj_buf:
                all_codes[:, codebook_idx] = code
            else:
                all_codes[:, codebook_idx] = code.reshape(bsz)

            if codebook_idx < self._num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[codebook_idx - 1](code)
                projected = projection(new_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
                previous_embedding = projected.to(dtype=student_dtype)
                if self._wrapper_config.return_proj_buf:
                    proj_buf[:bsz, codebook_idx + 1, :] = projected

    def _prefix_seq_lens(self, max_seq: int) -> list[int]:
        all_seq_lens = list(range(2, max_seq))
        if not self._prefix_graph_seq_lens:
            return all_seq_lens
        allowed = set(all_seq_lens)
        return sorted(seq_len for seq_len in self._prefix_graph_seq_lens if seq_len in allowed)

    def _warmup_buckets(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load Inductor compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device

        # Ensure proj_buf matches model parameter dtype to avoid dtype
        # mismatch during warmup compilation (see #2385).
        self._ensure_buffers(device, self._model_dtype, max(self._bucket_sizes))
        proj_buf = self._proj_buf

        if self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets
                if not capture_prefixes or needs_full_graph:
                    pos_ids = (
                        torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
                    )
                    self._bucket_pos_ids[bsz] = pos_ids
                    for _ in range(3):
                        self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        pos_ids = (
                            torch.arange(seq_len, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(bsz, -1)
                            .contiguous()
                        )
                        self._bucket_pos_ids[(bsz, seq_len)] = pos_ids
                        for _ in range(2):
                            self._compiled_model_fwd(proj_buf[:bsz, :seq_len, :], pos_ids)
            logger.info(
                "code_predictor: prefix warmup done for buckets %s prefix_buckets=%s seq_lens=%s",
                self._bucket_sizes,
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                prefix_seq_lens,
            )
        else:
            for bsz in self._bucket_sizes:
                pos_ids = (
                    torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
                )
                self._bucket_pos_ids[bsz] = pos_ids
                for _ in range(3):
                    self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
            logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    def _capture_cuda_graphs(self) -> None:
        """Capture a CUDA graph per bucket using vLLM's global graph pool."""
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf

        if self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets
                if not capture_prefixes or needs_full_graph:
                    static_input = proj_buf[:bsz, :max_seq, :]
                    pos_ids = self._bucket_pos_ids[bsz]

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(static_input, pos_ids)

                    self._device_graphs[bsz] = (g, static_output)

                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        static_input = proj_buf[:bsz, :seq_len, :]
                        pos_ids = self._bucket_pos_ids[(bsz, seq_len)]

                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=pool):
                            static_output = self._compiled_model_fwd(static_input, pos_ids)

                        self._device_graphs[(bsz, seq_len)] = (g, static_output)

            logger.info(
                "code_predictor: captured prefix CUDA graphs for buckets %s prefix_buckets=%s seq_lens=%s",
                self._bucket_sizes,
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                prefix_seq_lens,
            )
        else:
            for bsz in self._bucket_sizes:
                static_input = proj_buf[:bsz, :max_seq, :]
                pos_ids = self._bucket_pos_ids[bsz]

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    static_output = self._compiled_model_fwd(static_input, pos_ids)

                self._device_graphs[bsz] = (g, static_output)

            logger.info("code_predictor: captured CUDA graphs for buckets %s", self._bucket_sizes)

    def _capture_npu_graphs(self) -> None:
        """Capture an NPU graph per bucket using torch_npu's NPUGraph."""
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf
        pool = torch.npu.graph_pool_handle()

        if self._prefix_graphs_enabled:
            prefix_seq_lens = self._prefix_seq_lens(max_seq)
            needs_full_graph = set(prefix_seq_lens) != set(range(2, max_seq))
            for bsz in self._bucket_sizes:
                capture_prefixes = not self._prefix_graph_buckets or bsz in self._prefix_graph_buckets

                if not capture_prefixes or needs_full_graph:
                    static_input = proj_buf[:bsz, :max_seq, :]
                    pos_ids = self._bucket_pos_ids[bsz]
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g, pool=pool):
                        static_output = self._compiled_model_fwd(static_input, pos_ids)
                    self._device_graphs[bsz] = (g, static_output)

                if capture_prefixes:
                    for seq_len in prefix_seq_lens:
                        static_input = proj_buf[:bsz, :seq_len, :]
                        pos_ids = self._bucket_pos_ids[(bsz, seq_len)]
                        g = torch.npu.NPUGraph()
                        with torch.npu.graph(g, pool=pool):
                            static_output = self._compiled_model_fwd(static_input, pos_ids)
                        self._device_graphs[(bsz, seq_len)] = (g, static_output)

            logger.info(
                "code_predictor: captured prefix NPU graphs for buckets %s prefix_buckets=%s seq_lens=%s",
                self._bucket_sizes,
                sorted(self._prefix_graph_buckets) if self._prefix_graph_buckets else "all",
                prefix_seq_lens,
            )
            return

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.npu.NPUGraph()
            with torch.npu.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._device_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured NPU graphs for buckets %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Forward -- re-prefill + inline sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _multinomial(
        probs: torch.Tensor,
        generator: torch.Generator | None,
        generators: Sequence[torch.Generator | None] | None,
    ) -> torch.Tensor:
        """Sample one code per row, optionally with per-row generators.

        Per-row generators keep explicitly-seeded requests deterministic in a
        multi-row batch: each row consumes draws only from its own generator,
        so the transformer forward can stay batched (#4883).
        """
        if generators is None:
            return torch.multinomial(probs, num_samples=1, generator=generator)
        return torch.cat(
            [
                torch.multinomial(probs[row : row + 1], num_samples=1, generator=row_generator)
                for row, row_generator in enumerate(generators)
            ]
        )

    def _sample_logits(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator | None,
        generators: Sequence[torch.Generator | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one code and return the probability tensor used for scoring.

        Args:
            logits: Per-row code logits.
            do_sample: Whether sampling is enabled in per-call mode.
            temperature: Sampling temperature in per-call mode.
            top_k: Top-k value in per-call mode.
            top_p: Top-p value in per-call mode.
            generator: Optional shared random generator.
            generators: Optional per-row random generators.

        Returns:
            ``(sampled_code, probabilities)``.
        """
        if self._wrapper_config.sampling_mode == "stored":
            if self._top_k > 0:
                topk_vals, _ = logits.topk(self._top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            if self._top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = (cumulative_probs - sorted_probs) >= self._top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            return self._multinomial(probs, generator, generators), probs

        use_sampling = do_sample and temperature > 0
        if use_sampling and top_p != 1.0:
            raise NotImplementedError(
                "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
            )
        if use_sampling:
            scaled = logits * (1.0 / max(temperature, 1e-6))
            if top_k > 0:
                topk_vals, _ = scaled.topk(top_k, dim=-1)
                scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
            probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
            return self._multinomial(probs, generator, generators), probs

        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        return logits.argmax(dim=-1, keepdim=True), probs

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively via re-prefill."""
        bsz = int(layer0_code.shape[0])
        if generators is not None and len(generators) != bsz:
            raise ValueError(f"generators must have one entry per row: got {len(generators)} for batch {bsz}")
        num_groups = self._num_groups
        device = layer0_code.device

        # _setup_compile caches _model_dtype on first call; use it for buffers
        # so they always match model weight precision (#2385).
        self._setup_compile()
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        self._ensure_buffers(device, dtype, padded_bsz)

        proj_buf = self._proj_buf
        max_seq = num_groups + 1
        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        # Zero the padded region of the buffer
        proj_buf[:padded_bsz].zero_()

        # Fill buffer positions 0 (talker hidden) & 1 (layer0 embed)
        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)

        # Output codes -- shape depends on return mode
        if self._wrapper_config.return_proj_buf:
            all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
        else:
            all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz)

        early_exit_hits = 0

        # Autoregressive loop: predict layers 1..G-1
        for step in range(1, num_groups):
            graph_key: int | tuple[int, int] = padded_bsz
            seq_len = max_seq
            if self._prefix_graphs_enabled:
                prefix_key = (padded_bsz, step + 1)
                if prefix_key in self._device_graphs:
                    graph_key = prefix_key
                    seq_len = step + 1
            pos_ids = self._bucket_pos_ids.get(graph_key)
            if pos_ids is None:
                pos_ids = (
                    torch.arange(seq_len, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(padded_bsz, -1)
                    .contiguous()
                )

            # Use captured device graph if available, otherwise call compiled fn.
            device_graph_entry = self._device_graphs.get(graph_key)

            # Run transformer (device graph replay or compiled forward)
            if device_graph_entry is not None:
                device_graph_entry[0].replay()
                hidden_out = device_graph_entry[1]
            else:
                hidden_out = model_fwd(proj_buf[:padded_bsz, :seq_len, :], pos_ids)

            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])
            code, score_probs = self._sample_logits(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                generators=generators,
            )

            # Store code
            if self._wrapper_config.return_proj_buf:
                all_codes[:, step] = code
            else:
                all_codes[:, step] = code.reshape(bsz)

            # Embed predicted code -> project -> next buffer position
            projected_embed: torch.Tensor | None = None
            if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[step - 1](code)
                projected_embed = projection(new_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
                proj_buf[:bsz, step + 1, :] = projected_embed

            if (
                self._truncation_mode == "fixed"
                and step < num_groups - 1
                and step + 1 >= self._fixed_truncation_k
            ):
                assert projected_embed is not None
                self._complete_remaining_codebooks(
                    all_codes=all_codes,
                    proj_buf=proj_buf,
                    start_codebook=step + 1,
                    exit_hidden=hidden_out[:bsz, step, :],
                    previous_code_embedding=projected_embed,
                    bsz=bsz,
                    dtype=dtype,
                    projection=projection,
                    codec_embeds=codec_embeds,
                    lm_heads=lm_heads,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    generators=generators,
                )
                break

            if self._early_exit_enabled and step < num_groups - 1 and step + 1 >= self._early_exit_min_keep:
                trigger = self._early_exit_trigger(score_probs)
                if bool(trigger.all().item()):
                    early_exit_hits += 1
                else:
                    early_exit_hits = 0
                if early_exit_hits >= self._early_exit_patience:
                    assert projected_embed is not None
                    self._complete_remaining_codebooks(
                        all_codes=all_codes,
                        proj_buf=proj_buf,
                        start_codebook=step + 1,
                        exit_hidden=hidden_out[:bsz, step, :],
                        previous_code_embedding=projected_embed,
                        bsz=bsz,
                        dtype=dtype,
                        projection=projection,
                        codec_embeds=codec_embeds,
                        lm_heads=lm_heads,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        generators=generators,
                    )
                    break

        if self._wrapper_config.return_proj_buf:
            return all_codes, proj_buf[:bsz].clone()
        return all_codes

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_distillation_weights(self) -> set[str]:
        """Load the configured RVQ-tail sidecar checkpoint.

        Returns:
            Fully qualified parameter names loaded into ``tail_distillation``.

        Raises:
            FileNotFoundError: If the configured sidecar does not exist.
            RuntimeError: If checkpoint keys do not match the configured model.
        """
        student = self.tail_distillation
        if student is None:
            return set()

        weights_path = Path(self._distillation_weights).expanduser()
        if not weights_path.is_absolute():
            model_root = Path(str(self._vllm_config.model_config.model)).expanduser()
            weights_path = model_root / weights_path if model_root.is_dir() else weights_path
        weights_path = weights_path.resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"RVQ-tail distillation checkpoint not found: {weights_path}")

        from safetensors.torch import load_file

        state = load_file(str(weights_path), device="cpu")
        prefix = "tail_distillation."
        if state and all(name.startswith(prefix) for name in state):
            state = {name[len(prefix) :]: value for name, value in state.items()}
        incompatible = student.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "RVQ-tail distillation checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        logger.info("Loaded RVQ-tail distillation checkpoint from %s", weights_path)
        return {f"tail_distillation.{name}" for name, _ in student.named_parameters()}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly (no fused projection remapping needed)."""
        loaded: set[str] = set()
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], w))
            else:
                other_weights.append((name, w))

        loaded_model = self.model.load_weights(model_weights)
        loaded |= {f"model.{n}" for n in loaded_model}

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, w in other_weights:
            param = params.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)

        return loaded
