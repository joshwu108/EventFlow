"""Event embedding and causal Transformer context encoder.

Input embedding of event ``t`` (all terms are ``d_model`` vectors, summed)::

    e_t = sum_k Emb_k(c_{t,k})                      categorical fields
        + W_c [x_t * m_t, m_t]                      continuous values and their masks
        + Emb_inst(instrument)                       instrument identity
        + P_t                                        learned position in the window
        + W_tau phi(log1p(tau_t / s))                Fourier features of session time

where ``m_t`` is the continuous mask (masked values are zeroed *and* the mask
itself is an input, so "missing" is distinguishable from "zero") and ``s`` is
the training sessions' median duration. The encoder output ``h_t`` summarises
events ``0..t`` only; :func:`eventflow.tokenization.windows.causal_mask` and
the key padding mask enforce this.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from eventflow.models.config import ModelConfig
from eventflow.tokenization.tokenizer import FieldLayout
from eventflow.tokenization.windows import EventBatch, causal_mask


class EventEmbedding(nn.Module):
    def __init__(
        self, layout: FieldLayout, config: ModelConfig, num_instruments: int, time_scale: float
    ):
        super().__init__()
        d = config.d_model
        self.categorical = nn.ModuleList(nn.Embedding(size, d) for size in layout.categorical_sizes)
        self.continuous = nn.Linear(2 * len(layout.continuous), d)
        self.instrument = nn.Embedding(max(num_instruments, 1), d)
        self.position = nn.Embedding(config.max_context, d)
        self.time_frequencies = config.time_frequencies
        self.time_scale = float(time_scale)
        if self.time_frequencies:
            self.register_buffer(
                "freqs",
                math.pi * 2.0 ** torch.arange(self.time_frequencies, dtype=torch.float32),
                persistent=False,
            )
            self.time = nn.Linear(2 * self.time_frequencies, d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.max_context = config.max_context

    def forward(self, batch: EventBatch) -> torch.Tensor:
        B, T, _ = batch.categorical.shape
        if self.max_context < T:
            raise ValueError(f"sequence length {T} exceeds max_context {self.max_context}")
        e = sum(emb(batch.categorical[..., k]) for k, emb in enumerate(self.categorical))
        mask = batch.continuous_mask.float()
        e = e + self.continuous(torch.cat([batch.continuous * mask, mask], dim=-1))
        e = e + self.instrument(batch.instrument)[:, None, :]
        e = e + self.position(torch.arange(T, device=e.device))[None]
        if self.time_frequencies:
            # log1p compresses multi-hour sessions; the scale makes it dataset-agnostic.
            tau = torch.log1p(batch.session_time.float().clamp_min(0) / self.time_scale)
            args = tau[..., None] * self.freqs
            e = e + self.time(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))
        return self.dropout(self.norm(e))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None], dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))``."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), allowed))
        return x + self.ffn(self.ln2(x))


class ContextEncoder(nn.Module):
    """``EventBatch -> [B, T, d_model]`` causal summaries of each prefix."""

    def __init__(
        self, layout: FieldLayout, config: ModelConfig, num_instruments: int, time_scale: float
    ):
        super().__init__()
        self.embedding = EventEmbedding(layout, config, num_instruments, time_scale)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def attention_mask(valid: torch.Tensor) -> torch.Tensor:
        """``[B, T, T]`` True where query ``i`` may attend key ``j``.

        Allowed iff ``j <= i`` and key ``j`` is a real event. The diagonal is
        always allowed so that padded query rows never have an empty softmax
        (their outputs are excluded from every loss and sample).
        """
        T = valid.shape[1]
        blocked = causal_mask(T, valid.device)[None] | ~valid[:, None, :]
        eye = torch.eye(T, dtype=torch.bool, device=valid.device)[None]
        return ~blocked | eye

    def forward(self, batch: EventBatch) -> torch.Tensor:
        x = self.embedding(batch)
        allowed = self.attention_mask(batch.valid)
        for block in self.blocks:
            x = block(x, allowed)
        return self.final_norm(x)
