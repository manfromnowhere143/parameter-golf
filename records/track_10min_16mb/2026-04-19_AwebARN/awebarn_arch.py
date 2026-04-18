"""
AwebARN — Adaptive Recursion Net architecture.

Pure-PyTorch from-scratch implementation of:
  1. Delta-rule chunked linear-attention scan (no FLA / no Triton)
  2. EMA value mixing (Mega-style, Ma 2022)
  3. Mixture-of-Recursions per-token routing (MoR, 2025)

Designed for openai/parameter-golf submission, target <0.99 BPB at 27M params,
600s training budget on 8×H100 SXM.

Author: Daniel Wahnich (Aweb)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0):
        super().__init__()
        hidden = int(dim * mult)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ─────────────────────────────────────────────────────────────────────────────
# Delta-rule chunked scan (the heart of AwebARN)
# ─────────────────────────────────────────────────────────────────────────────

def delta_rule_chunked_scan(
    q: Tensor,           # (B, T, D)
    k: Tensor,           # (B, T, D)
    v: Tensor,           # (B, T, D)
    beta: Tensor,        # (B, T) — write-strength gate, in (0, 1]
    chunk_size: int = 64,
) -> Tensor:
    """
    Compute delta-rule recurrence:
        S_t = (I - β_t · q_t kᵀ_t) · S_{t-1} + β_t · v_t kᵀ_t
        o_t = S_t · q_t

    Implementation: split sequence into chunks of size C. Within each chunk,
    use the closed-form solution of the linear recurrence (cumulative product
    + Householder-style update). Across chunks, propagate carry state.

    Pure PyTorch — no Triton, no FLA. Compatible with torch.compile.

    Shapes:
        q, k, v: (B, T, D)
        beta: (B, T)
    Returns:
        o: (B, T, D)
    """
    B, T, D = q.shape
    assert T % chunk_size == 0, f"seq length {T} must be divisible by chunk_size {chunk_size}"
    C = chunk_size
    n_chunks = T // C

    # Reshape into chunks
    q_c = q.view(B, n_chunks, C, D)
    k_c = k.view(B, n_chunks, C, D)
    v_c = v.view(B, n_chunks, C, D)
    beta_c = beta.view(B, n_chunks, C)

    # Per-chunk attention update with delta correction
    # Within chunk, delta-rule reduces to:
    #   o_t = q_t · S_{0} + sum_{s≤t} q_t · (β_s v_s k_sᵀ - β_s S_{s-1} q_s k_sᵀ k_s) ...
    # The common GDN trick is: enforce ||k_t||=1 (normalized keys), then
    # the second-order cross-term simplifies. We approximate the chunked update
    # via a parallel-prefix scan over (β k v) pairs.

    # Initialize state S_0 = 0 of shape (B, D, D)
    S = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)

    out_chunks = []
    for c in range(n_chunks):
        qc = q_c[:, c]   # (B, C, D)
        kc = k_c[:, c]
        vc = v_c[:, c]
        bc = beta_c[:, c]  # (B, C)

        # Within-chunk: sequential update (small C=64, so loop is cheap)
        # In production this becomes a Kasai-style parallel scan.
        chunk_out = torch.empty_like(qc)
        S_curr = S
        for t in range(C):
            q_t = qc[:, t]                           # (B, D)
            k_t = kc[:, t]
            v_t = vc[:, t]
            b_t = bc[:, t].unsqueeze(-1)             # (B, 1)
            # Delta update: S_t = S_{t-1} + β · (v - S_{t-1} k) ⊗ k
            # Equivalent to: S_t = (I - β k kᵀ) S_{t-1} + β v kᵀ
            Sk = torch.einsum("bij,bj->bi", S_curr, k_t)  # (B, D)
            update = (v_t - Sk).unsqueeze(-1) * (b_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            S_curr = S_curr + update                       # (B, D, D)
            o_t = torch.einsum("bij,bj->bi", S_curr, q_t)
            chunk_out[:, t] = o_t

        out_chunks.append(chunk_out)
        S = S_curr  # carry state to next chunk

    return torch.cat(out_chunks, dim=1)  # (B, T, D)


def delta_rule_naive(q: Tensor, k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    """
    Reference O(N²) implementation for correctness checks.
    Same equation, no chunking, no parallel tricks.
    """
    B, T, D = q.shape
    S = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        b_t = beta[:, t].unsqueeze(-1)
        Sk = torch.einsum("bij,bj->bi", S, k_t)
        update = (v_t - Sk).unsqueeze(-1) * (b_t.unsqueeze(-1) * k_t.unsqueeze(-2))
        S = S + update
        out[:, t] = torch.einsum("bij,bj->bi", S, q_t)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EMA value mixer (Mega-style)
# ─────────────────────────────────────────────────────────────────────────────

class EMAValueMixer(nn.Module):
    """
    Mega-style exponential moving average over value vectors.
    v_t' = α * v_t + (1 - α) * v_{t-1}'
    Provides smoothing prior; α is a learnable per-channel scalar in (0, 1).
    """
    def __init__(self, dim: int, alpha_init: float = 0.1):
        super().__init__()
        # Parameterize as logit so we can apply sigmoid → α ∈ (0,1)
        logit_init = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.full((dim,), logit_init))

    def forward(self, v: Tensor) -> Tensor:
        """v: (B, T, D) → (B, T, D)"""
        alpha = torch.sigmoid(self.alpha_logit)  # (D,)
        # Recursive EMA — but we can vectorize via a per-channel scan
        # For correctness here, do a sequential scan (will optimize later)
        B, T, D = v.shape
        out = torch.empty_like(v)
        prev = torch.zeros(B, D, device=v.device, dtype=v.dtype)
        for t in range(T):
            cur = alpha * v[:, t] + (1.0 - alpha) * prev
            out[:, t] = cur
            prev = cur
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent Block (the unit that gets re-run by MoR)
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentBlock(nn.Module):
    def __init__(self, dim: int, mlp_mult: float = 2.0, ema_alpha_init: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.beta_proj = nn.Linear(dim, 1, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ema_mixer = EMAValueMixer(dim, alpha_init=ema_alpha_init)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult)
        # Zero-init output projection so block starts as identity
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.mlp.w_down.weight)

    def forward(self, x: Tensor, chunk_size: int = 64) -> Tensor:
        # Linear-recurrent attention with delta rule
        h = self.norm1(x)
        q = self.q_proj(h)
        k = F.normalize(self.k_proj(h), dim=-1)  # normalize keys for stable delta
        v = self.v_proj(h)
        v = self.ema_mixer(v)                    # Mega-style EMA on values
        beta = torch.sigmoid(self.beta_proj(h)).squeeze(-1)
        attn_out = delta_rule_chunked_scan(q, k, v, beta, chunk_size=chunk_size)
        x = x + self.o_proj(attn_out)
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Mixture-of-Recursions Router
# ─────────────────────────────────────────────────────────────────────────────

class MoRRouter(nn.Module):
    """
    Per-token router that decides recursion depth r_t ∈ {1, 2, 3}.
    Soft-top-1 with temperature anneal. Returns:
        weights: (B, T, 3) — soft assignment
        max_r: int — max recursion depth in this batch (for compute budget)
        load_balance_loss: scalar — KL(mean_r || uniform), to add to training loss
    """
    def __init__(self, dim: int, n_recursions: int = 3, temperature_init: float = 2.0):
        super().__init__()
        self.n_recursions = n_recursions
        self.proj = nn.Linear(dim, n_recursions, bias=True)
        self.register_buffer("temperature", torch.tensor(temperature_init))
        nn.init.zeros_(self.proj.weight)
        # Slight bias towards r=1 at init (cheaper); linear ramp from +0.5 to -0.5
        with torch.no_grad():
            if n_recursions == 1:
                bias = torch.zeros(1)
            else:
                bias = torch.linspace(0.5, -0.5, n_recursions)
            self.proj.bias.copy_(bias)

    def forward(self, x: Tensor) -> tuple[Tensor, int, Tensor]:
        # x: (B, T, D)
        logits = self.proj(x.detach()) / self.temperature  # (B, T, R)
        weights = F.softmax(logits, dim=-1)
        # Top-1 hard assignment for forward, soft for backward (straight-through)
        hard = F.one_hot(weights.argmax(-1), self.n_recursions).to(weights.dtype)
        weights_st = hard + (weights - weights.detach())
        # Load balance: KL(mean assignment || uniform)
        mean_assign = weights.mean(dim=(0, 1))  # (R,)
        uniform = torch.full_like(mean_assign, 1.0 / self.n_recursions)
        lb_loss = F.kl_div(mean_assign.log(), uniform, reduction="sum")
        # Max recursion depth in batch
        max_r = int(weights_st.argmax(-1).max().item()) + 1
        return weights_st, max_r, lb_loss


# ─────────────────────────────────────────────────────────────────────────────
# AwebARN main model
# ─────────────────────────────────────────────────────────────────────────────

class AwebARN(nn.Module):
    """
    Adaptive Recursion Net: stack of RecurrentBlocks where each block can be
    re-run 1-3 times per token via MoR routing.
    """
    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 448,
        n_layers: int = 5,
        mlp_mult: float = 2.0,
        n_recursions: int = 3,
        chunk_size: int = 64,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        self.load_balance_weight = load_balance_weight
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            RecurrentBlock(dim, mlp_mult=mlp_mult)
            for _ in range(n_layers)
        ])
        self.routers = nn.ModuleList([
            MoRRouter(dim, n_recursions=n_recursions) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        # Tied LM head — use tok_emb.weight in forward
        # Init
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None):
        x = self.tok_emb(input_ids)
        total_lb_loss = 0.0
        for block, router in zip(self.blocks, self.routers):
            weights, max_r, lb_loss = router(x)  # (B, T, R)
            total_lb_loss = total_lb_loss + lb_loss
            # Run the block 1-max_r times, accumulating weighted outputs
            x_running = x
            x_acc = torch.zeros_like(x)
            for r in range(max_r):
                x_running = block(x_running, chunk_size=self.chunk_size)
                # Add this recursion's contribution weighted by router's prob for r+1+
                w_r = weights[..., r:].sum(dim=-1, keepdim=True)  # tokens that want ≥ r+1 recursions
                x_acc = x_acc + w_r * x_running
            x = x_acc / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        if target_ids is None:
            return logits
        ce_loss = F.cross_entropy(logits.float().reshape(-1, self.vocab_size),
                                   target_ids.reshape(-1), reduction="mean")
        return ce_loss + self.load_balance_weight * total_lb_loss
