# AwebARN — Adaptive Recursion Net (from-scratch original architecture)

**Mission**: Build a genuinely original architecture for openai/parameter-golf that does NOT use the FLA library, NOT use any prior PR's code, and could plausibly beat GatedDeltaNet at 1.00 BPB.

**Honest probability** (from research): **15% to win, 30% to top-5, 35% mid, 20% regress**.

**Why this exists alongside AwebEMT**: AwebEMT (GDN+distillation) is the high-EV "win the competition" play (65% top-3). AwebARN is the **publishable original architecture** play — even if it lands at #5-10 in the competition, the architecture itself is a defensible novel contribution worth a writeup independent of rank.

---

## Architecture overview

```
                     ┌───────────────────────────────────────┐
        x_t ────────►│ Router (linear, d→3 logits)           │
                     │ → r_t ∈ {1,2,3}, top-1, soft-EMA mix  │
                     └────────────┬──────────────────────────┘
                                  │ r_t
                ┌─────────────────▼──────────────────────────┐
   recur 1..r_t │  RecurrentBlock (shared across recursions) │
                │  ┌──────────────────────────────────────┐  │
                │  │ RMSNorm                              │  │
                │  │ qkv proj                             │  │
                │  │ Delta-rule chunked scan (chunk=64):  │  │
                │  │    S_t = (I − β q_t kᵀ_t)·S_{t−1}    │  │
                │  │           + β v_t kᵀ_t               │  │
                │  │    o_t  = S_t · q_t                  │  │
                │  │ EMA value mix:  v ← α·v + (1−α)·v_prev│ │
                │  │ SwiGLU MLP (mult=2.0)                │  │
                │  └──────────────────────────────────────┘  │
                └────────────────┬───────────────────────────┘
                                 │
                       Output residual + token-shift skip
```

## Three novel mechanisms (none combined in any PR)

1. **Pure-PyTorch delta-rule chunked scan** — implements the GatedDeltaNet recurrence
   `S_t = (I − β q_t kᵀ_t)·S_{t−1} + β v_t kᵀ_t` without the FLA library. Uses a
   chunked Kasai-form parallel scan (chunk size 64) for GPU efficiency without
   custom Triton kernels.

2. **EMA value mixing** (Mega-style, Ma 2022) — adds a per-layer EMA over value
   vectors as a free regularization signal. `v_t ← α·v_t + (1−α)·v_{t−1}`. Provides
   smoothing prior. Zero PRs use this in nanochat speedrun.

3. **Mixture-of-Recursions routing** (MoR, Google 2025, arXiv:2507.10524) — per-token
   linear router decides whether each token re-runs the recurrent block 1×, 2×, or 3×.
   Average ~1.8× → effective depth ~7 with a 4-layer parameter footprint. One PR
   tried hybrid HELIX MoR (no record); zero PRs apply MoR to a pure recurrent backbone.

## Param budget (target 27M, max 16MB int6/int8)

| Component | Params |
|---|---|
| Embedding 1024 vocab × 448 dim | 0.46M |
| 5 × RecurrentBlock (qkv 3×448², SwiGLU 2.0×) | ~5.0M each = 25.0M |
| Router (5 × 448→3) | 0.007M |
| Tied LM head | 0 |
| LN + bias | ~0.1M |
| EMA value mixers (per layer) | 0.05M |
| **Total** | ~25.6M |

## Wall-clock estimate

| Phase | Time |
|---|---|
| Single recursion per layer (chunked scan) | ~22ms |
| Avg 1.8 recursions × 5 layers | ~198ms/step |
| 600s budget → steps | ~3000 |
| TTT eval | ~80s (same as PR #1711) |

## Risks (brutal honesty)

| Risk | Probability | Mitigation |
|---|---|---|
| Pure-PyTorch scan is 2-3× slower than FLA Triton | High | Use torch.compile + bf16; if still too slow, drop to 4 layers |
| MoR router collapses to constant r_t=1 (all tokens same) | Med | Aux load-balance loss (KL to uniform) at λ=0.01 |
| MoR router collapses to constant r_t=3 (all tokens deep) | Med | Same load-balance loss + temperature anneal 2.0→0.5 |
| EMA value mixing destabilizes early training | Low | α schedule: warm 0.0 → 0.1 over first 500 steps |
| Delta-rule scan numerical issues at long context | Med | Chunk size 64 (proven by GDN); enforce β ≤ 1 |

## Engineering plan

| Phase | Hours | GPU $ | Status |
|---|---|---|---|
| 1. Custom delta-rule chunked scan + correctness vs naive O(N²) | 6 | 0 | TODO |
| 2. RecurrentBlock + EMA value mixer | 3 | 0 | TODO |
| 3. MoR router + load-balance + soft top-1 | 4 | 0 | TODO |
| 4. Integration into training loop + Muon hookup | 3 | 0 | TODO |
| 5. CPU correctness suite (10+ tests) | 2 | 0 | TODO |
| 6. Smoke run 1×H100, 200s | 0 | $3 | TODO |
| 7. Tune α-init, recursion budget, router τ | 0 | $15 | TODO |
| 8. 3-seed validation 8×H100 | 0 | $40 | TODO |
| **Total** | **18h** | **~$58** | |

## Scope discipline

This is a multi-day mission. Tonight's deliverable: scaffolding only.
- ✅ Folder + research doc
- 🟡 Delta-rule scan stub
- 🟡 RecurrentBlock skeleton
- 🟡 CPU correctness test for the scan
- ⏳ Full training script (Day 2)
- ⏳ MoR router (Day 2)
- ⏳ GPU smoke (Day 3)

## Credits & citations

This architecture combines mechanisms from prior published work:

- **Delta rule** (Schlag et al. 2021, arXiv:2102.11174 "Linear Transformers Are Secretly Fast Weight Programmers") — the recurrent state update equation
- **GatedDeltaNet variant** (Yang et al. 2024) — for inspiration; we implement from scratch
- **EMA value mixing** (Ma et al. 2022, arXiv:2209.10655 "Mega: Moving Average Equipped Gated Attention") — Mega's exponential moving average
- **Mixture of Recursions** (2025, arXiv:2507.10524) — adaptive per-token recursion depth
- **Muon optimizer** (lineage from PR #549, abaybektursun) — kept as the optimizer base

## Why this architecture deserves to exist regardless of leaderboard outcome

Even if AwebARN places at #5-10 instead of winning:

1. **First implementation** of MoR routing on a pure-PyTorch recurrent backbone in nanochat speedrun
2. **First** application of Mega-style EMA value mixing in this competition
3. **Reference implementation** of delta-rule scan without FLA dependency — useful for environments where Triton is unavailable
4. **Clean, self-contained PyTorch code** (~600 LoC total) usable as a research artifact

The competition is a vehicle; the architecture is the contribution.

---

*Author*: Daniel Wahnich (Aweb)
*Branch*: `aweb-arn`
*Status*: Day 1 scaffolding in progress
