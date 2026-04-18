# Aweb-EMT — GatedDeltaNet + EMA-Teacher Distillation

**Aweb's submission for openai/parameter-golf**, track_10min_16mb.

## Goal

Beat the live frontier (PR #1711, 1.00980 BPB) by adding a genuinely novel mechanism on top of the GatedDeltaNet (FLA) backbone.

## Aweb's contribution: EMA-Teacher Distillation (EMT)

We apply the **Mean Teacher framework** (Tarvainen & Valpola, NeurIPS 2017) to nanochat speedrun training, replacing the standard cross-entropy loss with a soft mixture:

```
L = (1 - α) · CE(student_logits, target)  +  α · KL(student_logits || teacher_logits.detach())
```

The **teacher** is a separate copy of the student model whose weights are periodically synchronized from the EMA-smoothed state (already maintained in the frontier code at `EMA_DECAY=0.997`). The teacher provides a soft-label regularization signal that is more stable than the student's own logits, especially in late training when the student begins to overfit individual minibatches.

### Schedule

- **α** ramps linearly from `0.0` → `EMT_ALPHA_MAX` (default 0.3) between steps `EMT_WARMUP_START_FRAC × iterations` (default 30% in) and `EMT_WARMUP_END_FRAC × iterations` (default 70% in), then holds at `EMT_ALPHA_MAX` until end.
- **Teacher refresh**: every `EMT_TEACHER_UPDATE_EVERY` steps (default 16), the teacher's weights are reloaded from `ema_state`. This avoids per-step synchronization cost while keeping the teacher current.
- **Temperature** `EMT_TEMPERATURE` (default 1.0) scales both student and teacher logits before softmax, with KL multiplied by `T²` per the standard Hinton soft-target formulation.

### Why this hasn't been done in the competition

Verified via `gh search prs --repo openai/parameter-golf` for terms `mean teacher`, `EMA teacher`, `distillation`, `KL`, `soft targets`. Zero matches in the open-PR history. The frontier uses EMA only for weight averaging at eval time; using it as a soft-label teacher *during* training is genuinely novel here.

### Why it's legal

- **Issue #1017 condition 1 (causal)**: ✓ Teacher forward is the same causal pass as the student.
- **Issue #1017 condition 2 (full distribution)**: ✓ KL operates on full softmax over vocab.
- **Issue #1017 condition 3 (score-before-update)**: ✓ Distillation is a *training* technique. No eval-time changes. TTT remains legal score-first.
- **Issue #1017 condition 4 (single L→R pass)**: ✓ No rescoring. Single eval pass, unchanged from base.

## Stack

| Component | Source | Notes |
|---|---|---|
| Backbone | **GatedDeltaNet (FLA) K_KVShare_Wider** | PR #1687 (resouer), used by frontier PR #1711 |
| Layers | 10L, 544d, GQA 8H/4KV | inherited from base |
| Optimizer | Muon (mom=0.95→0.99) + AdamW for embeddings | inherited |
| Weight averaging | EMA(0.997) + SWA | inherited; EMA also feeds the teacher |
| TTT | Legal Score-First SGD (3 epochs, lr=0.005, freeze 2 blocks) | PR #461 (Christopher-Lee-McClendon) |
| Quantization | Int6 GPTQ matrices + Int8 embeddings | inherited |
| Compression | Brotli-11 | PR #1711 (aamodbhatt) |
| Eval | Sliding window stride=64 + Score-First TTT | inherited |
| **Aweb signature** | **EMA-Teacher Distillation** | **NEW** — Tarvainen 2017 applied to nanochat |

## Reproduction

```bash
SEED=1337 \
EMT_ENABLED=1 \
EMT_ALPHA_MAX=0.3 \
EMT_WARMUP_START_FRAC=0.3 \
EMT_WARMUP_END_FRAC=0.7 \
EMT_TEACHER_UPDATE_EVERY=16 \
EMT_TEMPERATURE=1.0 \
TTT_ENABLED=1 \
ARCH_MODE=A \
torchrun --standalone --nproc_per_node=8 train_gdn_7k.py
```

To reproduce the un-distilled baseline (PR #1711 equivalent):
```bash
EMT_ENABLED=0 SEED=1337 TTT_ENABLED=1 ARCH_MODE=A \
torchrun --standalone --nproc_per_node=8 train_gdn_7k.py
```

## Credits & lineage

Aweb-EMT stands on the work of:

- **Mean Teacher framework**: Tarvainen, A. & Valpola, H. (NeurIPS 2017), "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)
- **GatedDeltaNet backbone**: Yang et al. 2024 + the [flash-linear-attention library](https://github.com/sustcsonglin/flash-linear-attention) (sustcsonglin); architecture variant from **PR #1687** by @resouer
- **GDN integration into parameter-golf**: **PR #1711** by @aamodbhatt and **PR #1698** by @arsenis-cmd
- **Score-First Legal TTT**: **PR #461** by @Christopher-Lee-McClendon, with patterns from PRs #1416 (@erichroepke) and #1423 (@aryanbhosale)
- **Brotli-11 compression**: **PR #1711** (aamodbhatt) — saves ~900KB vs zstd-22
- **Mixed Int6/Int8 GPTQ + late QAT pattern**: PR #549 (abaybektursun) lineage

## Aweb originality

The contribution of this submission is the **adaptation of EMA-Teacher Distillation to small-LLM training in nanochat speedrun** — the schedule design (α-warmup window, teacher refresh cadence, temperature handling), the integration with the FLA-based GDN backbone, and the empirical validation. The base architecture and most engineering features are the work of the cited prior PRs and are credited above.

## Author

Daniel Wahnich (@manfromnowhere143) — Aweb
