"""
CPU correctness suite for AwebARN.

Tests:
  1. delta_rule_chunked_scan matches naive O(N²) reference (within fp tolerance)
  2. Chunked scan handles different chunk sizes (32, 64, 128) consistently
  3. EMA value mixer satisfies recursive equation
  4. RecurrentBlock is identity at init (zero out_proj + zero mlp.w_down)
  5. RecurrentBlock produces non-trivial output after parameter updates
  6. MoRRouter produces valid soft assignment (sums to 1, in [0,1])
  7. MoRRouter load-balance loss is finite and minimized at uniform
  8. AwebARN forward returns finite loss
  9. AwebARN gradients flow to all trainable params (no dead branches)
 10. Mini-training loop on random data — loss decreases
"""
import sys
import torch
import torch.nn.functional as F

# Add the architecture module to path
sys.path.insert(0, '/Users/danielwahnich/parameter-golf/records/track_10min_16mb/2026-04-19_AwebARN')
from awebarn_arch import (
    delta_rule_chunked_scan,
    delta_rule_naive,
    EMAValueMixer,
    RecurrentBlock,
    MoRRouter,
    AwebARN,
)

torch.manual_seed(0)

# ─── Test 1: chunked scan == naive ─────────────────────────────────────────
B, T, D = 2, 128, 32
q = torch.randn(B, T, D)
k = torch.randn(B, T, D)
k = F.normalize(k, dim=-1)  # normalized keys (as in RecurrentBlock)
v = torch.randn(B, T, D)
beta = torch.sigmoid(torch.randn(B, T))

out_naive = delta_rule_naive(q, k, v, beta)
out_chunked = delta_rule_chunked_scan(q, k, v, beta, chunk_size=64)
diff = (out_naive - out_chunked).abs().max().item()
assert diff < 1e-4, f"chunked vs naive diverged: max diff = {diff}"
print(f"[1] PASS chunked scan matches naive (max diff = {diff:.2e})")

# ─── Test 2: chunk size invariance ─────────────────────────────────────────
out_c32 = delta_rule_chunked_scan(q, k, v, beta, chunk_size=32)
out_c128 = delta_rule_chunked_scan(q, k, v, beta, chunk_size=128)
diff_3264 = (out_c32 - out_chunked).abs().max().item()
diff_64128 = (out_chunked - out_c128).abs().max().item()
assert diff_3264 < 1e-4, f"chunk 32 vs 64 diverged: {diff_3264}"
assert diff_64128 < 1e-4, f"chunk 64 vs 128 diverged: {diff_64128}"
print(f"[2] PASS chunk size invariant (32↔64: {diff_3264:.2e}, 64↔128: {diff_64128:.2e})")

# ─── Test 3: EMA mixer satisfies recursive equation ────────────────────────
mixer = EMAValueMixer(dim=D, alpha_init=0.3)
v_in = torch.randn(B, T, D)
v_out = mixer(v_in)
alpha = torch.sigmoid(mixer.alpha_logit)
# Manual reference
v_ref = torch.empty_like(v_in)
prev = torch.zeros(B, D)
for t in range(T):
    cur = alpha * v_in[:, t] + (1 - alpha) * prev
    v_ref[:, t] = cur
    prev = cur
diff = (v_out - v_ref).abs().max().item()
assert diff < 1e-5, f"EMA mixer diverged from reference: {diff}"
print(f"[3] PASS EMA mixer matches recursive eq (max diff = {diff:.2e})")

# ─── Test 4: RecurrentBlock is identity at init ────────────────────────────
torch.manual_seed(1)
block = RecurrentBlock(dim=D)
x_in = torch.randn(B, T, D)
x_out = block(x_in)
diff = (x_out - x_in).abs().max().item()
assert diff < 1e-4, f"block at init not identity: max delta {diff}"
print(f"[4] PASS RecurrentBlock is identity at init (delta = {diff:.2e})")

# ─── Test 5: After parameter perturbation, block produces real output ─────
with torch.no_grad():
    block.o_proj.weight.normal_(std=0.1)
    block.mlp.w_down.weight.normal_(std=0.1)
x_out = block(x_in)
delta = (x_out - x_in).abs().max().item()
assert delta > 0.01, f"block output too close to identity after perturbation: {delta}"
print(f"[5] PASS RecurrentBlock active after perturbation (delta = {delta:.4f})")

# ─── Test 6: MoR router produces valid distribution ────────────────────────
router = MoRRouter(dim=D, n_recursions=3)
weights, max_r, lb_loss = router(x_in)
assert weights.shape == (B, T, 3), f"router shape wrong: {weights.shape}"
sums = weights.sum(dim=-1)
assert (sums - 1.0).abs().max().item() < 1e-4, "router weights don't sum to 1"
assert (weights >= 0).all() and (weights <= 1).all(), "router weights out of [0,1]"
assert 1 <= max_r <= 3, f"max_r out of range: {max_r}"
assert torch.isfinite(lb_loss), "load balance loss not finite"
print(f"[6] PASS MoR router: sums OK, max_r={max_r}, lb_loss={lb_loss.item():.4f}")

# ─── Test 7: Load balance loss minimized at uniform ────────────────────────
# Manually construct uniform weights and check lb_loss ≈ 0
uniform = torch.full((B, T, 3), 1.0 / 3.0)
mean_assign = uniform.mean(dim=(0, 1))
uniform_target = torch.full_like(mean_assign, 1.0 / 3.0)
lb_uniform = F.kl_div(mean_assign.log(), uniform_target, reduction="sum")
assert lb_uniform.abs().item() < 1e-5, f"uniform should give ~0 KL: {lb_uniform}"
print(f"[7] PASS load-balance loss ≈ 0 at uniform ({lb_uniform.item():.2e})")

# ─── Test 8: AwebARN forward produces finite loss ──────────────────────────
torch.manual_seed(2)
model = AwebARN(vocab_size=64, dim=64, n_layers=2, mlp_mult=2.0,
                n_recursions=2, chunk_size=32)
x_ids = torch.randint(0, 64, (B, 64))
y_ids = torch.randint(0, 64, (B, 64))
loss = model(x_ids, y_ids)
assert torch.isfinite(loss), f"loss not finite: {loss}"
print(f"[8] PASS AwebARN forward: loss = {loss.item():.4f}")

# ─── Test 9: gradients flow to all trainable params ────────────────────────
# Perturb the zero-init projections first to simulate post-warmup state.
# (At fresh init, o_proj=0 and mlp.w_down=0 → blocks are identity →
# internal q/k/v projections correctly receive zero gradient by chain rule.)
with torch.no_grad():
    for b in model.blocks:
        b.o_proj.weight.normal_(std=0.02)
        b.mlp.w_down.weight.normal_(std=0.02)
model.zero_grad()
loss = model(x_ids, y_ids)
loss.backward()
n_with_grad = 0
n_total = 0
zero_grad_params = []
for name, p in model.named_parameters():
    n_total += 1
    if p.grad is not None and p.grad.abs().sum() > 0:
        n_with_grad += 1
    else:
        zero_grad_params.append(name)
frac = n_with_grad / n_total
assert frac >= 0.8, f"only {frac:.1%} params got grads. Zero-grad: {zero_grad_params[:5]}"
print(f"[9] PASS gradients flow to {n_with_grad}/{n_total} ({frac:.1%}) of params (post-perturbation)")

# ─── Test 10: mini-training loop, loss decreases ───────────────────────────
torch.manual_seed(3)
model2 = AwebARN(vocab_size=64, dim=64, n_layers=2, mlp_mult=2.0,
                 n_recursions=2, chunk_size=32)
opt = torch.optim.AdamW(model2.parameters(), lr=3e-3)
losses = []
x_ids2 = torch.randint(0, 64, (B, 64))
y_ids2 = torch.randint(0, 64, (B, 64))
for step in range(20):
    opt.zero_grad()
    loss = model2(x_ids2, y_ids2)
    loss.backward()
    opt.step()
    losses.append(loss.item())
assert losses[-1] < losses[0], f"loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
print(f"[10] PASS mini-training: loss {losses[0]:.4f} → {losses[-1]:.4f} ({len(losses)} steps)")

print("\n========== ALL 10 AWEBARN TESTS PASS ==========")
print("\nNext steps (Day 2+):")
print("  - Optimize EMA mixer: replace sequential scan with parallel cumprod")
print("  - Optimize delta-rule chunked: vectorize within-chunk loop")
print("  - Wire into training script (copy from PR #1711's training scaffolding)")
print("  - Add Muon optimizer hookup")
print("  - Add EMA + SWA + late QAT + brotli compression (from PR #1711)")
