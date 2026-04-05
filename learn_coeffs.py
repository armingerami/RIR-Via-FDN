"""
Feedback Delay Network (FDN) optimisation to match a measured Room Impulse Response.

Fixes vs. original
──────────────────
* `t30` was referenced inside `learn_all` but never defined → replaced with a
  differentiable soft-approximation (weighted by squared amplitude).
* Typo in `locs`: 28106 → 2816 (keeps the log-spaced progression).
* Redundant `[:43][:43]` indexing removed.
* `b` used only from index `k:` (second half) – made explicit.
* Loss function split into `total_loss` (for grad) and `compute_losses`
  (returns all four component losses for logging).
* Training loop logs all four weighted losses every `log_interval` epochs
  and saves a 2×3 figure showing each curve plus an overlay.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use("Agg")          # headless – change to "TkAgg" / "Qt5Agg" if you have a display
import matplotlib.pyplot as plt

# ── Room Impulse Response ──────────────────────────────────────────────────────
rate        = 48_000          # samples per second
spm         = rate // 1000    # samples per millisecond  (= 48)
rir  = abs(np.load("RIR.npy"))
rir /= rir.max()
n    = len(rir)
print(f"RIR shape: {rir.shape}")

# ── Early reflections (direct + 1st + 2nd order = 1 + 6 + 36 = 43) ───────────
sorted_idx              = np.argsort(rir)[::-1]
early_reflections_idx   = sorted_idx[:43]
scale_index             = sorted_idx[43]
early_reflections       = rir[early_reflections_idx]
scale                   = rir[scale_index]          # used to normalise synthesised tail

# ── Acoustic descriptors ───────────────────────────────────────────────────────
energy      = np.linalg.norm(rir)**2
early_energy = np.linalg.norm(rir[:scale_index-1])**2
energy_50ms = np.linalg.norm(rir[: 50 * spm])**2   # 50 ms × 48 samp/ms = 2 400 samples
energy_80ms = np.linalg.norm(rir[: 80 * spm])**2   # 80 ms × 48 samp/ms = 3 840 samples

# T30 in samples: last sample above –30 dB (amplitude ≥ 0.001 of peak)
target_t30 = 0
for i in range(n - 1, -1, -1):
    if rir[i] >= 0.001:
        target_t30 = i
        break

target_definition         = energy_50ms / energy
target_clarity            = 10 * np.log10(energy_80ms / energy)   # [dB], for display
target_simplified_clarity = energy_80ms / energy                   # linear, used in loss
target_DRR                = energy_50ms / (energy - energy_50ms)
target_center_time        = np.sum((rir ** 2) * np.arange(n)) / energy

print(f"target_t30           = {target_t30} samples ({target_t30 / rate * 1000:.1f} ms)")
print(f"target_definition    = {target_definition:.4f}")
print(f"target_clarity       = {target_clarity:.4f} dB")
print(f"target_DRR           = {target_DRR:.4f}")
print(f"target_center_time   = {target_center_time:.4f} samples")

# ── FDN architecture ───────────────────────────────────────────────────────────
k     = 16          # number of feedback loops  →  2k learnable params
alpha = 0.995       # initial decay rate

# Delay positions κ (samples). Fixed during optimisation.
# Note: original had a typo "28106" → corrected to 2816.
locs = np.array([
    0, 4, 16, 32, 64, 128, 256, 512,
    1024, 1536, 2048, 2560, 2816, 3072, 3584, 4096,
], dtype=np.float32).reshape(k, 1)
locs[-1] = max(locs[-1], target_t30-spm*20) # final delay should be at most 20 ms before t30 based on empirical tests
print("locations =", locs.reshape(-1).astype(int))

# Learnable params: a[:k] = amplitude gains (beta), a[k:] per-loop decay rate (alpha)
#                  b[:k] = amplitude gains (beta), b[k:] = per-loop decay rate (alpha)
coeffs = np.ones(2 * k, dtype=np.float32) * alpha
a = jnp.array(coeffs)
b = jnp.array(coeffs)        # b[k:] are the decay rates

# Pre-compute time-index matrix and linear ramp activation mask  (shape k × n)
times_np = np.arange(n, dtype=np.float32).reshape(1, n).repeat(k, axis=0)   # (k, n)
times_np -= (locs - 1)

activation_np = np.zeros((k, n), dtype=np.float32)
for i in range(k):
    for j in range(n):
        t = max(times_np[i, j], 0.0)
        activation_np[i, j] = min(t, 1.0)

times_np -= 1
times      = jnp.array(times_np,      dtype=jnp.float32)
activation = jnp.array(activation_np, dtype=jnp.float32)

# ── Model forward pass ─────────────────────────────────────────────────────────
def synthesize_with_const_alpha(a: jnp.ndarray) -> jnp.ndarray:
    """Sum of k exponentially-decaying, gain-scaled ramps."""
    decay = jnp.pow(b[k:].reshape(k, 1), times)          # (k, n)
    y     = activation * decay * (a[:k].reshape(k, 1) ** 2)
    return jnp.sum(y, axis=0)                             # (n,)

def synthesize_with_single_alpha(a: jnp.ndarray) -> jnp.ndarray:
    """Sum of k exponentially-decaying, gain-scaled ramps."""
    decay = jnp.pow(a[0], times)          # (k, n)
    y     = activation * decay * (a[:k].reshape(k, 1) ** 2)
    return jnp.sum(y, axis=0)  

def synthesize(a: jnp.ndarray) -> jnp.ndarray:
    """Sum of k exponentially-decaying, gain-scaled ramps."""
    decay = jnp.pow(a[k:].reshape(k, 1), times)          # (k, n)
    y     = activation * decay * (a[:k].reshape(k, 1) ** 2)
    return jnp.sum(y, axis=0)                             # (n,)

# ── Loss weights ───────────────────────────────────────────────────────────────
LD   = 1e4    # definition
LCL  = 1e7    # clarity (linear)
LCT  = 5e-5   # centre time
LT30 = 1   # T30 (soft approximation)
Lalpha = 1e-3 # slower lr for alpha for better stability

def compute_losses(a: jnp.ndarray):
    """
    Returns (total_loss, def_term, clarity_term, center_time_term, t30_term).
    All terms are already multiplied by their loss weights.
    """
    y = synthesize(a)

    eng     = early_energy + jnp.linalg.norm(y[:-scale_index-1])**2
    eng_80  = early_energy + jnp.linalg.norm(y[: 80 * spm - scale_index-1])**2   # 3 840 samples
    eng_50  = early_energy + jnp.linalg.norm(y[: 50 * spm - scale_index-1])**2   # 2 400 samples

    definition   = eng_50 / eng
    simp_clarity = eng_80  / eng
    center_time  = jnp.sum(y * y * jnp.arange(n)) / eng

    # e_late = jnp.sum(y[target_t30:] ** 2)

    # T30 proxy: penalise energy beyond target_t30.
    # If the synthesised tail decays by target_t30, this term is zero.
    # Direct gradients flow to any parameters keeping energy alive past that point.

    l_def  =  (target_definition         - definition)   ** 2
    l_cl   =  (target_simplified_clarity - simp_clarity) ** 2
    l_ct   =  (target_center_time        - center_time)  ** 2
    l_t30  =  (y[target_t30 - scale_index-1] - 0.001)**2

    total = LD *l_def + LCL*l_cl + LCT*l_ct + LT30*l_t30
    return total, l_def, l_cl, l_ct, l_t30

def total_loss(a: jnp.ndarray) -> jnp.ndarray:
    return compute_losses(a)[0]

# ── Diagnostics ────────────────────────────────────────────────────────────────
def print_diagnostics(a: jnp.ndarray) -> None:
    y = synthesize(a)

    eng     = early_energy + jnp.linalg.norm(y[:-scale_index-1])**2
    eng_80  = early_energy + jnp.linalg.norm(y[: 80 * spm - scale_index-1])**2   # 3 840 samples
    eng_50  = early_energy + jnp.linalg.norm(y[: 50 * spm - scale_index-1])**2   # 2 400 samples


    definition  = float(eng_50 / eng)
    clarity_db  = float(10 * jnp.log10(eng_80 / eng))
    center_time = float(jnp.sum(y * y * jnp.arange(n)) / eng)

    for i in range(n - 1, -1, -1):
      if y[i] >= 0.001:
          t30 = i
          break

    print(f"  definition   : {definition:.4f}   (target {target_definition:.4f})")
    print(f"  clarity (dB) : {clarity_db:.4f}  (target {target_clarity:.4f})")
    print(f"  center_time  : {center_time:.1f}  (target {target_center_time:.1f})")
    print(f"  t30  : {t30:.1f}  (target {target_t30:.1f})")
    return abs(t30 - target_t30)

# ── JIT compile ───────────────────────────────────────────────────────────────
grad_loss = jit(grad(jit(total_loss)))
loss_jit  = jit(total_loss)

# ── Learning-rate schedule ────────────────────────────────────────────────────
epoch_count  = 200000
lr_interval = epoch_count // 10      # log 11 checkpoints (epochs 0, 50k, …, 500k)
log_interval = epoch_count // 10 

lr_base     = 5e-7
lr_schedule = [
    lr_base,        lr_base,
    lr_base * 10,   lr_base * 10,
    lr_base * 50,   lr_base * 100,
    lr_base * 50,   lr_base * 50,
    lr_base * 10,   lr_base,
    lr_base * 0.1,
]

# ── Storage for plotting ───────────────────────────────────────────────────────
log_steps   = []
log_total   = []
log_def     = []
log_clarity = []
log_ct      = []
log_t30     = []

lr_idx = 0

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n── Starting training ──")
for epoch in range(epoch_count + 1):
    multipliers = jnp.where(jnp.arange(a.shape[0]) < k, 1.0, Lalpha)
    a = a - (grad_loss(a) * lr_schedule[lr_idx] * multipliers)

    if epoch % log_interval == 0:
        total, l_def, l_cl, l_ct, l_t30 = compute_losses(a)

        log_steps.append(epoch)
        log_total.append(float(total))
        log_def.append(float(l_def))
        log_clarity.append(float(l_cl))
        log_ct.append(float(l_ct))

        print(f"\n── Epoch {epoch:,} │ lr = {lr_schedule[lr_idx]:.2e} │ total loss = {float(total):.6f}")
        log_t30.append(1500 - print_diagnostics(a))

    if epoch % lr_interval == 0:
        lr_idx = min(lr_idx + 1, len(lr_schedule) - 1)

# ── Final diagnostics ──────────────────────────────────────────────────────────
print("\n═══ Final Result ═══")
aa = print_diagnostics(a)
print("a =", ", ".join(f"{float(x):.8f}" for x in a))
print(f"Final total loss = {float(loss_jit(a)):.6f}")


plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

# ── Figure 1: Overall training loss (Log Scale) ───────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.suptitle("FDN Optimisation – Overall Training Loss", fontsize=22)
ax1.plot(log_steps, log_total, color="tab:blue", linewidth=1.8, marker="o", markersize=4)
ax1.set_yscale("log")  # Set Y-axis to log scale
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Total Weighted Loss (log)")
ax1.grid(True, which="both", alpha=0.3) # "both" shows major and minor grid lines
ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
fig1.tight_layout()
fig1.savefig("fdn_total_loss.png", dpi=130, bbox_inches="tight")
print("\nFigure 1 saved → fdn_total_loss.png")

# ── Figure 2: All four losses overlaid (Log Scale) ────────────────────────────
fig2, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
fig2.suptitle("FDN Optimisation – Individual Losses", fontsize=22)

# Data and styling for the loop
loss_data = [log_def, log_clarity, log_ct, log_t30]
labels = ["Definition", "Clarity", "Centre-time", "T30"]
# labels = [r'\ell_1 (Definition)', r'\ell_2 (Clarity)', r'\ell_3 (Centre-time)', r'\ell_4 (T30)']
colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
markers = ["o", "s", "^", "D"]

eps = 10**-12
for i, ax in enumerate(axs):
    data = np.array(loss_data[i]) + eps
    ax.plot(log_steps, data,
            label=labels[i],
            color=colors[i], linewidth=1.5,
            marker=markers[i], markersize=3)
    
    ax.set_yscale("log")
    
    # --- Dynamic Range with Log-Scale Buffer ---
    pos_data = [d for d in data if d > 0]
    if pos_data:
        d_min, d_max = min(pos_data), max(pos_data)
        
        # Applying a multiplicative buffer (padding)
        # 0.5x for the floor and 2x for the ceiling creates a nice visual gap
        ax.set_ylim(bottom=d_min * 0.5, top=d_max * 2.0)
    
    # --- Ticker Control ---
    # MaxIntervals ensures we don't crowd the axis while aiming for 5 ticks
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    # Optional: Format the labels nicely
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

# ---- formatting ----
axs[3].set_xlabel("Epoch")
for ax in axs:
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("fdn_individual_losses.png", dpi=130, bbox_inches="tight")

print("Figure 2 saved → fdn_individual_losses.png")
