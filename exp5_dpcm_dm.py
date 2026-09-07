"""
Experiment 5 -- DPCM and Delta Modulation
==========================================
Implements:
  14. First-order DPCM predictor
  15. PCM vs DPCM prediction/quantization error comparison
  16. Delta modulation for small/moderate/large step sizes
  17. Slowly vs rapidly varying test inputs
  18. (Optional) Adaptive delta modulation

Produces the four required visualizations:
  - Original/predicted samples
  - Prediction error
  - Delta staircase
  - MSE vs step size

Design note: prediction, quantization, and reconstruction are kept as
explicit, separate steps (predict -> error -> quantize -> reconstruct)
so each stage can be inspected independently, as required by the prompt.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 110

rng = np.random.default_rng(42)
OUT = "/mnt/user-data/outputs"

# ---------------------------------------------------------------------
# 1. Test signals: slowly varying and rapidly varying
# ---------------------------------------------------------------------
Fs = 8000                      # sampling rate (Hz)
T = 0.05                       # duration (s) -> 400 samples, easy to inspect
n = np.arange(int(Fs * T))
t = n / Fs

f_slow = 40.0                  # slowly varying (relative to Fs) -> low max slope
f_fast = 1500.0                # rapidly varying -> high max slope

x_slow = 0.85 * np.sin(2 * np.pi * f_slow * t)
x_fast = 0.9 * np.sin(2 * np.pi * f_fast * t) + 0.15 * np.sin(2 * np.pi * (f_fast * 2.3) * t)

# Slope-overload condition for linear DM: no overload iff delta*Fs >= max|dx/dt|.
# max|dx/dt|(slow)  = 0.85*2*pi*40   ~ 213.6  -> need delta >= 0.0267 to avoid overload
# max|dx/dt|(fast)  ~ 0.9*2*pi*1500 + 0.15*2*pi*3450 ~ 11730 -> need delta >= 1.47 (never met below)
# This is exactly what makes "slow" a clean granular-noise demo and "fast" a clean
# slope-overload demo across the tested step sizes.

signals = {"slow": x_slow, "fast": x_fast}

# ---------------------------------------------------------------------
# 2. Uniform mid-rise quantizer (used for PCM and for the DPCM error signal)
# ---------------------------------------------------------------------
def uniform_quantize(x, n_bits, xmax):
    """Mid-rise uniform quantizer over [-xmax, xmax]."""
    levels = 2 ** n_bits
    step = 2 * xmax / levels
    idx = np.floor((x + xmax) / step)
    idx = np.clip(idx, 0, levels - 1)
    xq = -xmax + (idx + 0.5) * step
    return xq, step

# ---------------------------------------------------------------------
# 3. PCM encode/decode (direct quantization of the signal itself)
# ---------------------------------------------------------------------
def pcm_encode(x, n_bits, xmax):
    xq, step = uniform_quantize(x, n_bits, xmax)
    return xq, step

# ---------------------------------------------------------------------
# 4. First-order DPCM: predictor, quantizer, and reconstruction kept separate
#    Predictor: xhat[n] = xr[n-1]   (first-order, previous reconstructed sample)
# ---------------------------------------------------------------------
def dpcm_encode_decode(x, n_bits, err_max):
    N = len(x)
    xhat = np.zeros(N)     # predicted samples
    e = np.zeros(N)        # prediction error (pre-quantization)
    eq = np.zeros(N)       # quantized prediction error
    xr = np.zeros(N)       # reconstructed samples
    step = 2 * err_max / (2 ** n_bits)

    xr_prev = 0.0
    for i in range(N):
        xhat[i] = xr_prev                       # PREDICTION
        e[i] = x[i] - xhat[i]                    # ERROR
        eq[i], _ = uniform_quantize(np.array([e[i]]), n_bits, err_max)  # QUANTIZATION
        eq[i] = eq[i][0] if hasattr(eq[i], "__len__") else eq[i]
        xr[i] = xhat[i] + eq[i]                   # RECONSTRUCTION
        xr_prev = xr[i]

    return xhat, e, eq, xr, step

# fix scalar quantize call (uniform_quantize expects arrays) -----------
def uniform_quantize_scalar(v, n_bits, xmax):
    levels = 2 ** n_bits
    step = 2 * xmax / levels
    idx = np.floor((v + xmax) / step)
    idx = np.clip(idx, 0, levels - 1)
    vq = -xmax + (idx + 0.5) * step
    return vq, step

def dpcm_encode_decode_v2(x, n_bits, err_max):
    N = len(x)
    xhat = np.zeros(N)
    e = np.zeros(N)
    eq = np.zeros(N)
    xr = np.zeros(N)
    xr_prev = 0.0
    step = None
    for i in range(N):
        xhat[i] = xr_prev
        e[i] = x[i] - xhat[i]
        eq[i], step = uniform_quantize_scalar(e[i], n_bits, err_max)
        xr[i] = xhat[i] + eq[i]
        xr_prev = xr[i]
    return xhat, e, eq, xr, step

# ---------------------------------------------------------------------
# 5. Delta modulation (1-bit DPCM): fixed step size, optional adaptive
# ---------------------------------------------------------------------
def delta_modulate(x, delta, adaptive=False, grow=1.5, shrink=1.0/1.5,
                    delta_min=None, delta_max=None):
    """
    Linear (adaptive=False) or adaptive delta modulation.
    Returns: xr (reconstructed staircase), bits (+1/-1), delta_trace (step
    used at each sample -- constant for linear DM, varying for adaptive DM),
    step_taken (xr[n]-xr[n-1], should be exactly +-delta_trace[n]).
    """
    N = len(x)
    xr = np.zeros(N)
    bits = np.zeros(N)
    delta_trace = np.zeros(N)
    xr_prev = 0.0
    d = delta
    prev_bit = 1
    for i in range(N):
        b = 1.0 if x[i] >= xr_prev else -1.0
        if adaptive:
            if i > 0 and b == prev_bit:
                d = min(d * grow, delta_max) if delta_max else d * grow
            else:
                d = max(d * shrink, delta_min) if delta_min else d * shrink
        delta_trace[i] = d
        xr[i] = xr_prev + d * b
        bits[i] = b
        xr_prev = xr[i]
        prev_bit = b
    step_taken = np.diff(xr, prepend=0.0)
    step_taken[0] = xr[0]  # first step from 0
    return xr, bits, delta_trace, step_taken

# ---------------------------------------------------------------------
# 6. Mandatory validation: every DM output step must be exactly +-delta
# ---------------------------------------------------------------------
def validate_delta_steps(xr, delta_trace, tol=1e-9):
    steps = np.diff(xr)
    expected = delta_trace[1:]
    ok = np.all(np.isclose(np.abs(steps), expected, atol=tol))
    max_dev = np.max(np.abs(np.abs(steps) - expected))
    return ok, max_dev

# =======================================================================
# RUN: Task 14/15 -- PCM vs DPCM on the slowly-varying signal
# =======================================================================
x = x_slow
n_bits = 4
xmax_pcm = 1.0
xmax_err = 0.4   # DPCM error signal has much smaller dynamic range

xq_pcm, step_pcm = pcm_encode(x, n_bits, xmax_pcm)
xhat, e, eq, xr_dpcm, step_dpcm = dpcm_encode_decode_v2(x, n_bits, xmax_err)

mse_pcm = np.mean((x - xq_pcm) ** 2)
mse_dpcm = np.mean((x - xr_dpcm) ** 2)

# --- Visualization A: original / predicted samples --------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t * 1e3, x, label="Original x[n]", lw=1.6)
ax.plot(t * 1e3, xhat, label="DPCM predicted x_hat[n]", lw=1.2, ls="--")
ax.plot(t * 1e3, xr_dpcm, label="DPCM reconstructed x_r[n]", lw=1.0, alpha=0.8)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude")
ax.set_title(f"Original vs Predicted vs Reconstructed (DPCM, {n_bits}-bit, slow signal)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/A_original_predicted.png")
plt.close(fig)

# --- Visualization B: prediction error --------------------------------
fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axs[0].plot(t * 1e3, e, label="Prediction error e[n] (pre-quant.)", color="C1")
axs[0].plot(t * 1e3, eq, label="Quantized error e_q[n]", color="C3", alpha=0.7)
axs[0].set_ylabel("Error amplitude")
axs[0].legend(fontsize=8); axs[0].set_title("DPCM prediction error vs quantized error")

pcm_err = x - xq_pcm
dpcm_err = x - xr_dpcm
axs[1].plot(t * 1e3, pcm_err, label=f"PCM error (MSE={mse_pcm:.2e})", color="C0")
axs[1].plot(t * 1e3, dpcm_err, label=f"DPCM error (MSE={mse_dpcm:.2e})", color="C2")
axs[1].set_xlabel("Time (ms)"); axs[1].set_ylabel("Error amplitude")
axs[1].legend(fontsize=8); axs[1].set_title("Overall reconstruction error: PCM vs DPCM")
fig.tight_layout()
fig.savefig(f"{OUT}/B_prediction_error.png")
plt.close(fig)

# =======================================================================
# RUN: Task 16/17 -- Delta modulation, small/moderate/large step, slow/fast input
# =======================================================================
xmax_signal = max(np.max(np.abs(x_slow)), np.max(np.abs(x_fast)))
delta_small = 0.05      # > 0.0267 -> tracks the slow signal, shows granular noise only
delta_moderate = 0.15
delta_large = 0.45
deltas = {"small": delta_small, "moderate": delta_moderate, "large": delta_large}

results = {}
for sig_name, sig in signals.items():
    for d_name, d in deltas.items():
        xr, bits, dtrace, steps = delta_modulate(sig, d, adaptive=False)
        ok, maxdev = validate_delta_steps(xr, dtrace)
        mse = np.mean((sig - xr) ** 2)
        results[(sig_name, d_name)] = dict(xr=xr, bits=bits, dtrace=dtrace,
                                            mse=mse, valid=ok, maxdev=maxdev)

# --- Visualization C: delta staircase (grid: signals x step sizes) ----
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
for row, sig_name in enumerate(["slow", "fast"]):
    for col, d_name in enumerate(["small", "moderate", "large"]):
        ax = axs[row, col]
        sig = signals[sig_name]
        r = results[(sig_name, d_name)]
        ax.plot(t * 1e3, sig, label="Input x[n]", lw=1.3, color="black")
        ax.step(t * 1e3, r["xr"], where="post", label="DM staircase", lw=1.1, color="C1")
        ax.set_title(f"{sig_name} input, {d_name} step (delta={deltas[d_name]})\nMSE={r['mse']:.3e}",
                     fontsize=9)
        if row == 1:
            ax.set_xlabel("Time (ms)")
        if col == 0:
            ax.set_ylabel("Amplitude")
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Delta modulation staircase: granular noise (small delta) vs slope overload (large delta, fast input)")
fig.tight_layout()
fig.savefig(f"{OUT}/C_delta_staircase.png")
plt.close(fig)

# =======================================================================
# RUN: Task 18 (optional) -- Adaptive delta modulation, fast/slope-overload case
# =======================================================================
xr_lin, _, dtrace_lin, _ = delta_modulate(x_fast, delta_small, adaptive=False)
xr_adapt, bits_a, dtrace_adapt, steps_a = delta_modulate(
    x_fast, delta_small, adaptive=True, grow=1.5, shrink=1/1.5,
    delta_min=delta_small, delta_max=delta_large
)
mse_lin = np.mean((x_fast - xr_lin) ** 2)
mse_adapt = np.mean((x_fast - xr_adapt) ** 2)
ok_adapt, maxdev_adapt = validate_delta_steps(xr_adapt, dtrace_adapt)

zoom = slice(0, 80)  # first 10 ms at Fs=8000 -> zoom in so individual steps are visible
fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axs[0].plot(t[zoom] * 1e3, x_fast[zoom], color="black", lw=1.5, label="Input x[n]")
axs[0].step(t[zoom] * 1e3, xr_lin[zoom], where="post", color="C3", lw=1.2,
            label=f"Fixed-step DM (delta={delta_small}), full-signal MSE={mse_lin:.3e}")
axs[0].step(t[zoom] * 1e3, xr_adapt[zoom], where="post", color="C2", lw=1.2,
            label=f"Adaptive DM, full-signal MSE={mse_adapt:.3e}")
axs[0].legend(fontsize=8); axs[0].set_ylabel("Amplitude")
axs[0].set_title("Optional task 18: adaptive vs fixed-step delta modulation (zoomed, rapidly varying input)")

axs[1].plot(t[zoom] * 1e3, dtrace_adapt[zoom], color="C4", label="Adaptive step size delta[n]", marker=".", ms=3)
axs[1].axhline(delta_small, color="C3", ls="--", lw=1, label="Fixed delta")
axs[1].set_xlabel("Time (ms)"); axs[1].set_ylabel("Step size")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/D_adaptive_dm.png")
plt.close(fig)

# =======================================================================
# RUN: Task -- MSE vs step size (required visualization)
# =======================================================================
delta_sweep = np.linspace(0.01, 0.6, 40)
mse_sweep = {"slow": [], "fast": []}
for sig_name, sig in signals.items():
    for d in delta_sweep:
        xr, *_ = delta_modulate(sig, d, adaptive=False)
        mse_sweep[sig_name].append(np.mean((sig - xr) ** 2))

fig, ax = plt.subplots(figsize=(8, 5))
for sig_name in signals:
    ax.plot(delta_sweep, mse_sweep[sig_name], marker="o", ms=3,
            label=f"{sig_name} input")
    opt_idx = int(np.argmin(mse_sweep[sig_name]))
    ax.axvline(delta_sweep[opt_idx], ls=":", lw=1,
               color="C0" if sig_name == "slow" else "C1", alpha=0.6)
ax.set_xlabel("Step size delta")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.set_title("MSE vs step size: granular-noise-dominated (left) vs slope-overload-dominated (right)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/E_mse_vs_step.png")
plt.close(fig)

# =======================================================================
# Mandatory validation summary (printed + saved)
# =======================================================================
validation_lines = []
validation_lines.append("Mandatory validation: DM output step must equal exactly +delta or -delta\n")
for (sig_name, d_name), r in results.items():
    validation_lines.append(
        f"  input={sig_name:5s} step={d_name:8s} delta={deltas[d_name]:.3f} "
        f"-> valid={r['valid']}, max deviation={r['maxdev']:.2e}"
    )
validation_lines.append(
    f"  input=fast  step=adaptive delta in[{delta_small},{delta_large}] "
    f"-> valid={ok_adapt}, max deviation={maxdev_adapt:.2e}"
)
validation_text = "\n".join(validation_lines)
print(validation_text)

with open(f"{OUT}/validation_log.txt", "w") as f:
    f.write(validation_text + "\n")

# Save numeric summary for the report
summary = {
    "mse_pcm_slow": mse_pcm,
    "mse_dpcm_slow": mse_dpcm,
    "mse_lin_fast": mse_lin,
    "mse_adapt_fast": mse_adapt,
}
import json
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print("\nSummary:")
print(json.dumps(summary, indent=2, default=float))
print("\nAll figures written to", OUT)
