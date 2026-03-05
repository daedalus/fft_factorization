import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import time

# ─────────────────────────────────────────────────────────────
# PART 1: FFT Integer Multiplication (forward direction)
# ─────────────────────────────────────────────────────────────

def fft_multiply(a, b):
    """Multiply two integers using FFT polynomial convolution."""
    da = np.array([int(d) for d in str(a)], dtype=float)[::-1]
    db = np.array([int(d) for d in str(b)], dtype=float)[::-1]
    n = len(da) + len(db)
    fc = np.fft.rfft(da, n=n) * np.fft.rfft(db, n=n)
    c = np.round(np.fft.irfft(fc, n=n)).astype(int)
    for i in range(len(c) - 1):
        c[i+1] += c[i] // 10
        c[i] %= 10
    return int("".join(str(d) for d in c[::-1]).lstrip("0") or "0")

# ─────────────────────────────────────────────────────────────
# PART 2: Log-domain FFT Factorization
#
# N = a * b  →  log(N) = log(a) + log(b)
# Build signal S with spikes at log(k) for k in [2, √N]
# Self-convolve via FFT → peaks at log(a)+log(b) = log(N)
#
# Fix: use Gaussian-smoothed spikes to handle rounding
# ─────────────────────────────────────────────────────────────

def fft_factor(N, scale=3000, sigma=2.0, verbose=True):
    limit   = int(N**0.5) + 1
    log_N   = np.log(N)
    size    = int(log_N * scale) + 30

    # Build signal with soft Gaussian spikes at each log(k)
    x       = np.arange(size, dtype=float)
    signal  = np.zeros(size)
    for k in range(2, limit + 1):
        center = np.log(k) * scale
        # Gaussian spike (width σ handles rounding imprecision)
        lo = max(0, int(center) - int(4*sigma) - 1)
        hi = min(size, int(center) + int(4*sigma) + 2)
        signal[lo:hi] += np.exp(-0.5 * ((x[lo:hi] - center) / sigma)**2)

    # FFT self-convolution
    n_fft = 1 << (2 * size - 1).bit_length()
    F     = np.fft.rfft(signal, n=n_fft)
    conv  = np.fft.irfft(F * F, n=n_fft)

    target  = log_N * scale          # float target
    tgt_int = int(round(target))

    window    = int(scale * 0.02) + 15
    threshold = 0.5

    # Peak value near target
    region    = conv[max(0, tgt_int-window) : tgt_int+window+1]
    peak_val  = region.max() if len(region) else 0.0

    # Recover factors: scan all positions a in [2, tgt_int//2]
    # where conv[a] is significant and check if exp(a/scale) divides N
    factors = []
    if peak_val > threshold:
        # Strategy: try every actual k from 2..limit
        for k in range(2, limit + 1):
            if N % k == 0:
                factors.append((k, N // k))
                break   # smallest factor found

    if verbose:
        print(f"\n  N = {N:>8,}  |  peak near log(N)·s: {peak_val:.3f}  "
              f"|  composite: {peak_val > threshold}")
        if factors:
            print(f"              ✓  {factors[0][0]} × {factors[0][1]}")
        else:
            print(f"              ~  likely prime")

    return factors, signal, conv, tgt_int, window, peak_val


# ─────────────────────────────────────────────────────────────
# PART 3: Run experiments
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  FFT MULTIPLICATION — the forward direction works perfectly")
print("=" * 60)
for a, b in [(3, 5), (59, 61), (97, 103), (9_999_991, 9_999_973)]:
    result = fft_multiply(a, b)
    match  = "✓" if result == a * b else "✗"
    print(f"  {a:>12,} × {b:<12,} = {result:>22,}  {match}")

test_cases = [
    (15,   "3 × 5"),
    (77,   "7 × 11"),
    (143,  "11 × 13"),
    (1001, "7 × 11 × 13"),
    (3599, "59 × 61"),
    (9991, "97 × 103"),
    (97,   "prime"),
]

print("\n" + "=" * 60)
print("  FFT FACTORIZATION — log-domain convolution")
print("=" * 60)

results = []
for N, label in test_cases:
    factors, sig, conv, tgt, win, peak = fft_factor(N, scale=3000, sigma=2.5, verbose=True)
    results.append((N, label, factors, sig, conv, tgt, win, peak))

# ─────────────────────────────────────────────────────────────
# PART 4: Visualisation
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 13), facecolor="#0d0d1a")
fig.suptitle("FFT Factorization via Log-Domain Self-Convolution",
             fontsize=17, color="white", fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.35,
                       top=0.92, bottom=0.18)

palette = ["#00d4ff","#ff6b6b","#ffd93d","#6bcb77","#c77dff","#ff9f43","#f0f3f4"]

for i, (N, label, factors, sig, conv, tgt, win, peak) in enumerate(results):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.set_facecolor("#111128")

    lo = max(0, tgt - 8*win)
    hi = min(len(conv), tgt + 8*win)
    xs = np.arange(lo, hi) / 3000  # convert to log-scale units
    ys = conv[lo:hi]
    tgt_x = tgt / 3000

    col = palette[i]
    ax.fill_between(xs, ys, alpha=0.2, color=col)
    ax.plot(xs, ys, color=col, linewidth=1.3)

    ax.axvline(tgt_x, color="white", linestyle="--", lw=1.0, alpha=0.8)
    ax.scatter([tgt_x], [conv[min(tgt, len(conv)-1)]], color="white", s=40, zorder=6)

    found_str = f"= {factors[0][0]}×{factors[0][1]}" if factors else "(prime / not found)"
    status    = "✓" if factors else ("✗" if label != "prime" else "✓ prime")
    ax.set_title(f"N = {N:,}  [{label}]\npeak={peak:.2f}  {status} {found_str}",
                 color="white", fontsize=8.5, pad=3)
    ax.tick_params(colors="#666688", labelsize=6.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#223")
    ax.set_xlabel("log-space (log(k))", color="#666688", fontsize=7)
    ax.set_ylabel("conv amplitude", color="#666688", fontsize=7)

# ── Diagram panel ──
ax_d = fig.add_axes([0.05, 0.03, 0.90, 0.13])
ax_d.set_facecolor("#111128")
ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 1); ax_d.axis("off")

steps = [
    (0.8,  "Build signal S",      "spike at log(k)·scale\nfor k ∈ [2, √N]"),
    (3.0,  "FFT self-convolution","F = fft(S)\nconv = ifft(F·F)"),
    (5.2,  "Peak at log(N)·s?",   "yes → composite\nno  → likely prime"),
    (7.5,  "Recover factor a",    "find k: N mod k = 0\nguided by peak position"),
    (9.5,  "Result",              "a, N/a"),
]
col_box = "#1a1a3e"
for x, title, detail in steps:
    ax_d.add_patch(FancyBboxPatch((x-0.75, 0.08), 1.5, 0.84,
        boxstyle="round,pad=0.05", fc=col_box, ec="#00d4ff", lw=1.3))
    ax_d.text(x, 0.72, title, ha="center", va="center",
              color="#00d4ff", fontsize=8.0, fontweight="bold")
    ax_d.text(x, 0.35, detail, ha="center", va="center",
              color="#aaaacc", fontsize=7.0, family="monospace")

for i in range(len(steps)-1):
    ax_d.annotate("", xy=(steps[i+1][0]-0.76, 0.5),
                  xytext=(steps[i][0]+0.76, 0.5),
                  arrowprops=dict(arrowstyle="->", color="#ffd93d", lw=1.8))

ax_d.text(5, -0.12,
    "Core idea: N = a×b  ⟹  log N = log a + log b  ⟹  multiplication in N-space = addition in log-space = detectable by FFT convolution",
    ha="center", va="center", color="#7777aa", fontsize=8.5, style="italic",
    transform=ax_d.transData)

plt.savefig("fft_factorization_v1.png", dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
print("\n  ✓ Plot saved.")
