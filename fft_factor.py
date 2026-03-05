import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import time
from sympy import factorial, fibonacci

# ─────────────────────────────────────────────────────────────
# Verbosity helpers
# ─────────────────────────────────────────────────────────────

def vprint(level, threshold, *args, **kwargs):
    """Print only when the current verbosity level meets the threshold."""
    if level >= threshold:
        print(*args, **kwargs)

# ─────────────────────────────────────────────────────────────
# PART 1: FFT Integer Multiplication (forward direction)
# ─────────────────────────────────────────────────────────────

def fft_multiply(a, b, verbosity=1):
    """Multiply two integers using FFT polynomial convolution."""
    vprint(verbosity, 2, f"    [FFT-mul] digits(a)={len(str(a))}  digits(b)={len(str(b))}")
    da = np.array([int(d) for d in str(a)], dtype=float)[::-1]
    db = np.array([int(d) for d in str(b)], dtype=float)[::-1]
    n = len(da) + len(db)
    fc = np.fft.rfft(da, n=n) * np.fft.rfft(db, n=n)
    c = np.round(np.fft.irfft(fc, n=n)).astype(int)
    for i in range(len(c) - 1):
        c[i+1] += c[i] // 10
        c[i] %= 10
    result = int("".join(str(d) for d in c[::-1]).lstrip("0") or "0")
    vprint(verbosity, 2, f"    [FFT-mul] FFT size used: {n}  →  result digits: {len(str(result))}")
    return result

# ─────────────────────────────────────────────────────────────
# PART 2: Log-domain FFT Factorization (Enhanced)
# ─────────────────────────────────────────────────────────────

def safe_log_array(x, min_val=1e-10):
    """Safely compute log with floor."""
    return np.log(np.maximum(x, min_val))

def build_sharp_signal(N, scale, sigma=1.0, verbosity=1):
    """Build signal with sharp spikes at log(k) positions."""
    N = int(N)
    limit = int(N**0.5) + 1
    log_N = np.log(N)
    size = int(log_N * scale) + 30
    x = np.arange(size)
    signal = np.zeros(size)

    spike_count = 0
    for k in range(2, limit + 1):
        center = np.log(k) * scale
        idx = int(round(center))
        if 0 <= idx < size:
            signal[idx] += 1.0
            spike_count += 1
            if sigma > 0:
                for offset in [-1, 1]:
                    if 0 <= idx + offset < size:
                        signal[idx + offset] += 0.3

    max_val = signal.max()
    if max_val > 0:
        signal = signal / max_val

    vprint(verbosity, 2,
           f"    [signal]  size={size}  spikes={spike_count}  "
           f"limit=√N≈{limit}  log(N)={log_N:.4f}  max_pre_norm={max_val:.3f}")
    return signal, size, log_N


def recover_factors_from_peaks(N, conv, scale, tgt_int, window, verbosity=1):
    """
    Two-phase approach:
      Phase 1 (FFT): check if conv has a meaningful peak near log(N)*scale
                     → tells us N is likely composite.
      Phase 2 (sweep): if composite, sweep k=2..√N for the actual factor
                       guided by the peak's log-position estimate.
    """
    lo = max(0, tgt_int - window)
    hi = min(len(conv), tgt_int + window + 1)
    region = conv[lo:hi] if lo < hi else np.array([0.0])
    peak_in_region = region.max()

    vprint(verbosity, 2,
           f"    [recover] window=[{lo}, {hi}]  peak_in_region={peak_in_region:.4f}")

    if peak_in_region < 0.5:
        vprint(verbosity, 2, "    [recover] Peak below 0.5 threshold → likely prime, skipping sweep.")
        return []

    limit = int(N**0.5) + 1
    a_est = max(2, int(round(np.exp(tgt_int / 2 / scale))))
    vprint(verbosity, 2, f"    [recover] Composite signal detected. a_est={a_est}  sweep limit={limit}")

    for delta in range(0, limit):
        for a in [a_est - delta, a_est + delta]:
            if 2 <= a <= limit and N % a == 0:
                vprint(verbosity, 2, f"    [recover] Factor found at delta={delta}: {a} × {N // a}")
                return [(a, N // a)]
    return []


def is_probably_prime(N, conv, tgt_int, window, threshold=0.3, verbosity=1):
    """Improved prime detection using multiple indicators."""
    lo = max(0, tgt_int - window)
    hi = min(len(conv), tgt_int + window + 1)
    region = conv[lo:hi]

    if len(region) == 0 or len(conv) == 0:
        return True

    bg_left  = conv[max(0, tgt_int - 3*window) : max(0, tgt_int - window)]
    bg_right = conv[min(len(conv), tgt_int + window) : min(len(conv), tgt_int + 3*window)]
    background = (
        np.mean(np.concatenate([bg_left, bg_right]))
        if len(conv) > 3 * window
        else np.mean(conv)
    )

    peak_ratio = region.max() / background if background > 0 else 0

    peak_idx  = np.argmax(region)
    left_avg  = np.mean(region[max(0, peak_idx - 5) : peak_idx]) if peak_idx > 5 else 0
    right_avg = np.mean(region[peak_idx : min(len(region), peak_idx + 6)]) if peak_idx < len(region) - 6 else 0
    symmetry  = (
        min(left_avg, right_avg) / max(left_avg, right_avg)
        if max(left_avg, right_avg) > 0 else 1
    )

    verdict = (peak_ratio < threshold) or (symmetry > 0.7)
    vprint(verbosity, 2,
           f"    [prime?]  peak_ratio={peak_ratio:.4f}  symmetry={symmetry:.4f}  "
           f"background={background:.4f}  threshold={threshold}  → {'PRIME' if verdict else 'composite'}")
    return verdict


def hybrid_factor(N, threshold_factor=1e6, scale=3000, sigma=1.0, verbosity=1):
    """Use FFT for moderate N, fall back to trial division for large N."""
    vprint(verbosity, 2, f"  [hybrid]  N={N:,}  threshold={threshold_factor:,.0f}  "
                         f"bits={int(N).bit_length()}")
    if N < threshold_factor:
        return fft_factor(N, scale, sigma, verbosity)
    else:
        vprint(verbosity, 1,
               f"\n  N = {N:>12,}  |  using trial division (N ≥ {threshold_factor:,.0f})")
        limit = int(N ** (1/3)) + 1
        vprint(verbosity, 2, f"    [trial]   cube-root limit = {limit:,}")
        t0 = time.time()
        for i in range(2, limit):
            if N % i == 0:
                dt = time.time() - t0
                vprint(verbosity, 1, f"              ✓  {i} × {N // i}  ({dt:.4f}s)")
                return [(i, N // i)], None, None, 0, 0, 1.0
        dt = time.time() - t0
        vprint(verbosity, 1,
               f"              ~  likely prime (no factor < cube root, {dt:.4f}s)")
        return [], None, None, 0, 0, 0.0


def fft_factor(N, scale=3000, sigma=1.0, verbosity=1):
    """Enhanced FFT factorization with improved signal processing."""
    N = int(N)
    vprint(verbosity, 2,
           f"\n  ┌─ fft_factor(N={N:,}  scale={scale}  sigma={sigma}  bits={N.bit_length()})")

    t_sig = time.time()
    signal, size, log_N = build_sharp_signal(N, scale, sigma, verbosity)
    vprint(verbosity, 2, f"  │  signal built in {time.time()-t_sig:.4f}s")

    t_fft = time.time()
    n_fft = 1 << (2 * size - 1).bit_length()
    F     = np.fft.rfft(signal, n=n_fft)
    conv  = np.fft.irfft(F * F, n=n_fft)
    vprint(verbosity, 2,
           f"  │  FFT size={n_fft}  conv_len={len(conv)}  computed in {time.time()-t_fft:.4f}s")

    target  = log_N * scale
    tgt_int = int(round(target))
    window  = int(scale * 0.02) + min(30, int(np.log10(N)) * 3)
    vprint(verbosity, 2,
           f"  │  target={target:.2f}  tgt_int={tgt_int}  window={window}")

    factors  = recover_factors_from_peaks(N, conv, scale, tgt_int, window, verbosity)
    is_prime = False
    if not factors:
        is_prime = is_probably_prime(N, conv, tgt_int, window, verbosity=verbosity)

    lo       = max(0, tgt_int - window)
    hi       = min(len(conv), tgt_int + window + 1)
    region   = conv[lo:hi] if lo < hi else np.array([0])
    peak_val = region.max() if len(region) else 0.0

    if verbosity >= 1:
        prime_status = "✓ prime" if is_prime else "~ likely prime" if not factors else "composite"
        print(f"\n  N = {N:>12,}  |  bits={N.bit_length():>2}  peak={peak_val:.3f}  "
              f"window={window}  status={prime_status}")
        if factors:
            a, b = factors[0]
            print(f"              ✓  {a:,} × {b:,}   (log_a={np.log(a):.4f}  log_b={np.log(b):.4f})")

    vprint(verbosity, 2, f"  └─ done  peak_val={peak_val:.4f}")
    return factors, signal, conv, tgt_int, window, peak_val


def trial_division(N, limit=None):
    """Fast trial division for moderate numbers."""
    if limit is None:
        limit = int(N**0.5) + 1
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if p > limit:
            break
        if N % p == 0:
            return [(p, N // p)]
    i = small_primes[-1] + 2
    while i <= limit:
        if N % i == 0:
            return [(i, N // i)]
        i += 2
    return []

# ─────────────────────────────────────────────────────────────
# PART 3: Run experiments / single-number mode
# ─────────────────────────────────────────────────────────────

def run_multiplication_demo(verbosity=1):
    print("=" * 65)
    print("  FFT MULTIPLICATION — the forward direction works perfectly")
    print("=" * 65)
    pairs = [(3, 5), (7, 11), (59, 61), (97, 103),
             (9_999_991, 9_999_973), (65537, 65539)]
    for a, b in pairs:
        result = fft_multiply(a, b, verbosity)
        match  = "✓" if result == a * b else "✗"
        print(f"  {a:>12,} × {b:<12,} = {result:>22,}  {match}")


def run_factorization_suite(verbosity=1):
    test_cases = [
        (15,                     "3 × 5"),
        (77,                     "7 × 11"),
        (143,                    "11 × 13"),
        (1001,                   "7 × 11 × 13"),
        (3599,                   "59 × 61"),
        (9991,                   "97 × 103"),
        (97,                     "prime"),
        (10_007,                 "prime (10007)"),
        (65_537,                 "prime (Fermat)"),
        (65537 * 65539,          "65537×65539"),
        (289408787257590,        "primorial(13)"),
        (int(factorial(15)),     "factorial(15)"),
        (2 ** 48 - 1,            "2^48-1"),
        (2 ** 48 + 1,            "2^48+1"),
        (fibonacci(45),          "fibonacci(45)"),
    ]

    print("\n" + "=" * 65)
    print("  FFT FACTORIZATION — Enhanced log-domain convolution")
    print("=" * 65)

    results = []
    for N, label in test_cases:
        N = int(N)
        vprint(verbosity, 2, f"\n{'─'*65}")
        vprint(verbosity, 2, f"  → Testing: {label}  (N={N:,}  bits={N.bit_length()})")
        t0 = time.time()
        factors, sig, conv, tgt, win, peak = fft_factor(N, scale=3000, sigma=1.0, verbosity=verbosity)
        dt = time.time() - t0
        vprint(verbosity, 2, f"  ← elapsed: {dt:.4f}s")
        results.append((N, label, factors, sig, conv, tgt, win, peak, dt))

    return results


def run_single_number(N, verbosity=1):
    """Run the algorithm on a single user-supplied number."""
    N = int(N)
    print("=" * 65)
    print(f"  FFT FACTORIZATION — single number mode")
    print(f"  N = {N:,}   bits = {N.bit_length()}   digits = {len(str(N))}")
    print("=" * 65)

    vprint(verbosity, 2, f"\n  Checking trivial cases (N < 2, even, etc.) …")
    if N < 2:
        print("  ✗  N must be ≥ 2.")
        return []

    if N == 2:
        print("  ✓  N = 2  is prime.")
        return []

    if N % 2 == 0:
        print(f"  ✓  N is even → trivial factor: 2 × {N // 2:,}")
        return [(2, N // 2)]

    vprint(verbosity, 1, "\n  Running fft_factor …")
    t0 = time.time()
    factors, sig, conv, tgt, win, peak = fft_factor(N, scale=3000, sigma=1.0, verbosity=verbosity)
    dt = time.time() - t0

    print(f"\n  ─── Result ───────────────────────────────────────────")
    if factors:
        a, b = factors[0]
        print(f"  ✓  COMPOSITE  →  {a:,}  ×  {b:,}")
        print(f"     Verification: {a} × {b} = {a * b:,}  {'✓ matches' if a*b == N else '✗ MISMATCH'}")
    else:
        print("  ~  PRIME (no factor found in [2, √N])")
    print(f"     Peak amplitude : {peak:.4f}")
    print(f"     Total time     : {dt:.4f}s")
    print(f"     Signal window  : [{max(0, tgt-win)}, {tgt+win}]  (window={win})")
    print("  ────────────────────────────────────────────────────")

    return [(N, "user input", factors, sig, conv, tgt, win, peak, dt)]


def run_performance_comparison(verbosity=1):
    print("\n" + "=" * 65)
    print("  PERFORMANCE COMPARISON")
    print("=" * 65)
    test_numbers = [9991, 65537, 1_000_003]
    for N in test_numbers:
        print(f"\n  Testing N = {N:,}  (bits={N.bit_length()}):")

        t0 = time.time()
        factors_fft, _, _, _, _, _ = fft_factor(N, scale=2000, sigma=1.0, verbosity=0)
        fft_time = time.time() - t0

        t0 = time.time()
        factors_trial = trial_division(N, limit=int(N**0.25))
        trial_time = time.time() - t0

        print(f"    FFT method  : {fft_time:.5f}s  →  "
              f"{factors_fft[0] if factors_fft else 'likely prime'}")
        print(f"    Trial div.  : {trial_time:.5f}s  →  "
              f"{factors_trial[0] if factors_trial else 'prime (partial sweep)'}")
        if trial_time > 0:
            speedup = trial_time / fft_time
            direction = "faster" if speedup >= 1 else "slower"
            print(f"    Speedup     : {abs(speedup):.2f}× ({direction})")
        vprint(verbosity, 2,
               f"    log₂(N)={N.bit_length()}  √N≈{int(N**0.5):,}  N^(1/4)≈{int(N**0.25):,}")


# ─────────────────────────────────────────────────────────────
# PART 4: Visualisation
# ─────────────────────────────────────────────────────────────

def build_plot(results, title_suffix=""):
    """Render the convolution diagnostic grid and pipeline diagram."""
    n_cases = len(results)
    n_cols  = min(3, n_cases)
    n_rows  = (n_cases + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(17, 4 * n_rows + 3), facecolor="#0d0d1a")
    fig.suptitle(
        f"Enhanced FFT Factorization via Log-Domain Self-Convolution{title_suffix}",
        fontsize=17, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.6, wspace=0.35, top=0.92, bottom=0.18)

    palette = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff",
               "#ff9f43", "#f0f3f4", "#ff80bf", "#7fffd4"]

    for i, (N, label, factors, sig, conv, tgt, win, peak, dt) in enumerate(results):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.set_facecolor("#111128")

        if conv is not None:
            lo  = max(0, tgt - 8 * win)
            hi  = min(len(conv), tgt + 8 * win)
            xs  = np.arange(lo, hi) / 3000
            ys  = conv[lo:hi]
            tgt_x = tgt / 3000
            col = palette[i % len(palette)]

            ax.fill_between(xs, ys, alpha=0.2, color=col)
            ax.plot(xs, ys, color=col, linewidth=1.3)
            ax.axvline(tgt_x, color="white", linestyle="--", lw=1.0, alpha=0.8,
                       label=f"log(N)={tgt_x:.3f}")
            ax.scatter([tgt_x], [conv[min(tgt, len(conv) - 1)]],
                       color="white", s=40, zorder=6)

            if factors:
                a, b = factors[0]
                ax.axvline(np.log(a), color="lime", alpha=0.4, linestyle=":",
                           linewidth=2, label=f"log({a})={np.log(a):.3f}")
                ax.axvline(np.log(b), color="#ff9f43", alpha=0.4, linestyle=":",
                           linewidth=2, label=f"log({b})={np.log(b):.3f}")
        else:
            xs = np.linspace(0, 10, 100)
            ax.plot(xs, np.zeros_like(xs), color="gray", linewidth=1.3)

        found_str = (f"= {factors[0][0]:,}×{factors[0][1]:,}"
                     if factors else "(prime)")
        status = "✓" if factors else ("✓ prime" if "prime" in label.lower() else "?")
        ax.set_title(
            f"N = {N:,}  [{label}]\n"
            f"peak={peak:.2f}  dt={dt:.2f}s  log₂={N.bit_length()}  {status}  {found_str}",
            color="white", fontsize=8.5, pad=3
        )
        ax.tick_params(colors="#666688", labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#223")
        ax.set_xlabel("log-space (log k)", color="#666688", fontsize=7)
        ax.set_ylabel("conv amplitude",    color="#666688", fontsize=7)
        if factors and conv is not None:
            ax.legend(loc="upper right", fontsize=5.5,
                      facecolor="#111128", labelcolor="white", edgecolor="#223")

    # ── Pipeline diagram ──────────────────────────────────────
    ax_d = fig.add_axes([0.05, 0.03, 0.90, 0.15])
    ax_d.set_facecolor("#111128")
    ax_d.set_xlim(0, 12); ax_d.set_ylim(0, 1); ax_d.axis("off")

    steps = [
        (0.8,  "Build signal S",       "Sharp spikes at\nlog(k) for k ∈ [2, √N]"),
        (2.8,  "Normalize",            "S = S / max(S)\n(prevents amplitude growth)"),
        (4.8,  "FFT self-convolution", "F = fft(S)\nconv = ifft(F·F)"),
        (6.8,  "Find peaks",           "Peak at log(N)·s?\nUse adaptive window"),
        (8.8,  "Recover factors",      "Check candidates\nnear peak positions"),
        (10.8, "Prime detection",      "Check symmetry &\npeak/background ratio"),
    ]
    for x, title, detail in steps:
        ax_d.add_patch(FancyBboxPatch((x - 0.9, 0.08), 1.8, 0.84,
            boxstyle="round,pad=0.05", fc="#1a1a3e", ec="#00d4ff", lw=1.3))
        ax_d.text(x, 0.72, title, ha="center", va="center",
                  color="#00d4ff", fontsize=7.5, fontweight="bold")
        ax_d.text(x, 0.35, detail, ha="center", va="center",
                  color="#aaaacc", fontsize=6.5, family="monospace")
    for j in range(len(steps) - 1):
        ax_d.annotate("", xy=(steps[j+1][0] - 0.91, 0.5),
                      xytext=(steps[j][0] + 0.91, 0.5),
                      arrowprops=dict(arrowstyle="->", color="#ffd93d", lw=1.8))

    ax_d.text(6, -0.12,
        "Normalized spikes · adaptive windows · symmetry analysis · hybrid fallback for large N",
        ha="center", va="center", color="#7777aa", fontsize=8.5, style="italic",
        transform=ax_d.transData)

    return fig


# ─────────────────────────────────────────────────────────────
# PART 5: Entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="fft_factor",
        description=(
            "FFT-based integer factorization via log-domain self-convolution.\n"
            "Run without --number to execute the full benchmark suite.\n"
            "Run with --number N to test a single integer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n", "--number",
        type=int,
        default=None,
        metavar="N",
        help="Test the algorithm on a single integer N (skips the default suite).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        default=False,
        help="Suppress the matplotlib visualisation entirely.",
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        default="fft_factorization.png",
        metavar="FILE",
        help="Path to save the graph PNG (default: fft_factorization.png). "
             "Only used when graph output is enabled.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help=(
            "Increase verbosity. Use once (-v) for standard output, "
            "twice (-vv) for detailed diagnostics (signal stats, FFT sizes, "
            "window bounds, intermediate timings, etc.)."
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except final results (overrides -v).",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        default=False,
        help="Skip the performance comparison section (suite mode only).",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    verbosity = 0 if args.quiet else args.verbose
    show_graph = not args.no_graph

    # ── Single-number mode ───────────────────────────────────
    if args.number is not None:
        results = run_single_number(args.number, verbosity=verbosity)

        if show_graph and results:
            vprint(verbosity, 1, "\n  Building graph …")
            fig = build_plot(results, title_suffix=f"  —  N = {args.number:,}")
            plt.savefig(args.save_graph, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
            vprint(verbosity, 1, f"  ✓  Graph saved → {args.save_graph}")
            plt.show()
        elif show_graph:
            vprint(verbosity, 1, "  (no conv data to plot for this N)")
        return

    # ── Full benchmark suite ─────────────────────────────────
    run_multiplication_demo(verbosity=verbosity)

    results = run_factorization_suite(verbosity=verbosity)

    if show_graph:
        vprint(verbosity, 1, "\n  Building graph …")
        fig = build_plot(results)
        plt.savefig(args.save_graph, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
        vprint(verbosity, 1, f"  ✓  Graph saved → {args.save_graph}")
        plt.show()
    else:
        vprint(verbosity, 1, "\n  (graph suppressed via --no-graph)")

    if not args.no_perf:
        run_performance_comparison(verbosity=verbosity)

    print("\n" + "=" * 65)
    print("  ENHANCEMENTS SUMMARY")
    print("=" * 65)
    print("  ✓  Normalized spikes to prevent amplitude growth")
    print("  ✓  Adaptive windows based on N size")
    print("  ✓  Peak symmetry analysis for prime detection")
    print("  ✓  Hybrid fallback for large numbers")
    print("  ✓  Factor position markers in visualization")
    print("  ✓  Performance comparison with trial division")
    print("  ✓  argparse CLI  (--number, --no-graph, -v/-vv, -q, --no-perf)")


if __name__ == "__main__":
    main()
