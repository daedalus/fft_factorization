"""
gen_sweetspot.py
────────────────
Generates random near-square semiprimes N = p × q  (p ≈ q ≈ √N)
in increasing order of bit-width, then runs fft_factor on each one
to validate the algorithm's sweet-spot performance.

"Near-square" means both p and q are random primes of the same half-bit
size, so |log(p) − log(q)| is small and the FFT convolution peak lands
close to log(N)/2 — the ideal regime for the algorithm.

Usage
─────
  python gen_sweetspot.py                   # default bit schedule, no graph
  python gen_sweetspot.py --max-bits 40     # stop at 40-bit N
  python gen_sweetspot.py --graph           # show convolution plots
  python gen_sweetspot.py --pairs 5         # 5 pairs per bit-width
  python gen_sweetspot.py -vv               # full diagnostics
  python gen_sweetspot.py --csv out.csv          # save results table
  python gen_sweetspot.py --scaling-graph        # bits vs runtime+peak chart
  python gen_sweetspot.py --scaling-graph --save-scaling scaling.png
"""

import argparse
import csv
import math
import random
import sys
import time

import numpy as np

# ── optional matplotlib (only needed for --graph) ──────────────────────────
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── import fft_factor from the companion module ─────────────────────────────
try:
    from fft_factor import fft_factor, vprint
except ImportError:
    print("[ERROR] Cannot import fft_factor.py — make sure it is in the same directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Prime generation
# ─────────────────────────────────────────────────────────────────────────────

from sympy import isprime
def random_prime_of_bits(bits: int, rng: random.Random) -> int:
    """Return a random prime with exactly `bits` significant bits."""
    if bits < 2:
        raise ValueError("bits must be ≥ 2")
    lo = 1 << (bits - 1)          # 2^(bits-1)  — ensures MSB is set
    hi = (1 << bits) - 1          # 2^bits - 1
    while True:
        candidate = rng.randrange(lo | 1, hi + 1, 2)  # odd numbers only
        if isprime(candidate):
            return candidate


def gen_near_square_pair(n_bits: int, rng: random.Random,
                         max_imbalance: float = 0.10,
                         verbosity: int = 1) -> tuple[int, int, int]:
    """
    Generate p, q such that:
      - p and q each have ⌊n_bits/2⌋ or ⌈n_bits/2⌉ bits
      - |log2(p) - log2(q)| / (n_bits/2) ≤ max_imbalance
      - N = p*q has exactly n_bits bits (or n_bits ± 1)

    Returns (p, q, N) with p ≤ q.
    """
    half = n_bits // 2
    remainder = n_bits - half          # half or half+1 for odd n_bits

    attempts = 0
    while True:
        attempts += 1
        p = random_prime_of_bits(half,      rng)
        q = random_prime_of_bits(remainder, rng)
        if p == q:
            continue

        # Enforce near-square: relative log imbalance
        log_p, log_q = math.log2(p), math.log2(q)
        imbalance = abs(log_p - log_q) / (n_bits / 2)
        if imbalance > max_imbalance:
            vprint(verbosity, 2,
                   f"    [gen]  skip  imbalance={imbalance:.4f} > {max_imbalance}  "
                   f"(p_bits={log_p:.2f}  q_bits={log_q:.2f})")
            continue

        N = p * q
        vprint(verbosity, 2,
               f"    [gen]  found after {attempts} attempt(s)  "
               f"p={p}  q={q}  N_bits={N.bit_length()}  imbalance={imbalance:.4f}")
        return (min(p, q), max(p, q), N)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

# Column widths for the live table
_HDR = (
    f"{'bits':>5}  {'p':>22}  {'q':>22}  {'N':>46}  "
    f"{'found':>6}  {'peak':>6}  {'time':>8}  {'imbalance':>9}  verified"
)
_SEP = "─" * len(_HDR)


def run_sweep(bit_schedule: list[int],
              pairs_per_width: int,
              max_imbalance: float,
              verbosity: int,
              rng: random.Random,
              csv_path: str | None,
              show_graph: bool,
              show_scaling: bool,
              scaling_path: str) -> list[dict]:
    """
    Core sweep: for each bit-width in bit_schedule, generate `pairs_per_width`
    near-square semiprimes and run fft_factor on each.
    """
    print("=" * len(_HDR))
    print("  FFT SWEET-SPOT SWEEP — near-square semiprimes  (p ≈ q ≈ √N)")
    print("=" * len(_HDR))
    print(_HDR)
    print(_SEP)

    all_results: list[dict] = []
    # rows kept for graph (only those with conv data)
    graph_rows: list[tuple] = []

    for n_bits in bit_schedule:
        for pair_idx in range(pairs_per_width):
            vprint(verbosity, 2,
                   f"\n{'━'*70}\n  Generating {n_bits}-bit semiprime  "
                   f"(pair {pair_idx+1}/{pairs_per_width})")

            # ── generate pair ────────────────────────────────────────────────
            t_gen = time.time()
            p, q, N = gen_near_square_pair(n_bits, rng, max_imbalance, verbosity)
            gen_time = time.time() - t_gen

            log_p, log_q = math.log2(p), math.log2(q)
            imbalance = abs(log_p - log_q) / (n_bits / 2)

            # ── factorize ────────────────────────────────────────────────────
            t_fft = time.time()
            factors, sig, conv, tgt, win, peak = fft_factor(
                N, scale=3000, sigma=1.0, verbosity=0   # inner verbosity off
            )
            fft_time = time.time() - t_fft

            found    = bool(factors)
            correct  = (found and set(factors[0]) == {p, q})
            verified = "✓" if correct else ("✗ wrong" if found else "✗ miss")

            # ── verbose diagnostic line ──────────────────────────────────────
            if verbosity >= 2:
                print(f"\n  N = {N}  ({n_bits} bits)")
                print(f"    p = {p}  ({p.bit_length()} bits)")
                print(f"    q = {q}  ({q.bit_length()} bits)")
                print(f"    imbalance = {imbalance:.4f}   gen_time = {gen_time:.4f}s")
                print(f"    FFT peak  = {peak:.4f}         fft_time = {fft_time:.4f}s")
                if found:
                    fa, fb = factors[0]
                    print(f"    recovered: {fa} × {fb}  {'✓' if correct else '✗'}")

            # ── table row ────────────────────────────────────────────────────
            row = (
                f"{n_bits:>5}  {p:>22}  {q:>22}  {N:>46}  "
                f"{'yes' if found else 'no':>6}  {peak:>6.3f}  "
                f"{fft_time:>7.3f}s  {imbalance:>9.4f}  {verified}"
            )
            print(row)

            rec = dict(
                n_bits=n_bits, p=p, q=q, N=N,
                found=found, correct=correct, peak=peak,
                fft_time=fft_time, gen_time=gen_time, imbalance=imbalance,
            )
            all_results.append(rec)

            if conv is not None:
                graph_rows.append((N, f"{n_bits}b  {p}×{q}", factors,
                                   sig, conv, tgt, win, peak, fft_time))

        # bit-width separator
        print(_SEP)

    # ── summary ──────────────────────────────────────────────────────────────
    total   = len(all_results)
    n_found = sum(r["found"] for r in all_results)
    n_ok    = sum(r["correct"] for r in all_results)

    print(f"\n  Pairs tested : {total}")
    print(f"  Factor found : {n_found}/{total}  ({100*n_found/total:.1f}%)")
    print(f"  Correct      : {n_ok}/{total}  ({100*n_ok/total:.1f}%)")

    avg_by_bits: dict[int, list[float]] = {}
    for r in all_results:
        avg_by_bits.setdefault(r["n_bits"], []).append(r["fft_time"])
    print("\n  Average FFT time by bit-width:")
    for b, times in sorted(avg_by_bits.items()):
        print(f"    {b:>3} bits : {sum(times)/len(times):.4f}s  "
              f"({len(times)} sample{'s' if len(times)>1 else ''})")

    # ── CSV export ───────────────────────────────────────────────────────────
    if csv_path:
        fields = ["n_bits","p","q","N","found","correct","peak",
                  "fft_time","gen_time","imbalance"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  ✓ Results saved → {csv_path}")

    # ── conv grid graph ──────────────────────────────────────────────────────
    if show_graph and graph_rows:
        if not HAS_MPL:
            print("\n  [warn] matplotlib not available — skipping conv graph.")
        else:
            _plot_sweep(graph_rows)

    # ── scaling graph ────────────────────────────────────────────────────────
    if show_scaling:
        if not HAS_MPL:
            print("\n  [warn] matplotlib not available — skipping scaling graph.")
        else:
            _plot_scaling(all_results, scaling_path)

    return all_results


def _plot_sweep(rows: list[tuple]):
    """Render one convolution panel per semiprime."""
    n_cases = len(rows)
    n_cols  = min(4, n_cases)
    n_rows  = (n_cases + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 3.8 * n_rows + 2.5), facecolor="#0d0d1a")
    fig.suptitle("FFT Sweet-Spot Sweep — Near-Square Semiprimes  (p ≈ q ≈ √N)",
                 fontsize=15, color="white", fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.65, wspace=0.35, top=0.93, bottom=0.17)

    palette = ["#00d4ff","#ff6b6b","#ffd93d","#6bcb77","#c77dff",
               "#ff9f43","#f0f3f4","#ff80bf","#7fffd4","#a0c4ff"]

    for i, (N, label, factors, sig, conv, tgt, win, peak, dt) in enumerate(rows):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.set_facecolor("#111128")

        lo  = max(0, tgt - 10 * win)
        hi  = min(len(conv), tgt + 10 * win)
        xs  = np.arange(lo, hi) / 3000
        ys  = conv[lo:hi]
        col = palette[i % len(palette)]

        ax.fill_between(xs, ys, alpha=0.18, color=col)
        ax.plot(xs, ys, color=col, linewidth=1.2)

        tgt_x = tgt / 3000
        ax.axvline(tgt_x, color="white", linestyle="--", lw=0.9, alpha=0.8,
                   label=f"log(N)={tgt_x:.3f}")
        ax.scatter([tgt_x], [conv[min(tgt, len(conv)-1)]],
                   color="white", s=30, zorder=6)

        if factors:
            a, b = factors[0]
            ax.axvline(math.log(a), color="lime",   alpha=0.45, linestyle=":",
                       linewidth=1.8, label=f"log p={math.log(a):.3f}")
            ax.axvline(math.log(b), color="#ff9f43", alpha=0.45, linestyle=":",
                       linewidth=1.8, label=f"log q={math.log(b):.3f}")

        found_str = (f"✓ {factors[0][0]}×{factors[0][1]}"
                     if factors else "✗ missed")
        ax.set_title(
            f"{label}\npeak={peak:.3f}  dt={dt:.3f}s  {found_str}",
            color="white", fontsize=7.5, pad=3
        )
        ax.tick_params(colors="#666688", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#223")
        ax.set_xlabel("log k", color="#666688", fontsize=6.5)
        ax.set_ylabel("conv",  color="#666688", fontsize=6.5)
        ax.legend(loc="upper left", fontsize=5,
                  facecolor="#111128", labelcolor="white", edgecolor="#223")

    # ── annotation strip ─────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.05, 0.02, 0.90, 0.12])
    ax_d.set_facecolor("#111128"); ax_d.axis("off")
    ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 1)

    notes = [
        (1.2,  "Sweet spot",       "p ≈ q ≈ √N\n|log p − log q| small"),
        (3.6,  "Why it helps",     "Peak lands near\nlog(N)/2 — easy recovery"),
        (6.0,  "Imbalance metric", "|log₂p − log₂q|\n÷ (n_bits/2)  ∈ [0, 0.1]"),
        (8.4,  "Still O(√N)",      "FFT narrows sweep;\nno asymptotic gain"),
    ]
    for x, title, detail in notes:
        ax_d.add_patch(FancyBboxPatch((x-1.1, 0.06), 2.2, 0.88,
            boxstyle="round,pad=0.05", fc="#1a1a3e", ec="#00d4ff", lw=1.2))
        ax_d.text(x, 0.74, title,  ha="center", va="center",
                  color="#00d4ff", fontsize=7.5, fontweight="bold")
        ax_d.text(x, 0.34, detail, ha="center", va="center",
                  color="#aaaacc", fontsize=6.5, family="monospace")

    plt.savefig("sweetspot_sweep.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d1a")
    print("  ✓  Graph saved → sweetspot_sweep.png")
    plt.show()


def _plot_scaling(results: list[dict], save_path: str):
    """
    Dual-axis scaling chart:
      Left  y-axis (cyan)   — FFT runtime per sample, with per-bit-width mean + scatter
      Right y-axis (orange) — FFT convolution peak amplitude, same x-axis
    """
    # ── aggregate per bit-width ──────────────────────────────────────────────
    from collections import defaultdict
    by_bits: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_bits[r["n_bits"]].append(r)

    bits_sorted = sorted(by_bits)
    mean_time   = [sum(r["fft_time"] for r in by_bits[b]) / len(by_bits[b]) for b in bits_sorted]
    mean_peak   = [sum(r["peak"]     for r in by_bits[b]) / len(by_bits[b]) for b in bits_sorted]

    # scatter jitter for individual points
    all_bits_scatter  = [r["n_bits"]   for r in results]
    all_time_scatter  = [r["fft_time"] for r in results]
    all_peak_scatter  = [r["peak"]     for r in results]
    found_mask        = [r["found"]    for r in results]

    # ── figure ───────────────────────────────────────────────────────────────
    COL_BG    = "#0d0d1a"
    COL_PANEL = "#111128"
    COL_TIME  = "#00d4ff"   # cyan  — runtime
    COL_PEAK  = "#ff9f43"   # orange — peak
    COL_FOUND = "#6bcb77"   # green dot  — factor found
    COL_MISS  = "#ff6b6b"   # red dot    — missed

    fig, ax1 = plt.subplots(figsize=(13, 6), facecolor=COL_BG)
    ax1.set_facecolor(COL_PANEL)
    fig.suptitle(
        "FFT Sweet-Spot Scaling  —  bits(N) vs  Runtime  &  Convolution Peak",
        fontsize=14, color="white", fontweight="bold", y=0.97
    )

    ax2 = ax1.twinx()

    # ── runtime (left axis) ──────────────────────────────────────────────────
    # individual scatter coloured by found/missed
    for x, y, ok in zip(all_bits_scatter, all_time_scatter, found_mask):
        ax1.scatter(x, y, color=COL_FOUND if ok else COL_MISS,
                    s=28, alpha=0.65, zorder=4, linewidths=0)

    ax1.plot(bits_sorted, mean_time, color=COL_TIME,
             linewidth=2.2, marker="o", markersize=5, zorder=5,
             label="mean runtime (s)")
    ax1.fill_between(bits_sorted, mean_time, alpha=0.10, color=COL_TIME)

    ax1.set_xlabel("N  bit-width", color="#aaaacc", fontsize=11)
    ax1.set_ylabel("FFT runtime  (seconds)", color=COL_TIME, fontsize=10)
    ax1.tick_params(axis="y", colors=COL_TIME, labelsize=8)
    ax1.tick_params(axis="x", colors="#aaaacc", labelsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#223344")
    ax1.yaxis.label.set_color(COL_TIME)

    # ── peak (right axis) ────────────────────────────────────────────────────
    ax2.scatter(all_bits_scatter, all_peak_scatter,
                color=COL_PEAK, s=22, alpha=0.50, zorder=3,
                marker="D", linewidths=0)
    ax2.plot(bits_sorted, mean_peak, color=COL_PEAK,
             linewidth=2.0, marker="D", markersize=5, zorder=5,
             linestyle="--", label="mean peak amplitude")
    ax2.fill_between(bits_sorted, mean_peak, alpha=0.08, color=COL_PEAK)

    ax2.set_ylabel("conv peak amplitude", color=COL_PEAK, fontsize=10)
    ax2.tick_params(axis="y", colors=COL_PEAK, labelsize=8)
    ax2.spines["right"].set_edgecolor(COL_PEAK)
    ax2.spines["left"].set_edgecolor(COL_TIME)
    ax2.yaxis.label.set_color(COL_PEAK)

    # ── legend ───────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=COL_TIME,  linewidth=2, label="mean runtime (s)"),
        Line2D([0], [0], color=COL_PEAK,  linewidth=2, linestyle="--",
               label="mean peak amplitude"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_FOUND,
               markersize=7, label="factor found", linewidth=0),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MISS,
               markersize=7, label="factor missed", linewidth=0),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
               facecolor=COL_PANEL, labelcolor="white", edgecolor="#334")

    # ── reference line: O(√N) ∝ 2^(bits/2) ─────────────────────────────────
    if len(bits_sorted) >= 3:
        ref_x  = np.array(bits_sorted, dtype=float)
        # scale to match mean_time at the midpoint
        mid_i  = len(bits_sorted) // 2
        ref_y  = np.array([2 ** (b / 2) for b in bits_sorted], dtype=float)
        scale  = mean_time[mid_i] / ref_y[mid_i] if ref_y[mid_i] > 0 else 1.0
        ref_y *= scale
        ax1.plot(ref_x, ref_y, color="#ffffff", linewidth=1.0,
                 linestyle=":", alpha=0.35, label="O(√N) reference")
        ax1.text(ref_x[-1], ref_y[-1] * 1.05, "O(√N)",
                 color="#888899", fontsize=7.5, ha="right")

    # ── grid ─────────────────────────────────────────────────────────────────
    ax1.grid(True, color="#1e1e3a", linestyle="--", linewidth=0.6, alpha=0.7)
    ax1.set_xticks(bits_sorted)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COL_BG)
    print(f"  ✓  Scaling graph saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="gen_sweetspot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--min-bits", type=int, default=8, metavar="B",
        help="Smallest N bit-width to test (default: 8).",
    )
    p.add_argument(
        "--max-bits", type=int, default=56, metavar="B",
        help="Largest N bit-width to test (default: 56).",
    )
    p.add_argument(
        "--step", type=int, default=4, metavar="S",
        help="Bit-width increment between levels (default: 4).",
    )
    p.add_argument(
        "--pairs", type=int, default=2, metavar="K",
        help="Number of random pairs to generate per bit-width (default: 2).",
    )
    p.add_argument(
        "--max-imbalance", type=float, default=0.10, metavar="F",
        help=(
            "Max allowed |log₂p − log₂q| / (n_bits/2). "
            "Lower → tighter near-square constraint (default: 0.10)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--graph", action="store_true", default=False,
        help="Show convolution plots for each semiprime (requires matplotlib).",
    )
    p.add_argument(
        "--scaling-graph", action="store_true", default=False,
        help=(
            "Show a dual-axis scaling chart: bits(N) on x-axis, "
            "FFT runtime (cyan, left) and conv peak (orange, right)."
        ),
    )
    p.add_argument(
        "--save-scaling", type=str, default="sweetspot_scaling.png", metavar="FILE",
        help="Path to save the scaling graph PNG (default: sweetspot_scaling.png).",
    )
    p.add_argument(
        "--csv", type=str, default=None, metavar="FILE",
        help="Save results to a CSV file.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=1,
        help="Increase verbosity (-v standard, -vv full diagnostics).",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", default=False,
        help="Suppress all output except the results table.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    verbosity = 0 if args.quiet else args.verbose

    rng = random.Random(args.seed)
    if args.seed is not None:
        vprint(verbosity, 1, f"  seed = {args.seed}")

    # Build bit schedule: even values only (so half-bits are integers)
    schedule = list(range(
        args.min_bits + (args.min_bits % 2),   # round up to even
        args.max_bits + 1,
        args.step if args.step % 2 == 0 else args.step + 1,
    ))

    if not schedule:
        print("[ERROR] Empty bit schedule — check --min-bits / --max-bits / --step")
        sys.exit(1)

    vprint(verbosity, 1,
           f"\n  Bit schedule : {schedule}"
           f"\n  Pairs/width  : {args.pairs}"
           f"\n  Max imbalance: {args.max_imbalance}"
           f"\n  Seed         : {args.seed}\n")

    run_sweep(
        bit_schedule    = schedule,
        pairs_per_width = args.pairs,
        max_imbalance   = args.max_imbalance,
        verbosity       = verbosity,
        rng             = rng,
        csv_path        = args.csv,
        show_graph      = args.graph,
        show_scaling    = args.scaling_graph,
        scaling_path    = args.save_scaling,
    )


if __name__ == "__main__":
    main()
