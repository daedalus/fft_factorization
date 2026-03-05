# FFT Integer Factorization

Exploration of using FFT log-domain self-convolution to factor integers.

## Core idea

Since `N = a × b` implies `log(N) = log(a) + log(b)`, multiplication becomes
addition in log-space. Building a signal with spikes at `log(k)` for all
`k ∈ [2, √N]` and self-convolving it via FFT produces a peak at `log(N)*scale`
whenever N is composite — revealing the existence (and approximate location)
of a factor pair.

## Version history

| Version | Commit | Key changes |
|---------|--------|-------------|
| v1 | `90a9ace` | Gaussian spikes, monolithic `fft_factor()`, 7 test cases |
| v2 | `dc89691` | Decomposed architecture, `scipy.signal.find_peaks`, multi-criteria prime detection, hybrid fallback, 9 test cases |
| v3 | `9948f3b` | Fix normalization bug (`/max` not `/count`), fix factor recovery (two-phase: FFT detects, guided sweep identifies) |

## Time complexity

| Step | Cost |
|------|------|
| Signal construction | O(√N) — the bottleneck |
| FFT convolution | O(log N · log log N) |
| Factor recovery (best) | O(\|a − √N\|) — O(1) for twin primes |
| Factor recovery (worst) | O(√N) |
| **Total** | **O(√N)** |

The algorithm is asymptotically equivalent to trial division. The FFT's
practical contribution is narrowing the recovery sweep to start at `√N`
rather than 2 — a real speedup for near-balanced semiprimes (the algorithm's
sweet spot), but not a complexity improvement.

## Sweet spot

Near-square composites: `N = p × q` where `p ≈ q ≈ √N`.
Products of twin/cousin/sexy primes (e.g. 59×61, 97×103) are ideal targets.
This is also the structure of RSA keys — though RSA key sizes (2048+ bits)
are far beyond what any classical O(√N) method can reach.

## Requirements

```
pip install numpy matplotlib scipy
```

## Usage

```bash
python fft_factor.py
```
