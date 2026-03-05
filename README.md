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

## Why this doesn't compete with Pollard ρ / ECM / NFS

The signal construction loop iterates over every integer in `[2, √N]`, which
is the *same* work as naive trial division — the FFT convolution is fast, but
it cannot avoid paying the O(√N) cost upfront just to build the input. State-
of-the-art algorithms sidestep this entirely:

| Algorithm | Complexity | Avoids O(√N) how? |
|-----------|------------|-------------------|
| Pollard ρ | O(N^¼) expected | Random walk in ℤ/Nℤ — finds a factor after O(√p) steps where p is the *smallest* factor, not √N |
| ECM | O(exp(√(log p · log log p))) | Works in the group of an elliptic curve mod N; cost depends on the *size of the factor p*, not N |
| GNFS | O(exp((log N)^⅓ · (log log N)^⅔)) | Algebraic sieve — sub-exponential in the *full* bit-size of N |

In short: Pollard ρ and ECM exploit algebraic structure to find small factors
cheaply without scanning the full range; NFS exploits smooth-number density to
factor arbitrary N in sub-exponential time. The FFT approach here has no
equivalent shortcut — it must enumerate all candidates up to √N to build the
signal, making it fundamentally O(√N) and outclassed by Pollard ρ even for
20-bit inputs.

## Requirements

```
pip install numpy matplotlib scipy
```

## Usage

```bash
python fft_factor.py
```
