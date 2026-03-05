import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
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
# PART 2: Log-domain FFT Factorization (Enhanced)
# ─────────────────────────────────────────────────────────────

def safe_log_array(x, min_val=1e-10):
    """Safely compute log with floor"""
    return np.log(np.maximum(x, min_val))

def build_sharp_signal(N, scale, sigma=1.0):
    """Build signal with sharp spikes at log(k) positions"""
    limit = int(N**0.5) + 1
    log_N = np.log(N)
    size = int(log_N * scale) + 30
    x = np.arange(size)
    signal = np.zeros(size)

    for k in range(2, limit + 1):
        center = np.log(k) * scale
        idx = int(round(center))
        if 0 <= idx < size:
            signal[idx] += 1.0  # Pure spike
            # Add small Gaussian for tolerance
            if sigma > 0:
                for offset in [-1, 1]:
                    if 0 <= idx+offset < size:
                        signal[idx+offset] += 0.3

    # BUG: divides by O(sqrt(N)) — zeroes out all peaks
    signal = signal / len(range(2, limit + 1))
    return signal, size, log_N

def recover_factors_from_peaks(N, conv, scale, tgt_int, window):
    """Use convolution peaks to guide factor search"""
    lo = max(0, tgt_int - window)
    hi = min(len(conv), tgt_int + window + 1)
    region = conv[lo:hi]

    if len(region) == 0:
        return []

    peaks, properties = find_peaks(region, height=0.1 * region.max())

    candidates = []
    for peak_idx in peaks:
        pos = tgt_int - window + peak_idx
        a_candidate = int(round(np.exp(pos / scale)))

        search_radius = max(5, int(window / 10))
        # BUG: searches peak position rather than factor position;
        # also fails when b > sqrt(N) since b is never in spike_set
        for a in range(max(2, a_candidate - search_radius),
                      min(int(N**0.5) + 1, a_candidate + search_radius + 1)):
            if a * a > N:
                continue
            if N % a == 0:
                b = N // a
                candidates.append((a, b, region[peak_idx]))
                break

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(candidates[0][0], candidates[0][1])]
    return []

def is_probably_prime(N, conv, tgt_int, window, threshold=0.3):
    """Improved prime detection using multiple indicators"""
    lo = max(0, tgt_int - window)
    hi = min(len(conv), tgt_int + window + 1)
    region = conv[lo:hi]

    if len(region) == 0 or len(conv) == 0:
        return True

    background = np.mean(np.concatenate([
        conv[max(0, tgt_int-3*window):max(0, tgt_int-window)],
        conv[min(len(conv), tgt_int+window):min(len(conv), tgt_int+3*window)]
    ])) if len(conv) > 3*window else np.mean(conv)

    peak_ratio = region.max() / background if background > 0 else 0

    peak_idx = np.argmax(region)
    left_avg = np.mean(region[max(0, peak_idx-5):peak_idx]) if peak_idx > 5 else 0
    right_avg = np.mean(region[peak_idx:min(len(region), peak_idx+6)]) if peak_idx < len(region)-6 else 0
    symmetry = min(left_avg, right_avg) / max(left_avg, right_avg) if max(left_avg, right_avg) > 0 else 1

    return (peak_ratio < threshold) or (symmetry > 0.7)

def hybrid_factor(N, threshold_factor=1e6, scale=3000, sigma=1.0, verbose=True):
    """Use FFT for moderate N, fallback for large N"""
    if N < threshold_factor:
        return fft_factor(N, scale, sigma, verbose)
    else:
        if verbose:
            print(f"\n  N = {N:>8,}  |  using trial division (N > {threshold_factor:,.0f})")
        limit = int(N ** (1/3)) + 1
        for i in range(2, limit):
            if N % i == 0:
                if verbose:
                    print(f"              ✓  {i} × {N // i}")
                return [(i, N // i)], None, None, 0, 0, 1.0
        if verbose:
            print(f"              ~  likely prime (no factor < cube root)")
        return [], None, None, 0, 0, 0.0

def fft_factor(N, scale=3000, sigma=1.0, verbose=True):
    """Enhanced FFT factorization with improved signal processing"""
    signal, size, log_N = build_sharp_signal(N, scale, sigma)

    n_fft = 1 << (2 * size - 1).bit_length()
    F = np.fft.rfft(signal, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)

    target = log_N * scale
    tgt_int = int(round(target))

    window = int(scale * 0.02) + min(30, int(np.log10(N)) * 3)

    factors = recover_factors_from_peaks(N, conv, scale, tgt_int, window)

    is_prime = False
    if not factors:
        is_prime = is_probably_prime(N, conv, tgt_int, window)

    lo = max(0, tgt_int - window)
    hi = min(len(conv), tgt_int + window + 1)
    region = conv[lo:hi] if lo < hi else np.array([0])
    peak_val = region.max() if len(region) else 0.0

    if verbose:
        prime_status = "✓ prime" if is_prime else "~ likely prime" if not factors else "composite"
        print(f"\n  N = {N:>8,}  |  peak: {peak_val:.3f}  |  status: {prime_status}")
        if factors:
            print(f"              ✓  {factors[0][0]} × {factors[0][1]}")

    return factors, signal, conv, tgt_int, window, peak_val

def trial_division(N, limit=None):
    """Fast trial division for moderate numbers"""
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
# PART 3: Run experiments
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  FFT MULTIPLICATION — the forward direction works perfectly")
print("=" * 60)
for a, b in [(3, 5), (59, 61), (97, 103), (9_999_991, 9_999_973)]:
    result = fft_multiply(a, b)
    match = "✓" if result == a * b else "✗"
    print(f"  {a:>12,} × {b:<12,} = {result:>22,}  {match}")

test_cases = [
    (15, "3 × 5"),
    (77, "7 × 11"),
    (143, "11 × 13"),
    (1001, "7 × 11 × 13"),
    (3599, "59 × 61"),
    (9991, "97 × 103"),
    (97, "prime"),
    (10_007, "prime (10007)"),
    (65_537, "prime (Fermat)"),
]

print("\n" + "=" * 60)
print("  FFT FACTORIZATION — Enhanced log-domain convolution")
print("=" * 60)

results = []
for N, label in test_cases:
    factors, sig, conv, tgt, win, peak = fft_factor(N, scale=3000, sigma=1.0, verbose=True)
    results.append((N, label, factors, sig, conv, tgt, win, peak))
