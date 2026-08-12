from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, clamped to [0, 1].
    With no observations at all the honest interval is the whole range: we
    know nothing about the rate, rather than knowing it is zero.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # The bounds are analytically exactly 0 and 1 at the extremes; hold them there so
    # floating-point residue never prints a rate like -6.3e-18.
    lo = 0.0 if k == 0 else max(0.0, (center - margin) / denom)
    hi = 1.0 if k == n else min(1.0, (center + margin) / denom)
    return (lo, hi)


def mean_stderr(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var / n))


_LOGFACT = {0: 0.0}


def _logfact(n: int) -> float:
    if n not in _LOGFACT:
        v = _LOGFACT[max(_LOGFACT)]
        for i in range(max(_LOGFACT) + 1, n + 1):
            v += math.log(i)
            _LOGFACT[i] = v
    return _LOGFACT[n]


def _lhyper(a: int, b: int, c: int, d: int) -> float:
    r1, r2, c1, tot = a + b, c + d, a + c, a + b + c + d
    return (
        _logfact(r1)
        + _logfact(r2)
        + _logfact(c1)
        + _logfact(b + d)
        - _logfact(tot)
        - _logfact(a)
        - _logfact(b)
        - _logfact(c)
        - _logfact(d)
    )


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError("counts must be non-negative")
    r1, c1 = a + b, a + c
    r2 = c + d
    obs = _lhyper(a, b, c, d)
    tol = 1e-9
    p = 0.0
    lo = max(0, c1 - r2)
    hi = min(c1, r1)
    for x in range(lo, hi + 1):
        logp = _lhyper(x, r1 - x, c1 - x, r2 - (c1 - x))
        if logp <= obs + tol:
            p += math.exp(logp)
    return min(1.0, p)
