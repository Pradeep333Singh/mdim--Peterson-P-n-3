"""
Exact proof (for ALL k >= k_min) that the rows of a representation table never
produce the same vector for two different elements.

A table row describes elements x_i, i = lo(k), lo(k)+q, ..., hi(k), with
coordinates  r_c(x_i) = (num_c * i + g_c(k)) / q,  where lo, hi, g_c are
integer-affine in k.  For two rows R1 (index i = lo1 + q1*s) and R2
(index j = lo2 + q2*t) a collision means integers s, t, k with

    q2*(num1_c*i + g1_c(k)) = q1*(num2_c*j + g2_c(k))   for every coordinate c,
    0 <= q1*s <= hi1(k) - lo1(k),   0 <= q2*t <= hi2(k) - lo2(k),   k >= k_min,
    (and s != t when R1 is R2).

All equations are linear in (s, t, k) with integer coefficients. We decide
existence of an integer solution exactly (rational arithmetic, no floating
point), covering every k >= k_min, including the unbounded range.
"""
from fractions import Fraction as Fr
from math import gcd
from itertools import combinations_with_replacement


def lcm(a, b):
    return a * b // gcd(a, b)


def row_eqs(R1, R2):
    """Equations sum(cs*s + ct*t + ck*k + c0) = 0, one per coordinate."""
    (a1, b1), q1 = R1.lo, R1.q
    (a2, b2), q2 = R2.lo, R2.q
    eqs = []
    for c in range(len(R1.num)):
        n1, n2 = int(R1.num[c]), int(R2.num[c])
        g1, g2 = R1.gamma[c], R2.gamma[c]
        # q2*(n1*(a1 k + b1 + q1 s) + g1[0] k + g1[1]) - q1*(n2*(a2 k + b2 + q2 t) + g2[0] k + g2[1])
        cs = q2 * n1 * q1
        ct = -q1 * n2 * q2
        ck = q2 * (n1 * a1 + g1[0]) - q1 * (n2 * a2 + g2[0])
        c0 = q2 * (n1 * b1 + g1[1]) - q1 * (n2 * b2 + g2[1])
        eqs.append((cs, ct, ck, c0))
    return eqs


def bounds(R):
    """0 <= q*s <= (ha-la) k + (hb-lb)"""
    (la, lb), (ha, hb) = R.lo, R.hi
    return R.q, ha - la, hb - lb


def k_range_from_ineq(coef_k, const):
    """Solve coef_k*k + const >= 0 for real k: returns (lo, hi) with None = unbounded."""
    if coef_k == 0:
        return (None, None) if const >= 0 else "empty"
    x = Fr(-const, 1) / coef_k
    return (x, None) if coef_k > 0 else (None, x)


def feasible(R1, R2, same, kmin, kmax_scan=4000):
    """True if a collision exists for some integer k >= kmin. Exact."""
    eqs = row_eqs(R1, R2)
    q1, A1, B1 = bounds(R1)
    q2, A2, B2 = bounds(R2)

    # We treat k as a parameter. For fixed k the system in (s, t) has constant
    # coefficients (cs, ct); solve it symbolically as affine functions of k.
    M = [(cs, ct) for cs, ct, _, _ in eqs]
    rhs = [(-ck, -c0) for _, _, ck, c0 in eqs]         # rhs = -ck*k - c0, as (k-coef, const)

    # rank of M
    nz = [r for r in M if r != (0, 0)]
    rank = 0
    if nz:
        rank = 1
        r0 = nz[0]
        for r in nz[1:]:
            if r0[0] * r[1] - r0[1] * r[0] != 0:
                rank = 2
                break

    def residual_ok_set(sol_s, sol_t):
        """sol_* are (k-coef, const) Fractions: returns list of affine forms that must vanish."""
        res = []
        for (cs, ct), (rk, rc) in zip(M, rhs):
            fk = cs * sol_s[0] + ct * sol_t[0] - rk
            fc = cs * sol_s[1] + ct * sol_t[1] - rc
            res.append((fk, fc))
        return res

    if rank == 2:
        # pick two independent rows, solve by Cramer's rule
        idx = None
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                det = M[i][0] * M[j][1] - M[i][1] * M[j][0]
                if det != 0:
                    idx = (i, j, det)
                    break
            if idx:
                break
        i, j, det = idx
        (a, b), (c, dd) = M[i], M[j]
        (rik, ric), (rjk, rjc) = rhs[i], rhs[j]
        sol_s = (Fr(rik * dd - b * rjk, det), Fr(ric * dd - b * rjc, det))
        sol_t = (Fr(a * rjk - c * rik, det), Fr(a * rjc - c * ric, det))
        # all residual equations must vanish
        ks_fixed = None
        for fk, fc in residual_ok_set(sol_s, sol_t):
            if fk == 0 and fc != 0:
                return False
            if fk != 0:
                kv = Fr(-fc) / fk
                if ks_fixed is not None and kv != ks_fixed:
                    return False
                ks_fixed = kv
        # inequalities on k
        ineqs = [  # each: coef_k * k + const >= 0
            (sol_s[0], sol_s[1]),                                   # s >= 0
            (A1 - q1 * sol_s[0], B1 - q1 * sol_s[1]),               # q1 s <= A1 k + B1
            (sol_t[0], sol_t[1]),                                   # t >= 0
            (A2 - q2 * sol_t[0], B2 - q2 * sol_t[1]),
            (Fr(1), Fr(-kmin)),                                     # k >= kmin
        ]
        lo, hi = None, None
        for ck, cc in ineqs:
            r = k_range_from_ineq(ck, cc)
            if r == "empty":
                return False
            l, h = r
            if l is not None:
                lo = l if lo is None else max(lo, l)
            if h is not None:
                hi = h if hi is None else min(hi, h)
        if ks_fixed is not None:
            cand = [ks_fixed] if ks_fixed.denominator == 1 else []
        else:
            L = 1
            for f in sol_s + sol_t:
                L = lcm(L, f.denominator)
            start = int(lo) if lo.denominator == 1 else int(lo) + 1
            if hi is None:
                cand = range(start, start + L)          # one full period suffices (unbounded above)
            else:
                cand = range(start, int(hi) + 1)
        for k in cand:
            k = Fr(k)
            if lo is not None and k < lo or hi is not None and k > hi:
                continue
            s = sol_s[0] * k + sol_s[1]
            t = sol_t[0] * k + sol_t[1]
            if s.denominator == 1 and t.denominator == 1:
                if same and s == t:
                    continue
                return True
        return False

    # rank <= 1: scan k explicitly; then argue by periodicity (reported separately)
    return ("scan", rank)


def scan_feasible(R1, R2, same, kmin, kmax):
    """Brute force over k in [kmin, kmax] (used only for rank <= 1 pairs)."""
    eqs = row_eqs(R1, R2)
    q1, A1, B1 = bounds(R1)
    q2, A2, B2 = bounds(R2)
    for k in range(kmin, kmax + 1):
        S, T = (A1 * k + B1) // q1, (A2 * k + B2) // q2
        if S < 0 or T < 0:
            continue
        for s in range(S + 1):
            # solve for t from first equation with ct != 0
            for cs, ct, ck, c0 in eqs:
                if ct != 0:
                    num = -(cs * s + ck * k + c0)
                    if num % ct:
                        t = None
                    else:
                        t = num // ct
                    break
            else:
                t = "any"
            ts = range(T + 1) if t == "any" else ([t] if t is not None and 0 <= t <= T else [])
            for t in ts:
                if same and s == t:
                    continue
                if all(cs * s + ct * t + ck * k + c0 == 0 for cs, ct, ck, c0 in eqs):
                    return k
    return None


def check_table(fitted, kmin, groups):
    rows = [(g[0], r) for g in groups for r in fitted[g[0]]]
    stats = {"pairs": 0, "rank2": 0, "lowrank": 0, "collisions": []}
    lowrank = []
    for (ia, (ga, Ra)), (ib, (gb, Rb)) in combinations_with_replacement(list(enumerate(rows)), 2):
        same = ia == ib
        if same and Ra.lo == Ra.hi:
            continue                                    # single element row
        stats["pairs"] += 1
        res = feasible(Ra, Rb, same, kmin)
        if res is True:
            stats["collisions"].append((ga, Ra, gb, Rb))
        elif isinstance(res, tuple):
            stats["lowrank"] += 1
            lowrank.append((same, Ra, Rb, ga, gb))
        else:
            stats["rank2"] += 1
    return stats, lowrank


def feasible_rank_le1(R1, R2, same, kmin):
    """Exact decision for pairs whose (s,t)-coefficient matrix has rank <= 1.

    Returns (verdict, reason) with verdict in {False, True, None}; None = undecided.
    """
    eqs = row_eqs(R1, R2)
    q1, A1, B1 = bounds(R1)
    q2, A2, B2 = bounds(R2)
    nz = [(cs, ct) for cs, ct, _, _ in eqs if (cs, ct) != (0, 0)]
    # conditions on k: list of (fk, fc) meaning fk*k + fc == 0
    kconds = []
    if nz:
        g = gcd(nz[0][0], nz[0][1])
        al, be = nz[0][0] // g, nz[0][1] // g
        if al < 0 or (al == 0 and be < 0):
            al, be = -al, -be
        u = None                                   # u(k) = alpha*s + beta*t as (k-coef, const) Fractions
        for cs, ct, ck, c0 in eqs:
            if (cs, ct) == (0, 0):
                kconds.append((Fr(ck), Fr(c0)))
                continue
            lam = Fr(cs, al) if al else Fr(ct, be)
            uk, uc = Fr(-ck) / lam, Fr(-c0) / lam
            if u is None:
                u = (uk, uc)
            else:
                kconds.append((uk - u[0], uc - u[1]))
    else:
        al = be = 0
        u = None
        for cs, ct, ck, c0 in eqs:
            kconds.append((Fr(ck), Fr(c0)))

    # resolve k conditions
    kfix = None
    for fk, fc in kconds:
        if fk == 0:
            if fc != 0:
                return False, "inconsistent"
            continue
        kv = -fc / fk
        if kv.denominator != 1 or kv < kmin or (kfix is not None and kv != kfix):
            return False, "k forced to an impossible value"
        kfix = kv

    def exists_at(k):
        S, T = (A1 * k + B1) // q1, (A2 * k + B2) // q2
        if S < 0 or T < 0:
            return False
        if u is None:
            return any(not (same and s == t) for s in range(S + 1) for t in range(min(T, 1) + 1)) \
                if not same else (S >= 1 or T >= 1)
        target = u[0] * k + u[1]
        if target.denominator != 1:
            return False
        target = int(target)
        for s in range(S + 1):
            if be == 0:
                if al * s == target:
                    for t in range(T + 1):
                        if not (same and s == t):
                            return True
                continue
            num = target - al * s
            if num % be == 0:
                t = num // be
                if 0 <= t <= T and not (same and s == t):
                    return True
        return False

    if kfix is not None:
        return exists_at(int(kfix)), "k fixed; exact enumeration"

    if u is None:
        return None, "no equation involves s,t (rows constant)"

    # same row: alpha*s+beta*t with (alpha,beta) ~ (1,-1); s != t
    # real feasibility: min over box <= u(k) <= max over box, for some k >= kmin
    # box: 0 <= s <= (A1 k + B1)/q1 , 0 <= t <= (A2 k + B2)/q2
    Smax = (Fr(A1, q1), Fr(B1, q1))
    Tmax = (Fr(A2, q2), Fr(B2, q2))
    lo_form = (min(al, 0) * Smax[0] + min(be, 0) * Tmax[0], min(al, 0) * Smax[1] + min(be, 0) * Tmax[1])
    hi_form = (max(al, 0) * Smax[0] + max(be, 0) * Tmax[0], max(al, 0) * Smax[1] + max(be, 0) * Tmax[1])
    ineqs = [(u[0] - lo_form[0], u[1] - lo_form[1]),          # u - min >= 0
             (hi_form[0] - u[0], hi_form[1] - u[1]),          # max - u >= 0
             (Smax[0], Smax[1]), (Tmax[0], Tmax[1]), (Fr(1), Fr(-kmin))]
    lo, hi = None, None
    for ck, cc in ineqs:
        r = k_range_from_ineq(ck, cc)
        if r == "empty":
            return False, "box misses line for all k"
        l, h = r
        if l is not None:
            lo = l if lo is None else max(lo, l)
        if h is not None:
            hi = h if hi is None else min(hi, h)
    if hi is not None and lo is not None and hi < lo:
        return False, "box misses line for all k"
    start = int(lo) if lo.denominator == 1 else int(lo) + 1
    if same and u == (0, 0):
        return False, "same row, forces s = t"
    if hi is not None:
        return any(exists_at(k) for k in range(start, int(hi) + 1)), "bounded k range; exact enumeration"
    # unbounded k: alpha*s + beta*s' is an integer, so no solution if lambda(k) is never
    # an integer (gcd(alpha, beta) = 1 after normalisation); lambda has period L in k.
    L = u[0].denominator * u[1].denominator
    if all((u[0] * k + u[1]).denominator != 1 for k in range(start, start + L)):
        return False, "lambda(k) never integral (periodic in k)"
    return None, "unbounded k, line meets box"
