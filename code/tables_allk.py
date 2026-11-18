"""
From 'the table agrees with the distance formula for all k <= K' to 'for ALL k'.

Fix a table row (elements x_i, i = a*k + b + q*s, 0 <= s <= S(k)), a coordinate
(basis vertex w at index alpha*k + beta) and residues k = 3*kap + rho,
s = 3*sig + tau. For n >= 24 the true distance is
    g = min over endpoints e of x_i and m in {-1,0,1} of F(|w - e + m n|),
and with residues fixed, every candidate value is an AFFINE function of
(kap, sig) valid on a half-plane (sign of w - e + m n), F_uu itself being a
minimum of two affine functions. The table entry T is affine in (kap, sig).

All lines on which some comparison or sign can change are listed: sign lines,
pairwise equalities of candidates (and of T), and the row boundaries. Beyond the
largest kappa at which two such lines cross (KAP0), every cell of the
arrangement is a wedge or strip on which g is ONE affine candidate. Agreement of
T with g at the lattice points of every cell for KAP0 < kap <= KAP0 + MARGIN
then forces T = g on the whole cell, i.e. for all larger k.

We (1) compute KAP0 exactly, (2) check agreement for every k up to
3*(KAP0 + MARGIN) + 2 using the closed-form distance (tables_p3.table_matches).
"""
from fractions import Fraction as Fr
from itertools import combinations

from tables_p3 import GROUPS, W_of, make_table, table_matches
import os as _os, sys as _sys
_h = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_h, ".."), _os.path.join(_h, "..", "..")]   # folder containing mdtool
from mdtool.tables import _lin

MARGIN = 30


def affine(xk, xs, x0):
    return (Fr(xk), Fr(xs), Fr(x0))


def candidates_for(Xtype, Ytype, x):
    """Affine pieces of F_XY(|x|) where x = (xk*kap + xs*sig + x0) with x0 = x mod 3 fixed.
    Returns list of (sign, affine) with sign +1 (valid where x >= 0) or -1 (x <= 0)."""
    xk, xs, x0 = x
    out = []
    for sign in (+1, -1):
        yk, ys, y0 = sign * xk, sign * xs, sign * x0
        R = y0 % 3
        Q = (Fr(yk, 3), Fr(ys, 3), Fr(y0 - R, 3))          # (y - R)/3, affine (exact: yk, ys mult of 3)
        if Xtype == Ytype == "u":
            out.append((sign, (yk, ys, y0)))                 # t itself
            out.append((sign, (Q[0], Q[1], Q[2] + 2 + R)))
        elif Xtype != Ytype:
            out.append((sign, (Q[0], Q[1], Q[2] + 1 + R)))
        else:
            out.append((sign, (Q[0], Q[1], Q[2] + (0, 3, 4)[R])))
    return out


def lines_for_row(group, row, wtypes, walpha, wbeta, p):
    """All critical lines c_k*kap + c_s*sig + c_0 = 0 for one row, all residues."""
    gname, kind, X, off = group
    (a, b), (ha, hb), q = row.lo, row.hi, row.q
    ends = [(X, 0)] if kind == "V" else [(X[0], 0), (X[1], off)]
    lines = []
    for rho in range(3):
        for tau in range(3):
            # i = a*(3kap+rho) + b + q*(3sig+tau)
            for (Y, al, be) in zip(wtypes, walpha, wbeta):
                cands = []
                for (Xe, oe) in ends:
                    for m in (-1, 0, 1):
                        # x = w - e + m*n,  n = 6k + p,  k = 3kap + rho
                        xk = 3 * (al - a + 6 * m)
                        xs = -3 * q
                        x0 = (al - a + 6 * m) * rho + be - b - q * tau - oe + m * p
                        lines.append((Fr(xk), Fr(xs), Fr(x0)))       # sign line x = 0
                        cands += [aff for _, aff in candidates_for(Xe, Y, (xk, xs, x0))]
                # equalities between candidates (the table entry itself is affine on each
                # cell, so no lines involving it are needed)
                for A, B in combinations(cands, 2):
                    lines.append(tuple(Fr(u) - Fr(v) for u, v in zip(A, B)))
            # row boundaries in (kap, sig): s >= 0  and  q*s <= (ha-a)*k + (hb-b)
            lines.append((Fr(0), Fr(3), Fr(tau)))
            lines.append((Fr(3 * (ha - a)), Fr(-3 * q), Fr((ha - a) * rho + hb - b - q * tau)))
    return lines


def max_crossing(lines):
    """Largest kappa at which two non-parallel lines cross."""
    best = Fr(-10**9)
    L = [l for l in set(lines) if l[1] != 0 or l[0] != 0]
    for (k1, s1, c1), (k2, s2, c2) in combinations(L, 2):
        det = k1 * s2 - k2 * s1
        if det == 0:
            continue
        kap = (s1 * c2 - s2 * c1) / det
        best = max(best, kap)
    return best


def geometry(lines):
    """For the non-vertical critical lines: smallest positive difference of slopes (d sigma/d kappa)
    and the largest kappa-period of lattice points along a line direction."""
    from math import gcd
    slopes, period = set(), 1
    for ck, cs, c0 in set(lines):
        if cs == 0:
            continue                      # vertical: lies at kappa <= kappa0 (counted in crossings)
        slopes.add(-ck / cs)
        L = 1
        for f in (ck, cs):
            L = L * f.denominator // gcd(L, f.denominator)
        a, b = int(ck * L), int(cs * L)
        g = gcd(a, b)
        period = max(period, abs(b) // g)
    sl = sorted(slopes)
    delta = min((y - x for x, y in zip(sl, sl[1:])), default=None)
    return delta, period


def analyse(basis, p):
    ks, fit = make_table(basis, p)
    W = W_of(basis, 6 * ks[0] + p)
    wtypes = [Y for Y, _ in W]
    fits = [_lin({k: W_of(basis, 6 * k + p)[c][1] for k in ks}) for c in range(len(W))]
    walpha = [f[0] for f in fits]
    wbeta = [f[1] for f in fits]
    kap0, delta, period = Fr(-10**9), None, 1
    for g in GROUPS:
        for row in fit[g[0]]:
            L = lines_for_row(g, row, wtypes, walpha, wbeta, p)
            kap0 = max(kap0, max_crossing(L))
            d, per = geometry(L)
            if d is not None:
                delta = d if delta is None else min(delta, d)
            period = max(period, per)
    return ks, fit, kap0, delta, period


if __name__ == "__main__":
    import sys
    plan = [("W0", 0, 12), ("W3", 3, 21), ("W4", 4, 16), ("W5", 1, 19), ("W5", 2, 20), ("W5", 5, 17)]
    for basis, p, n0 in plan:
        ks, fit, kap0, delta, period = analyse(basis, p)
        kmin = -(-(n0 - p) // 6)
        need_k = max(3 * (int(kap0) + 1 + MARGIN) + 2, 4)       # check every k up to this
        # cells beyond kappa0: diverging ones reach width >= 3 within ceil(3/delta) steps;
        # parallel strips repeat with kappa-period <= period. Both must fit in the margin.
        steps = -(-3 // delta) if delta else 0
        geom_ok = steps + 2 <= MARGIN and 2 * period + 1 <= MARGIN
        ok = all(table_matches(fit, basis, p, k) for k in range(max(kmin, 4), need_k + 1))
        print(f"{basis} n=6k+{p}: last crossing kappa0 = {float(kap0):.2f} (k < {3 * (int(kap0) + 1)}); "
              f"min slope gap = {delta}, max period = {period}, margin {MARGIN}: geometry {'OK' if geom_ok else 'FAILS'}; "
              f"table = distance formula for every k in [{max(kmin, 4)}, {need_k}]: {ok}"
              + ("  => holds for ALL k >= %d" % max(kmin, 4) if ok and geom_ok else ""), flush=True)
