"""
Representation tables r(x | W) for the mixed bases of P(n,3), n = 6k + p,
derived from the closed-form distance formula (distances.py).

For each element type (u_i, v_i, u_i u_{i+1}, u_i v_i, v_i v_{i+3}) the vector
r(x_i | W) is piecewise affine in (i, k) on residue classes of i. We fit the
pieces on a few k, then check the fitted table against the closed-form distances
for EVERY k with n0 <= n <= N_CHECK.
"""
import numpy as np

from distances import d
import os as _os, sys as _sys
_h = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_h, ".."), _os.path.join(_h, "..", "..")]   # folder containing mdtool
from mdtool.tables import fit_group, _expr, _coord, _lin

GROUPS = [("u_i", "V", "u", 0), ("v_i", "V", "v", 0),
          ("u_iu_{i+1}", "E", ("u", "u"), 1), ("u_iv_i", "E", ("u", "v"), 0),
          ("v_iv_{i+3}", "E", ("v", "v"), 3)]


def W_of(name, n):
    from check_bases import W5, W0, W3, W4
    return {"W5": W5, "W0": W0, "W3": W3, "W4": W4}[name](n)


def rep_rows(n, W, group):
    """Matrix of representations for one element group, rows i = 1..n."""
    gname, kind, X, off = group
    R = []
    for i in range(1, n + 1):
        if kind == "V":
            R.append([d(n, X, i, Y, j) for (Y, j) in W])
        else:
            A, B = X
            j2 = (i - 1 + off) % n + 1
            R.append([min(d(n, A, i, Y, j), d(n, B, j2, Y, j)) for (Y, j) in W])
    return np.array(R, dtype=int)


def make_table(basis, p, M=6, k_start=5, samples=6):
    for shift in range(0, 30, 2):
        ks = list(range(k_start + shift, k_start + shift + samples))
        fitted = {}
        ok = True
        for g in GROUPS:
            rows = fit_group({k: rep_rows(M * k + p, W_of(basis, M * k + p), g) for k in ks})
            if rows is None:
                ok = False
                break
            fitted[g[0]] = rows
        if ok:
            return ks, fitted
    raise RuntimeError(f"no table for {basis}, p={p}")


def table_matches(fitted, basis, p, k, M=6):
    n = M * k + p
    for g in GROUPS:
        R = rep_rows(n, W_of(basis, n), g)
        pred = {}
        for row in fitted[g[0]]:
            e = row.evaluate(k)
            if e is None:
                return False
            pred.update(e)
        if sorted(pred) != list(range(1, n + 1)):
            return False
        if any(list(R[i - 1]) != pred[i] for i in pred):
            return False
    return True


def basis_tex(basis, p, ks, M=6):
    out = []
    for (Y, _), idx in zip(W_of(basis, M * ks[0] + p), range(10)):
        fit = _lin({k: W_of(basis, M * k + p)[idx][1] for k in ks})
        out.append(f"{Y}_{{{_expr(fit[0], 0, fit[1])}}}")
    return r"\{" + ", ".join(out) + r"\}"


def latex_table(fitted, basis, p, ks, label, caption):
    nform = f"n=6k" + (f"+{p}" if p else "")
    lines = [r"\begin{longtable}{lll}",
             rf"\caption{{{caption} $W={basis_tex(basis, p, ks)}$, ${nform}$.}}\label{{{label}}}\\",
             r"\hline Element & Range & $r(\cdot\,|\,W)$ \\ \hline", r"\endfirsthead",
             r"\hline Element & Range & $r(\cdot\,|\,W)$ \\ \hline", r"\endhead"]
    for g in GROUPS:
        for row in fitted[g[0]]:
            a, b, c = row.tex(g[0])
            lines.append(f"{a} & {b} & {c} \\\\")
    lines += [r"\hline", r"\end{longtable}"]
    return "\n".join(lines)
