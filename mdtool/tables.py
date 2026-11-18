"""
Symbolic representation tables r(v | W) for a basis formula.

Write n = M*k + p (M = 2 or 4). For each layer (or edge type) the vector
r(v_i | W) is piecewise linear in i, possibly separately on residue classes
of i (i odd / i even, or mod 3, 4) and possibly with fractional slope such as
(i+1)/2. Breakpoints and intercepts are linear in k.

We compute exact distances for several consecutive k, split each layer into
maximal linear runs, fit every breakpoint and coefficient as an integer-linear
function of k, and then CHECK the fitted table against exact distances at a
value of k not used in the fit. Only checked tables should be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .families import rule_label


# ----------------------------------------------------------------- helpers
def _lin(vals: Dict[int, int]) -> Optional[Tuple[int, int]]:
    """Fit v(k) = a*k + b with integers; None if not exactly linear."""
    ks = sorted(vals)
    if len(ks) == 1:
        return (0, vals[ks[0]])
    a = vals[ks[1]] - vals[ks[0]]
    b = vals[ks[0]] - a * ks[0]
    return (a, b) if all(vals[k] == a * k + b for k in ks) else None


def _expr(ck: int, ci: int, c0: int, ivar: str = "i") -> str:
    parts = []
    terms = [(ck, "k"), (ci, ivar)]
    terms.sort(key=lambda t: t[0] < 0)          # positive terms first: i-k-1, 2k-i+2
    for coef, sym in terms:
        if coef == 0:
            continue
        mag = "" if abs(coef) == 1 else str(abs(coef))
        parts.append(("-" if coef < 0 else "+") + mag + sym)
    if c0 or not parts:
        parts.append(("-" if c0 < 0 else "+") + str(abs(c0)))
    if parts[0].startswith("-"):                 # "-i+2" -> "2-i"
        pos = next((j for j, t in enumerate(parts) if t.startswith("+")), None)
        if pos is not None:
            parts.insert(0, parts.pop(pos))
    s = "".join(parts)
    return s[1:] if s.startswith("+") else s


def _coord(ci: int, g: Tuple[int, int], q: int) -> str:
    """(ci*i + g0*k + g1) / q, simplified when divisible."""
    if ci % q == 0 and g[0] % q == 0 and g[1] % q == 0:
        return _expr(g[0] // q, ci // q, g[1] // q)
    return rf"\frac{{{_expr(g[0], ci, g[1])}}}{{{q}}}"


def _segments(R: np.ndarray):
    """Greedy split of rows into maximal runs with a constant step vector."""
    segs, a, s = [], 0, len(R)
    while a < s:
        if a == s - 1:
            segs.append((a, a, np.zeros(R.shape[1], dtype=int)))
            break
        delta = R[a + 1] - R[a]
        b = a + 1
        while b + 1 < s and np.array_equal(R[b + 1] - R[b], delta):
            b += 1
        segs.append((a, b, delta))
        a = b + 1
    return segs


@dataclass
class Row:
    lo: Tuple[int, int]              # 1-based first i, as a*k + b
    hi: Tuple[int, int]              # 1-based last i
    num: np.ndarray                  # numerator coefficient of i, per coordinate
    gamma: List[Tuple[int, int]]     # numerator constant (linear in k), per coordinate
    q: int = 1                       # i steps by q (one residue class); value = (num*i + gamma)/q

    def tex(self, name: str) -> Tuple[str, str, str]:
        lo, hi = _expr(self.lo[0], 0, self.lo[1]), _expr(self.hi[0], 0, self.hi[1])
        if self.lo == self.hi:
            rng = f"$i = {lo}$"
        else:
            rng = f"${lo} \\le i \\le {hi}"
            if self.q > 1:
                if self.lo[0] % self.q == 0:
                    res = self.lo[1] % self.q
                    rng += (r",\ i \text{ " + ("even" if res == 0 else "odd") + "}" if self.q == 2
                            else rf",\ i \equiv {res} \pmod{{{self.q}}}")
                else:
                    rng += rf",\ i \text{{ in steps of }} {self.q}"
            rng += "$"
        coords = ",\\ ".join(_coord(int(c), g, self.q) for c, g in zip(self.num, self.gamma))
        return f"${name}$", rng, f"$({coords})$"

    def evaluate(self, k: int):
        lo, hi = self.lo[0] * k + self.lo[1], self.hi[0] * k + self.hi[1]
        out = {}
        for i in range(lo, hi + 1, self.q):
            vals = [int(c) * i + g[0] * k + g[1] for c, g in zip(self.num, self.gamma)]
            if any(v % self.q for v in vals):
                return None
            out[i] = [v // self.q for v in vals]
        return out


def _fit_residue(samples, q, r):
    subs = {k: R[r::q] for k, R in samples.items()}
    if any(len(R) == 0 for R in subs.values()):
        return [] if all(len(R) == 0 for R in subs.values()) else None
    segs = {k: _segments(R) for k, R in subs.items()}
    counts = {len(v) for v in segs.values()}
    if len(counts) != 1:
        return None
    rows = []
    for j in range(counts.pop()):
        steps = [segs[k][j][2] for k in subs]
        if any(not np.array_equal(steps[0], d) for d in steps):
            return None
        step = steps[0]                          # change per q steps of i  -> slope step/q
        pos = lambda a: r + 1 + q * a            # 1-based i of sub-row a
        lo = _lin({k: pos(segs[k][j][0]) for k in subs})
        hi = _lin({k: pos(segs[k][j][1]) for k in subs})
        if lo is None or hi is None:
            return None
        gamma = []
        for c in range(len(step)):
            # q*value = step*i + G   at the first row of the run
            g = _lin({k: q * int(subs[k][segs[k][j][0], c]) - int(step[c]) * pos(segs[k][j][0])
                      for k in subs})
            if g is None:
                return None
            gamma.append(g)
        single = lo == hi
        rows.append(Row(lo, hi, step if not single else np.zeros_like(step),
                        gamma if not single else [(g[0], g[1] + int(s) * 0) for g, s in zip(gamma, step)],
                        q))
        if single:   # constant row: value = gamma/q exactly; recompute without i-term
            rows[-1].num = np.zeros_like(step)
            rows[-1].gamma = [_lin({k: q * int(subs[k][segs[k][j][0], c]) for k in subs})
                              for c in range(len(step))]
    return rows


def fit_group(samples: Dict[int, np.ndarray]) -> Optional[List[Row]]:
    """Try one piecewise-linear table, then split i by residue mod 2, 3, 4."""
    for q in (1, 2, 3, 4):
        rows, ok = [], True
        for r in range(q):
            part = _fit_residue(samples, q, r)
            if part is None:
                ok = False
                break
            rows += part
        if ok:
            rows.sort(key=lambda w: (w.lo[0], w.lo[1]))
            return rows
    return None


# ----------------------------------------------------------------- groups
def _vertex_groups(study, n, basis_vertices):
    fam, inst = study.family, study.instance(n, "dim")
    cols = [inst.index[v] for v in basis_vertices]
    out = {}
    for L in fam.layer_order:
        idx = [inst.index[(L, i)] for i in range(fam.size(L, n))]
        out[f"{L}_i"] = inst.D[np.ix_(idx, cols)]
    return out


def _edge_groups(study, n, basis_vertices):
    fam, inst = study.family, study.instance(n, "dim")
    cols = [inst.index[v] for v in basis_vertices]
    out = {}
    for rule, lst in fam.edge_types(n).items():
        if not lst:
            continue
        lst = sorted(lst)
        R = np.array([np.minimum(inst.D[inst.index[u], cols], inst.D[inst.index[v], cols])
                      for _, (u, v) in lst])
        out[rule_label(rule)] = R
    return out


@dataclass
class SymbolicTable:
    M: int
    p: int
    basis_tex: str
    groups: Dict[str, Optional[List[Row]]]
    checked_k: int
    check_ok: bool
    ks: Sequence[int]
    valid_n_from: Optional[int] = None      # table matches exact distances for all such n up to the check

    @property
    def nform(self):
        return f"n = {self.M}k" + (f"+{self.p}" if self.p else "")

    def latex(self, caption: str = "") -> str:
        rng = (f" Checked against exact distances for {self.valid_n_from} $\\le n \\le$ {self.M * self.checked_k + self.p}."
               if self.check_ok else " NOT verified.")
        lines = [r"\begin{table}[H]", r"\centering", r"\renewcommand{\arraystretch}{1.35}",
                 rf"\caption{{{caption}Representations with respect to $W = {self.basis_tex}$, ${self.nform}$.{rng}}}",
                 r"\begin{tabular}{lll}", r"\hline",
                 r"Element & Range & $r(\cdot\,|\,W)$ \\ \hline"]
        for name, rows in self.groups.items():
            if rows is None:
                lines.append(rf"${name}$ & \multicolumn{{2}}{{l}}{{(no linear pattern found; see numeric table)}} \\")
                continue
            for r in rows:
                a, b, c = r.tex(name)
                lines.append(f"{a} & {b} & {c} \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def markdown(self) -> str:
        out = [f"**{self.nform}**, W = ${self.basis_tex}$  "
               f"(fitted on k = {self.ks[0]}..{self.ks[-1]}; checked against exact distances: "
               + (f"PASS for every {self.nform} with {self.valid_n_from} <= n <= {self.M * self.checked_k + self.p})"
                  if self.check_ok else "FAIL)"), "",
               "| Element | Range | r(· \\| W) |", "|---|---|---|"]
        for name, rows in self.groups.items():
            if rows is None:
                out.append(f"| ${name}$ | no linear pattern | |")
                continue
            for r in rows:
                a, b, c = r.tex(name)
                out.append(f"| {a} | {b} | {c} |")
        return "\n".join(out)


def symbolic_table(study, formula, p: int, n_from: int, kind: str = "vertex",
                   M: int = 2, samples: int = 6, fit_from_n: int = 16) -> SymbolicTable:
    """n = M*k + p. kind: 'vertex' (dim, ftdim, ldim, sdim), 'edge' (edim), 'mixed'.
    The pattern is fitted on n >= fit_from_n, then checked exactly for every n of this
    form from the smallest admissible value up to a held-out larger n."""
    k0 = max(-(-(max(n_from, fit_from_n) - p) // M), 1)
    ks = list(range(k0, k0 + samples))
    k_check = k0 + samples + 3

    def groups_at(k):
        n = M * k + p
        S = formula.vertices(n)
        g = {}
        if kind in ("vertex", "mixed"):
            g.update(_vertex_groups(study, n, S))
        if kind in ("edge", "mixed"):
            g.update(_edge_groups(study, n, S))
        return g

    # adaptive start: patterns often stabilise only for larger n
    for shift in (0, 2, 4, 6, 8):
        ks = list(range(k0 + shift, k0 + shift + samples))
        data = {k: groups_at(k) for k in ks}
        names = list(data[ks[0]])
        fitted = {nm: fit_group({k: data[k][nm] for k in ks}) for nm in names}
        if all(v is not None for v in fitted.values()):
            break
    k_check = ks[-1] + 3
    k0 = ks[0]

    btex = []
    for t in formula.templates:
        fit = _lin({k: t.idx(M * k + p) + 1 for k in ks})
        btex.append(f"{t.layer}_{{{_expr(fit[0], 0, fit[1])}}}" if fit else t.tex())
    basis_tex = r"\{" + ",\\ ".join(btex) + r"\}"

    def matches(k):
        try:
            truth = groups_at(k)
        except Exception:
            return False
        for nm, rows in fitted.items():
            if rows is None:
                return False
            pred = {}
            for r in rows:
                e = r.evaluate(k)
                if e is None:
                    return False
                pred.update(e)
            R = truth[nm]
            if sorted(pred) != list(range(1, len(R) + 1)) or any(list(R[i - 1]) != pred[i] for i in pred):
                return False
        return True

    ok = all(matches(k) for k in range(k0, k_check + 1))
    valid_from = None
    if ok:
        kk = k0
        while kk - 1 >= 1 and M * (kk - 1) + p >= max(n_from, study.family.min_n) and matches(kk - 1):
            kk -= 1
        valid_from = M * kk + p
    return SymbolicTable(M, p, basis_tex, fitted, k_check, ok, ks, valid_from)


def best_tables(study, formula, n_from: int, kind="vertex", parities=(0, 1)):
    """Cases n = 2k, 2k+1 if those fit; otherwise n mod 4, then n mod 6, then n mod 12."""
    best = None
    for M in (2, 4, 6, 12):
        tabs = [symbolic_table(study, formula, p, n_from, kind, M=M)
                for p in range(M) if p % 2 in parities]
        score = sum(t.check_ok for t in tabs) / len(tabs)
        if score == 1:
            return tabs
        if best is None or score > best[0]:
            best = (score, tabs)
    return best[1]


def numeric_table(study, formula, n: int, kind="vertex") -> str:
    """Plain numeric table for one n (appendix / sanity check)."""
    S = formula.vertices(n)
    g = {}
    if kind in ("vertex", "mixed"):
        g.update(_vertex_groups(study, n, S))
    if kind in ("edge", "mixed"):
        g.update(_edge_groups(study, n, S))
    out = [f"n = {n}, W = {[f'{L}{i+1}' for L, i in S]}", ""]
    for nm, R in g.items():
        out.append(f"{nm}: " + "  ".join(f"{i+1}:({','.join(map(str, row))})" for i, row in enumerate(R)))
    return "\n".join(out)
