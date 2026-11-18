"""
Find a basis *formula* that works for every n in a range.

A position template picks one vertex of a layer as a function of n, e.g.
  a_1,  a_2,  a_{floor(n/2)+1},  a_{ceil(n/2)},  a_n,  b_{n-1}, ...
A candidate basis is k templates. It is accepted only if, for EVERY n tested,
the k vertices are distinct and form a resolving set (in the chosen variant).
Since k equals the exact value from the ILP, an accepted formula is a metric
basis for every n tested, which is exactly the statement a paper then proves.

If the family is invariant under rotation i -> i+m, one basis vertex may be
assumed WLOG to have a small index, which shrinks the search a lot.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Template:
    layer: str
    kind: str          # 'const' | 'floor' | 'ceil' | 'last' | 'frac'
    j: int             # offset parameter
    m: int             # layer size = m*n + c
    c: int
    fp: int = 1        # for 'frac': position floor(fp*s/fq) + j
    fq: int = 2

    def size(self, n):
        return self.m * n + self.c

    def idx(self, n) -> int:
        """0-based index for this n (may be out of range -> -1)."""
        s = self.size(n)
        if self.kind == "const":
            k = self.j - 1
        elif self.kind == "floor":
            k = s // 2 + self.j - 1
        elif self.kind == "ceil":
            k = (s + 1) // 2 + self.j - 1
        elif self.kind == "frac":
            k = (self.fp * s) // self.fq + self.j - 1
        else:  # last
            k = s - self.j
        return k if 0 <= k < s else -1

    @property
    def complexity(self) -> int:
        return {"const": 0, "last": 1, "floor": 2, "ceil": 2, "frac": 3}[self.kind] + abs(self.j) // 3

    def size_tex(self) -> str:
        if self.m == 1 and self.c == 0:
            return "n"
        base = "n" if self.m == 1 else f"{self.m}n"
        return base + (f"+{self.c}" if self.c > 0 else (f"{self.c}" if self.c < 0 else ""))

    def tex(self) -> str:
        s = self.size_tex()
        j = self.j
        off = lambda t: (f"+{t}" if t > 0 else (f"{t}" if t < 0 else ""))
        if self.kind == "const":
            sub = str(j)
        elif self.kind == "floor":
            sub = rf"\lfloor {s}/2 \rfloor{off(j)}"
        elif self.kind == "ceil":
            sub = rf"\lceil {s}/2 \rceil{off(j)}"
        elif self.kind == "frac":
            num = s if self.fp == 1 else (f"{self.fp}n" if s == "n" else f"{self.fp}({s})")
            sub = rf"\lfloor {num}/{self.fq} \rfloor{off(j)}"
        else:
            sub = f"{s}{off(1 - j)}" if self.m else str(self.c + 1 - j)
        return f"{self.layer}_{{{sub}}}"


def candidate_templates(family, ns: Sequence[int], cmax: int = 4) -> List[Template]:
    out, seen = [], {}
    for L in family.layer_order:
        m, c = family.layers[L]
        if m == 0:
            raw = [Template(L, "const", j, m, c) for j in range(1, c + 1)]
        else:
            raw = ([Template(L, "const", j, m, c) for j in range(1, cmax + 1)] +
                   [Template(L, "floor", j, m, c) for j in range(-3, 4)] +
                   [Template(L, "ceil", j, m, c) for j in range(-3, 4)] +
                   [Template(L, "last", j, m, c) for j in range(1, cmax)] +
                   [Template(L, "frac", j, m, c, fp, fq)
                    for fp, fq in ((1, 4), (3, 4), (1, 3), (2, 3)) for j in range(-1, 3)])
        for t in sorted(raw, key=lambda t: t.complexity):
            sig = (L, tuple(t.idx(n) for n in ns))
            if -1 in sig[1]:
                continue
            if sig not in seen:          # identical on every n -> keep simplest
                seen[sig] = t
                out.append(t)
    return out


@dataclass
class BasisFormula:
    templates: Tuple[Template, ...]
    holds_for: Tuple[int, ...]

    @property
    def score(self):
        return sum(t.complexity for t in self.templates) + len({t.layer for t in self.templates}) * 0.1

    def tex(self) -> str:
        return r"\{" + ",\\ ".join(t.tex() for t in self.templates) + r"\}"

    def vertices(self, n):
        return [(t.layer, t.idx(n)) for t in self.templates]


def find_basis_formulas(study, variant: str, ns: Sequence[int], k: int,
                        max_results: int = 20, time_limit: float = 120.0,
                        use_symmetry: bool = True, cmax: int = 6) -> List[BasisFormula]:
    """All template bases of size k (up to max_results) valid for every n in ns."""
    fam = study.family
    ns = sorted(ns)
    insts = {n: study.instance(n, variant) for n in ns}
    cands = candidate_templates(fam, ns, cmax=cmax)

    # column index of each candidate for each n
    colmap = {n: np.array([insts[n].index[(t.layer, t.idx(n))] for t in cands]) for n in ns}
    # check order: a couple of mid/large n first (most discriminating), then the rest
    order = sorted(ns, key=lambda n: (n % 2, -n))

    symmetric = use_symmetry and fam.rotation_is_automorphism(max(ns))
    if symmetric:
        firsts = [a for a, t in enumerate(cands)
                  if fam.layers[t.layer][0] > 0 and t.kind == "const" and t.j <= fam.layers[t.layer][0]]
        fixed_layer = [a for a, t in enumerate(cands) if fam.layers[t.layer][0] == 0]
        combos = itertools.chain(
            ((f,) + rest for f in firsts
             for rest in itertools.combinations([b for b in range(len(cands)) if b > f or b not in firsts], k - 1)),
            itertools.combinations(fixed_layer, k))
    else:
        combos = itertools.combinations(range(len(cands)), k)

    # sentinel rows: the pairs resolved by the fewest vertices fail most candidates fast
    sentinel = {}
    for n in ns:
        inst = insts[n]
        cnt = inst.rows.sum(axis=1)
        sentinel[n] = inst.rows[np.argsort(cnt, kind="stable")[:400]]

    found, seen_sets, t0 = [], set(), time.time()
    for combo in combos:
        if time.time() - t0 > time_limit:
            break
        combo = tuple(sorted(combo))
        if combo in seen_sets:
            continue
        seen_sets.add(combo)
        ok = True
        for n in order:
            cols = colmap[n][list(combo)]
            if (sentinel[n][:, cols].sum(axis=1) < insts[n].r).any():
                ok = False
                break
        if not ok:
            continue
        for n in order:
            cols = colmap[n][list(combo)]
            if len(set(cols.tolist())) < k:
                ok = False
                break
            inst = insts[n]
            if (inst.rows[:, cols].sum(axis=1) < inst.r).any():
                ok = False
                break
        if ok:
            found.append(BasisFormula(tuple(cands[a] for a in combo), tuple(ns)))
            if len(found) >= max_results * 5:
                break
    found.sort(key=lambda b: b.score)
    return found[:max_results]


def valid_from(study, variant: str, formula: BasisFormula, n_hi: int, step: int = 1) -> int:
    """Smallest n0 such that the formula is a resolving set for all n0 <= n <= n_hi
    (scanning downward from n_hi; step=2 restricts to n of the same parity as n_hi)."""
    from .core import is_resolving
    n0 = n_hi + step
    for n in range(n_hi, study.family.min_n - 1, -step):
        try:
            S = formula.vertices(n)
        except Exception:
            break
        if any(i < 0 for _, i in S) or len(set(S)) < len(S):
            break
        if not is_resolving(study.instance(n, variant), S)[0]:
            break
        n0 = n
    return n0


def ranked_formulas(study, variant, ns, k, step=1, **kw):
    """find_basis_formulas + the range of validity, best first:
    valid from the smallest n, then simplest positions."""
    res = find_basis_formulas(study, variant, ns, k, **kw)
    out = [(valid_from(study, variant, b, max(ns), step), b) for b in res]
    out.sort(key=lambda t: (t[0], t[1].score))
    return out
