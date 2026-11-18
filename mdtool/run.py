"""
Command-line runner.

Examples
  python -m mdtool.run prism --variants dim edim ftdim --n 4-30
  python -m mdtool.run --spec "name=myfam; a:1,0 b:1,0; a-a:1,1 b-b:1,1 a-b:1,0 a-b:1,1" --n 5-30
  python -m mdtool.run --list

Outputs (in results/<family>/):
  values.csv      exact value of every variant for every n (ILP)
  report.md       readable summary: values, basis formulas, symbolic tables
  paper_tables.tex / .pdf   theorem statements + representation tables (LaTeX)
  graph_n<N>.png/.pdf       drawing of the graph with a metric basis highlighted
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from .core import VARIANTS, Study
from .families import FAMILIES, Family, parse_family_spec
from .patterns import ranked_formulas
from .tables import best_tables, numeric_table

KIND = {"edim": "edge", "ftedim": "edge", "mdim": "mixed"}


# ----------------------------------------------------------------- analysis
def classify(vals: Dict[int, int]):
    """'const' (k), 'parity' ({0: k0, 1: k1}), or 'other'."""
    ns = sorted(vals)
    if len({vals[n] for n in ns}) == 1:
        return "const", vals[ns[0]]
    ev = {vals[n] for n in ns if n % 2 == 0}
    od = {vals[n] for n in ns if n % 2 == 1}
    if len(ev) == 1 and len(od) == 1:
        return "parity", {0: ev.pop(), 1: od.pop()}
    return "other", None


def stable_tail(vals: Dict[int, int], min_len: int = 6):
    """Classify the longest tail (largest n) that is constant or parity-constant.
    Small n often behave differently; papers state results for n >= n0."""
    ns = sorted(vals)
    for start in range(len(ns) - min_len + 1):
        sub = {n: vals[n] for n in ns[start:]}
        kind, k = classify(sub)
        if kind != "other":
            return kind, k, ns[start]
    return "other", None, None


def periodic_tail(vals: Dict[int, int], periods=(1, 2, 3, 4, 6, 8, 12)):
    """Smallest period P such that vals[n] == vals[n+P] on a long enough tail.
    Returns (P, {residue: (value, first n from which that residue is constant)}) or None."""
    ns = sorted(vals)
    hi = ns[-1]
    for P in periods:
        n0 = None
        for n in reversed(ns):
            if n + P > hi:
                continue
            if vals[n] == vals[n + P]:
                n0 = n
            else:
                break
        if n0 is None or (hi - n0 + 1) < max(3 * P, 8):
            continue
        res = {}
        for r in range(P):
            cls = [n for n in ns if n % P == r]
            v = vals[cls[-1]]
            first = cls[-1]
            for n in reversed(cls):
                if vals[n] == v:
                    first = n
                else:
                    break
            res[r] = (v, first)
        return P, res
    return None


def guess_closed_form(vals: Dict[int, int]) -> Optional[Tuple[str, int]]:
    """Search floor((a*n+b)/c) and ceil((a*n+b)/c) fitting a tail of the data."""
    ns = sorted(vals)
    best = None
    for a in range(1, 5):
        for c in range(1, 7):
            if math.gcd(a, c) != 1:
                continue
            for b in range(-8, 9):
                for name, f in (("floor", lambda x: math.floor(x)), ("ceil", lambda x: math.ceil(x))):
                    n0 = None
                    for n in reversed(ns):
                        if f((a * n + b) / c) == vals[n]:
                            n0 = n
                        else:
                            break
                    if n0 is None or ns[-1] - n0 < 7:
                        continue
                    num = (f"{a}n" if a > 1 else "n") + (f"+{b}" if b > 0 else (f"{b}" if b < 0 else ""))
                    expr = num if c == 1 else (rf"\lfloor ({num})/{c} \rfloor" if name == "floor"
                                              else rf"\lceil ({num})/{c} \rceil")
                    cand = (n0, a + c + abs(b), name != "floor", expr)
                    if best is None or cand < best:
                        best = cand
    return (best[3], best[0]) if best else None


# ----------------------------------------------------------------- drawing
def draw(family: Family, n: int, basis, path_base: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G = family.build(n)
    pos, rot_layers = {}, [L for L in family.layer_order if family.layers[L][0] > 0]
    for L in family.layer_order:
        s = family.size(L, n)
        if family.layers[L][0] == 0:
            for i in range(s):
                ang = 2 * math.pi * i / max(s, 1)
                pos[(L, i)] = (0.35 * math.cos(ang) * (s > 1), 0.35 * math.sin(ang) * (s > 1))
        else:
            rad = 1 + (len(rot_layers) - 1 - rot_layers.index(L))
            for i in range(s):
                ang = math.pi / 2 - 2 * math.pi * i / s
                pos[(L, i)] = (rad * math.cos(ang), rad * math.sin(ang))
    bset = set(basis or [])
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color="#555555")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=170,
                           node_color=["#d1495b" if v in bset else "white" for v in G.nodes()],
                           edgecolors="black", linewidths=0.9)
    if G.number_of_nodes() <= 60:
        lbl = {v: f"${v[0]}_{{{v[1] + 1}}}$" for v in G.nodes()}
        def push(p, v):
            r = math.hypot(*p)
            if r < 1e-9:
                return (p[0] + 0.18, p[1] + 0.18)
            d = 0.22 if r > 1.5 else -0.24           # outer layers: label outside; inner: inside
            return (p[0] * (1 + d / r), p[1] * (1 + d / r))
        off = {v: push(p, v) for v, p in pos.items()}
        nx.draw_networkx_labels(G, off, labels=lbl, ax=ax, font_size=7)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- main
def run(family: Family, variants: List[str], n_lo: int, n_hi: int, out: str,
        search_from: int = 8, max_formulas: int = 3, figure_n: Optional[int] = None,
        time_limit: float = 300.0, compile_pdf: bool = True, reuse: bool = False):
    os.makedirs(out, exist_ok=True)
    study = Study(family)
    n_lo = max(n_lo, family.min_n)
    ns = list(range(n_lo, n_hi + 1))
    md = [f"# {family.name}: metric-dimension study", "",
          f"{family.description}  ", f"Layers: {family.layers}  ", f"Edge rules: {family.rules}  ",
          f"n = {n_lo}..{n_hi}. All values are exact (integer linear programming).", ""]
    tex_body, values, first_basis = [], {}, None

    # 1. exact values (reuse values.csv from an earlier run when it covers what is needed)
    cached = {}
    vpath = os.path.join(out, "values.csv")
    if reuse and os.path.exists(vpath):
        with open(vpath) as f:
            for row in csv.DictReader(f):
                for v in variants:
                    if v in row and row[v] != "":
                        cached.setdefault(v, {})[int(row["n"])] = int(row[v])
    for var in variants:
        t0, values[var] = time.time(), {}
        for n in ns:
            if n in cached.get(var, {}):
                values[var][n] = cached[var][n]
            else:
                values[var][n] = study.value(n, var, time_limit=time_limit)[0]
        print(f"[{var}] {VARIANTS[var]}: {[values[var][n] for n in ns]}  ({time.time() - t0:.1f}s)")
    with open(os.path.join(out, "values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n"] + variants)
        for n in ns:
            w.writerow([n] + [values[v][n] for v in variants])

    md += ["## Exact values", "", "| n | " + " | ".join(variants) + " |",
           "|---|" + "---|" * len(variants)]
    md += [f"| {n} | " + " | ".join(str(values[v][n]) for v in variants) + " |" for n in ns]
    md.append("")

    # 2. per variant: pattern, basis formula, tables
    for var in variants:
        vals = values[var]
        per = periodic_tail(vals)
        md += [f"## {var}: {VARIANTS[var]}", ""]
        if per is None:
            cf = guess_closed_form(vals)
            if cf:
                md.append(f"Not constant. Fits ${var}(G_n) = {cf[0]}$ for {cf[1]} <= n <= {n_hi}.")
                tex_body.append(rf"\subsection*{{{VARIANTS[var].capitalize()}}}"
                                rf"Computed values fit $\mathrm{{{var}}}(G_n) = {cf[0]}$ for "
                                rf"${cf[1]} \le n \le {n_hi}$ (conjecture; proof needed).")
            else:
                md.append("No constant or periodic pattern on this range and no simple floor/ceil formula; "
                          "try a larger range of n.")
            md.append("")
            continue
        P, resvals = per
        if P > 1:
            groups = {}
            for r, (v, _) in resvals.items():
                groups.setdefault(v, []).append(r)
            md += ["**Summary:** " + "; ".join(
                f"{var} = {v} for n ≡ {', '.join(map(str, rs))} (mod {P})" for v, rs in sorted(groups.items()))
                + f" (for large n; see each case for the exact start).", ""]
        cases = [((P, r), v, n0) for r, (v, n0) in sorted(resvals.items())]

        for (mod, res), kk, n0 in cases:
            lo = max(search_from, n0)
            cls_txt = ("" if mod == 1 else (" for even n" if mod == 2 and res == 0 else
                       (" for odd n" if mod == 2 else f" for n ≡ {res} (mod {mod})")))
            md += [f"**Result:** {var}(G_n) = {kk}{cls_txt}, {n0} <= n <= {n_hi}.", ""]
            # search the class as a whole, then split into residue classes mod 4 if needed
            found = []
            for M in ([1, 2, 4] if mod == 1 else sorted({mod, 2 * mod, 4} if mod == 2 else {mod, 2 * mod})):
                if M < mod or M % mod:
                    continue
                parts = [(M, r) for r in range(M) if r % mod == res]
                trial = []
                for (MM, rr) in parts:
                    sns = [n for n in ns if n >= lo and n % MM == rr]
                    if len(sns) < 3:
                        trial = None
                        break
                    ranked = ranked_formulas(study, var, sns, kk, step=MM, max_results=max_formulas * 4)
                    if not ranked:
                        trial = None
                        break
                    trial.append(((MM, rr), sns, ranked))
                if trial:
                    found = trial
                    break
            if not found:
                md += ["No basis formula from the template library works for every n in this class; "
                       "a basis exists for each n (see the ILP) but needs a custom pattern.", ""]
                continue
            for (MM, rr), sns, ranked in found:
                ctxt = "all n" if MM == 1 else (f"n ≡ {rr} (mod {MM})")
                md.append(f"Basis formulas for {ctxt} (best first):")
                for v0, b in ranked[:max_formulas]:
                    md.append(f"- $W = {b.tex()}$, resolving for all tested n >= {v0}")
                md.append("")
                v0, best = ranked[0]
                if first_basis is None and var == variants[0]:
                    first_basis = best
                if MM == 1:
                    tabs = best_tables(study, best, max(v0, lo), KIND.get(var, "vertex"), (0, 1))
                elif MM == 2:
                    tabs = best_tables(study, best, max(v0, lo), KIND.get(var, "vertex"), (rr,))
                else:
                    from .tables import symbolic_table
                    t1 = symbolic_table(study, best, rr, max(v0, lo), KIND.get(var, "vertex"), M=MM)
                    if not t1.check_ok:   # pattern in i may need a finer split of n
                        t2s = [symbolic_table(study, best, rr + MM * j, max(v0, lo), KIND.get(var, "vertex"), M=2 * MM)
                               for j in range(2)]
                        tabs = t2s if all(t.check_ok for t in t2s) else [t1]
                    else:
                        tabs = [t1]
                for t in tabs:
                    md += [t.markdown(), ""]
                nthm = "n" if MM == 1 else (rf"n \equiv {rr} \pmod{{{MM}}}")
                tex_body.append(
                    rf"\subsection*{{{VARIANTS[var].capitalize()}" + ("" if MM == 1 else rf" (${nthm}$)") + "}"
                    + "\n" + rf"\noindent\textbf{{Conjecture (verified for ${v0} \le n \le {n_hi}$).}} "
                    rf"For ${nthm}$, $n \ge {v0}$: $\mathrm{{{var}}}(G_n) = {kk}$, and $W = {best.tex()}$ is a basis."
                    + "\n\n" + "\n\n".join(t.latex() for t in tabs if t.check_ok))
                md += ["Numeric check (one n):", "", "```",
                       numeric_table(study, best, sns[0], KIND.get(var, "vertex")), "```", ""]

    # 3. figure
    fn = figure_n or max(n_lo, 8)
    basis = first_basis.vertices(fn) if first_basis else None
    draw(family, fn, basis, os.path.join(out, f"graph_n{fn}"),
         f"{family.name}, n = {fn}" + (" (basis in red)" if basis else ""))
    md += ["## Figure", "", f"![graph](graph_n{fn}.png)", ""]

    md += ["## How to use this in a paper", "",
           "1. Upper bound: the table gives r(v|W) for every vertex/edge; show the vectors are distinct "
           "(the tool has checked this for every n listed).",
           "2. Lower bound: the ILP proves no smaller set works for each n in the range; the general "
           "proof is usually a case analysis on where a set of size k-1 could sit.",
           "3. Small n that behave differently are listed in the values table; treat them separately.",
           "4. Everything here is computational evidence for n in the stated range, not a proof for all n.", ""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(md))

    tex = "\n".join([r"\documentclass[11pt]{article}", r"\usepackage{amsmath,amssymb,graphicx,float}",
                     r"\usepackage[margin=2cm]{geometry}", r"\begin{document}",
                     rf"\section*{{{family.name.replace('_', ' ')}: computed results}}",
                     rf"{family.description}. Exact values for ${n_lo} \le n \le {n_hi}$ by integer linear programming.",
                     rf"\begin{{figure}}[ht]\centering\includegraphics[width=0.5\textwidth]{{graph_n{fn}.pdf}}\end{{figure}}",
                     *tex_body, r"\end{document}"])
    with open(os.path.join(out, "paper_tables.tex"), "w") as f:
        f.write(tex)
    if compile_pdf:
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper_tables.tex"], cwd=out,
                           capture_output=True, timeout=120)
            for ext in ("aux", "log"):
                p = os.path.join(out, f"paper_tables.{ext}")
                if os.path.exists(p):
                    os.remove(p)
        except Exception as e:  # noqa: BLE001
            print("LaTeX compile skipped:", e)
    print(f"Done. Results in {out}/")
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description="Computer-assisted metric dimension study of a graph family.")
    ap.add_argument("family", nargs="?", help="built-in family name (see --list)")
    ap.add_argument("--spec", help="custom family spec, e.g. "
                    "'name=myfam; a:1,0 b:1,0; a-a:1,1 b-b:1,1 a-b:1,0'")
    ap.add_argument("--variants", nargs="+", default=["dim", "edim", "ftdim"], choices=list(VARIANTS))
    ap.add_argument("--n", default="5-30", help="range of n, e.g. 5-30")
    ap.add_argument("--search-from", type=int, default=8, help="smallest n used when searching basis formulas")
    ap.add_argument("--out", default=None, help="output folder (default results/<family>)")
    ap.add_argument("--figure-n", type=int, default=None)
    ap.add_argument("--max-formulas", type=int, default=3)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse values.csv from a previous run (skips the ILPs)")
    ap.add_argument("--list", action="store_true", help="list built-in families")
    a = ap.parse_args(argv)
    if a.list:
        for name, f in FAMILIES.items():
            print(f"{name:18s} {f.description}")
        return
    fam = parse_family_spec(a.spec) if a.spec else FAMILIES[a.family]
    lo, hi = (int(x) for x in a.n.split("-"))
    run(fam, a.variants, lo, hi, a.out or os.path.join("results", fam.name), a.search_from,
        a.max_formulas, a.figure_n, compile_pdf=not a.no_pdf, reuse=a.reuse)


if __name__ == "__main__":
    main()
