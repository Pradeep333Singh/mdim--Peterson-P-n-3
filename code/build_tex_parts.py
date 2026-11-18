"""Writes appendix_tables.tex and distinct_stats.json for the paper."""
import json
from collections import Counter

from tables_p3 import GROUPS, make_table, latex_table
from distinct import check_table, feasible_rank_le1

PLAN = [  # (basis, p, label, caption prefix, k_min used in the theorem)
    ("W0", 0, "tab:W0", r"Mixed representations for $n\equiv 0 \pmod 6$.", 4),
    ("W3", 3, "tab:W3", r"Mixed representations for $n\equiv 3 \pmod 6$.", 4),
    ("W4", 4, "tab:W4", r"Mixed representations for $n\equiv 4 \pmod 6$.", 4),
    ("W5", 1, "tab:W51", r"Representations with respect to the $5$-set, $n\equiv 1 \pmod 6$.", 4),
    ("W5", 2, "tab:W52", r"Representations with respect to the $5$-set, $n\equiv 2 \pmod 6$.", 4),
    ("W5", 5, "tab:W55", r"Representations with respect to the $5$-set, $n\equiv 5 \pmod 6$.", 4),
]

if __name__ == "__main__":
    parts, stats = [], {}
    for basis, p, label, cap, kmin in PLAN:
        ks, fit = make_table(basis, p)
        tex = latex_table(fit, basis, p, ks, label, cap)
        tex = tex.replace("_iu_{i+1}", "_i u_{i+1}").replace("_iv_i", "_i v_i").replace("_iv_{i+3}", "_i v_{i+3}")
        parts.append(tex)
        st, low = check_table(fit, kmin, GROUPS)
        c = Counter()
        for same, Ra, Rb, ga, gb in low:
            v, why = feasible_rank_le1(Ra, Rb, same, kmin)
            if v is not False:
                raise RuntimeError(f"no certificate for a pair of rows: {basis}, p={p}: {why}")
            c[why] += 1
        if st["collisions"]:
            raise RuntimeError(f"collision found in table {basis}, p={p}")
        rows = sum(len(v) for v in fit.values())
        stats[f"{basis}_{p}"] = {"rows": rows, "pairs": st["pairs"], "rank2_exact": st["rank2"],
                                 "rank_le1": dict(c)}
        print(basis, p, stats[f"{basis}_{p}"])
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "output")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "appendix_tables.tex"), "w").write("\n\\clearpage\n".join(parts) + "\n")
    json.dump(stats, open(os.path.join(out, "distinct_stats.json"), "w"), indent=1)
    print("wrote output/appendix_tables.tex and output/distinct_stats.json "
          "(compare with ../tex/appendix_tables.tex and distinct_stats.json)")
