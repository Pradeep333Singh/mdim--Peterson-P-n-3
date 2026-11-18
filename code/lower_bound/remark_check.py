"""
Checks for the remark at the end of Section 5 (not needed for the proofs).
 (1) For n = 1 (mod 6), n = 37..61, 4-sets satisfying only the 63 incident vertex-edge and
     edge-edge types exist; we count them (sets containing u_1 or v_1), check that none is a
     mixed generator, and list the kinds and index gaps of the pairs they fail to separate.
 (2) With the vertex-vertex and non-incident vertex-edge types restricted to delta <= 5,
     4-sets satisfying all types still exist for these n.
Usage: python remark_check.py
"""
import sys
from collections import Counter
from itertools import combinations

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from exact_mixed import graph                 # noqa: E402
from exact_search import build, no_four_set   # noqa: E402


def all_sets(n, **family):
    row_masks, cover, verts = build(n, **family)
    full = (1 << len(row_masks)) - 1
    out = set()

    def rec(S, covered):
        if covered == full:
            out.add(tuple(sorted(S)))
            return
        if len(S) == 4:
            return
        unc = full & ~covered
        i = (unc & -unc).bit_length() - 1
        cand = row_masks[i]
        while cand:
            low = cand & -cand
            a = low.bit_length() - 1
            rec(S + [a], covered | cover[a])
            cand ^= low

    for first in (("u", 1), ("v", 1)):
        a = verts.index(first)
        rec([a], cover[a])
    res = set()
    for s in out:
        for extra in combinations([w for w in range(len(verts)) if w not in s], 4 - len(s)):
            res.add(tuple(sorted(s + extra)))
    return res, verts


if __name__ == "__main__":
    for n in (37, 43, 49, 55, 61):
        V, idx, X = graph(n)
        base = [i for _, i in V] + [i for i in range(1, n + 1) for _ in range(3)]
        sets, verts = all_sets(n, kinds=("VE", "EE"))
        kinds, gaps, gens = Counter(), Counter(), 0
        for s in sets:
            R = X[:, [idx[verts[a]] for a in s]]
            _, inv, cnt = np.unique(R, axis=0, return_inverse=True, return_counts=True)
            inv = inv.ravel()
            gens += not (cnt > 1).any()
            for g in np.flatnonzero(cnt > 1):
                for a, b in combinations(np.flatnonzero(inv == g), 2):
                    kinds["".join(sorted(("V" if a < 2 * n else "E") + ("V" if b < 2 * n else "E")))] += 1
                    d = (base[a] - base[b]) % n
                    gaps[min(d, n - d)] += 1
        short = not no_four_set(n, RV=5)
        print(f"n={n}: {len(sets)} 4-sets satisfy the 63 types; mixed generators among them: {gens}; "
              f"unseparated pairs by kind {dict(kinds)}, by index gap {dict(sorted(gaps.items()))}; "
              f"a 4-set satisfies the family with delta <= 5: {short}", flush=True)
