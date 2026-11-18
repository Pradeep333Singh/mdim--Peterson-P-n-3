"""
Test of the shrinking lemma (Lemma 5.3), over EVERY admissible block position.

For n = 6k+p (p = 1, 2, 5) and k in K_LIST, and many 4-sets S (random, clustered, antipodal,
u and v at the same index), for every b such that B = {b,b+1,b+2} and B' = B - 3k avoid
Z(S): delete B and B', relabel, and check that for every type tau and every base
j outside B u B', S covers (tau, j) in P(n,3) iff S' covers (tau, image of j) in P(n-6,3).
Control: blocks that DO meet Z(S) must break this correspondence in some cases.
Usage: python test_shrink.py [seed]
"""
import random
import sys

import numpy as np

from local_constraints import C_rows, types

NT = len(types())


def Z(S, n, k, p):
    z = set()
    for _, m in S:
        for c in range(-15, 6):
            z.add((m + c - 1) % n + 1)
        for c in range(-15 - p, 7):
            z.add((m - 3 * k + c - 1) % n + 1)
    return z


def cover(rows, verts, S, n):
    cols = [verts.index(w) for w in S]
    return rows[:, cols].any(axis=1).reshape(NT, n)       # [type, base-1]


def compare(rows_n, verts_n, rows_m, verts_m, n, k, S, b):
    B = [(b + i - 1) % n + 1 for i in range(3)]
    Bp = [(x - 3 * k - 1) % n + 1 for x in B]
    rem = set(B) | set(Bp)
    keep = [j for j in range(1, n + 1) if j not in rem]
    mp = {j: a + 1 for a, j in enumerate(keep)}
    S2 = [(L, mp[m]) for L, m in S]
    c1 = cover(rows_n, verts_n, S, n)
    c2 = cover(rows_m, verts_m, S2, n - 6)
    j_old = np.array(keep) - 1
    j_new = np.array([mp[j] for j in keep]) - 1
    return np.array_equal(c1[:, j_old], c2[:, j_new])


def sets(n, k, rnd, count):
    out = []
    for _ in range(count):
        kind = rnd.randrange(4)
        if kind == 0:
            S = [(rnd.choice("uv"), rnd.randint(1, n)) for _ in range(4)]
        elif kind == 1:
            m = rnd.randint(1, n)
            S = [(rnd.choice("uv"), (m + rnd.randint(0, 12) - 1) % n + 1) for _ in range(4)]
        elif kind == 2:
            m = rnd.randint(1, n)
            S = [(rnd.choice("uv"), (m + rnd.randint(-3, 3) + (3 * k if t % 2 else 0) - 1) % n + 1) for t in range(4)]
        else:
            m = rnd.randint(1, n)
            S = [("u", m), ("v", m), (rnd.choice("uv"), rnd.randint(1, n)), (rnd.choice("uv"), rnd.randint(1, n))]
        if len(set(S)) == 4:
            out.append(S)
    return out


if __name__ == "__main__":
    rnd = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    K_LIST = [13, 14, 15, 17, 20]
    tests = fails = ctrl_tests = ctrl_breaks = 0
    for p in (1, 2, 5):
        for k in K_LIST:
            n = 6 * k + p
            rows_n, verts_n = C_rows(n)
            rows_m, verts_m = C_rows(n - 6)
            for S in sets(n, k, rnd, 25):
                z = Z(S, n, k, p)
                for b in range(1, n + 1):
                    B = {(b + i - 1) % n + 1 for i in range(3)}
                    Bp = {(x - 3 * k - 1) % n + 1 for x in B}
                    if any(m in B | Bp for _, m in S):
                        continue
                    ok = compare(rows_n, verts_n, rows_m, verts_m, n, k, S, b)
                    if not (B | Bp) & z:
                        tests += 1
                        if not ok:
                            fails += 1
                            print("MISMATCH", p, k, S, b)
                    else:
                        ctrl_tests += 1
                        ctrl_breaks += not ok
    print(f"admissible blocks: {tests} tests, {fails} mismatches; "
          f"control (blocks meeting Z(S)): {ctrl_breaks}/{ctrl_tests} break the correspondence")
    print("shrinking lemma test passed" if fails == 0 and ctrl_breaks > 0 else "SHRINK TEST FAILED")
