"""
Negative controls for the distinctness certificates (Section 6.3).
Tables are built for sets that are known NOT to be mixed generators; the certificate
procedure must report collisions, in agreement with direct computation by BFS.
"""
import tables_p3
from check_bases import P, W0, W4, is_mixed_resolving
from distinct import check_table, feasible_rank_le1
from tables_p3 import GROUPS, make_table

ORIG = tables_p3.W_of


def W3set(n):
    return [("u", 1), ("u", n // 2 + 1), ("v", 3)]


CASES = [("W0 used for n = 6k+1 (mdim = 5 there)", W0, 1),
         ("W4 used for n = 6k+5 (mdim = 5 there)", W4, 5),
         ("a 3-set for n = 6k", W3set, 0)]

if __name__ == "__main__":
    ok = True
    for name, f, p in CASES:
        tables_p3.W_of = lambda b, n, f=f: f(n)
        ks, fit = make_table("x", p)
        st, low = check_table(fit, 4, GROUPS)
        found = len(st["collisions"]) + sum(1 for same, Ra, Rb, ga, gb in low
                                            if feasible_rank_le1(Ra, Rb, same, 4)[0] is True)
        bfs_bad = [6 * k + p for k in range(4, 15) if not is_mixed_resolving(P(6 * k + p), f(6 * k + p))]
        good = found > 0 and len(bfs_bad) == 11
        ok &= good
        print(f"{name}: certificate procedure finds {found} colliding row pairs; "
              f"BFS: not a mixed generator for {len(bfs_bad)}/11 tested n  -> {'OK' if good else 'PROBLEM'}")
    tables_p3.W_of = ORIG
    print("negative controls passed" if ok else "NEGATIVE CONTROLS FAILED")
