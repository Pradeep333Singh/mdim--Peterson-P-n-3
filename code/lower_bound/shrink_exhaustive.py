"""
Exhaustive test of the shrinking lemma (Lemma 5.3) for a single vertex.

The proof of Lemma 5.3 treats each vertex w of S separately, so it is enough to test one
vertex: for w = u_1 and w = v_1, every k in K_LIST, p in {1,2,5}, and EVERY block position b
with B, B' disjoint from Z({w}): delete the blocks and compare, for all 160 types and all
remaining bases, whether w lies in C_tau(j) before and after. Control: blocks meeting Z({w})
(but not containing w) must break the correspondence for some b.
Usage: python shrink_exhaustive.py
"""
from local_constraints import C_rows
from test_shrink import Z, compare

if __name__ == "__main__":
    K_LIST = [13, 14, 15, 20]
    bad = 0
    for p in (1, 2, 5):
        for k in K_LIST:
            n = 6 * k + p
            rows_n, verts_n = C_rows(n)
            rows_m, verts_m = C_rows(n - 6)
            for w in (("u", 1), ("v", 1)):
                z = Z([w], n, k, p)
                adm = fails = ctrl = breaks = 0
                for b in range(1, n + 1):
                    B = {(b + i - 1) % n + 1 for i in range(3)}
                    Bp = {(x - 3 * k - 1) % n + 1 for x in B}
                    if w[1] in B | Bp:
                        continue
                    ok = compare(rows_n, verts_n, rows_m, verts_m, n, k, [w], b)
                    if not (B | Bp) & z:
                        adm += 1
                        fails += not ok
                    else:
                        ctrl += 1
                        breaks += not ok
                bad += fails > 0 or breaks == 0 or adm != n - 2 * (24 + p)
                print(f"p={p} k={k} w={w[0]}_1: admissible b: {adm} (= n - 2(24+p)), mismatches {fails}; "
                      f"non-admissible b: {breaks}/{ctrl} break the correspondence", flush=True)
    print("exhaustive single-vertex shrinking test passed" if not bad else "SHRINK TEST FAILED")
