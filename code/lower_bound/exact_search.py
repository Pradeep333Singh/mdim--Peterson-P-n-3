"""
Exact (integer, exhaustive) proof that no 4-set satisfies the local family L in P(n,3).

Hitting-set search: S must meet every row C_tau(j). Fix the first vertex (u_1 or v_1, by
rotation). Repeatedly pick an uncovered row with few candidate vertices (fewest among at most 4000 examined) and branch
on those vertices. Depth is at most 3 more vertices. Rows are bitsets over vertices; the
covered-rows test uses Python big integers (exact). Every branch is explored, so a
'none found' answer is an exhaustive proof.
Usage: python exact_search.py n [n ...]
"""
import sys
import time

from local_constraints import C_rows


def build(n, **family):
    rows, verts = C_rows(n, **family)
    # deduplicate rows; represent each row as an int bitmask over vertices
    uniq = {}
    for r in rows:
        mask = 0
        for a in r.nonzero()[0]:
            mask |= 1 << int(a)
        uniq[mask] = True
    row_masks = list(uniq)
    V = len(verts)
    # for each vertex, the set (bitmask over rows) of rows it covers
    cover = [0] * V
    for i, m in enumerate(row_masks):
        mm = m
        while mm:
            low = mm & -mm
            a = low.bit_length() - 1
            cover[a] |= 1 << i
            mm ^= low
    return row_masks, cover, verts


def search(row_masks, cover, start, depth_left, covered, all_rows):
    if covered == all_rows:
        return True
    if depth_left == 0:
        return False
    # choose an uncovered row with few candidates (fewest among at most 4000 examined); completeness does not depend on this choice
    unc = all_rows & ~covered
    best, best_cnt = None, None
    mm = unc
    checked = 0
    while mm and checked < 4000:
        low = mm & -mm
        i = low.bit_length() - 1
        c = bin(row_masks[i]).count("1")
        if best is None or c < best_cnt:
            best, best_cnt = i, c
        mm ^= low
        checked += 1
    cand = row_masks[best]
    while cand:
        low = cand & -cand
        a = low.bit_length() - 1
        if search(row_masks, cover, start, depth_left - 1, covered | cover[a], all_rows):
            return True
        cand ^= low
    return False


def no_four_set(n, **family):
    row_masks, cover, verts = build(n, **family)
    all_rows = (1 << len(row_masks)) - 1
    for first in (("u", 1), ("v", 1)):
        a = verts.index(first)
        if search(row_masks, cover, a, 3, cover[a], all_rows):
            return False
    return True


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        t = time.time()
        ok = no_four_set(n)
        print(f"n={n}: {'no 4-set satisfies L (exhaustive, exact)' if ok else 'A 4-SET EXISTS'}  ({time.time() - t:.0f}s)",
              flush=True)
