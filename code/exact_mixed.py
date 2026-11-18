"""
Exact proof that P(n,3) has no mixed metric generator of size 4 (Proposition 4.6).

Elements: the 2n vertices and 3n edges. A set S is a mixed generator iff every pair of
distinct elements is separated by some w in S (different distance to w).

Method (exhaustive, integer arithmetic only, no solver):
  1. BFS distances; element-to-vertex distance matrix X (edge distance = min over ends).
  2. Constraint subset: the H pairs separated by the FEWEST vertices (hardest pairs).
  3. Depth-first search over 4-sets containing u_1 or v_1 (enough by rotation): repeatedly
     take an uncovered subset-pair with fewest separators and branch on all its separators.
     This enumerates every 4-set that satisfies the subset (exhaustive).
  4. Every such 4-set is then tested against ALL pairs directly (are the 5n distance
     vectors pairwise distinct?).
A 4-set that is a mixed generator satisfies the subset, so it would be found in step 3 and
accepted in step 4. Hence 'none' is a proof.

Usage: python exact_mixed.py n [n ...]
"""
import sys
import time

from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def graph(n):
    V = [("u", i) for i in range(1, n + 1)] + [("v", i) for i in range(1, n + 1)]
    idx = {v: a for a, v in enumerate(V)}
    m = lambda i: (i - 1) % n + 1
    E = []
    for i in range(1, n + 1):
        E += [(("u", i), ("u", m(i + 1))), (("u", i), ("v", i)), (("v", i), ("v", m(i + 3)))]
    r = [idx[a] for a, b in E] + [idx[b] for a, b in E]
    c = [idx[b] for a, b in E] + [idx[a] for a, b in E]
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(2 * n, 2 * n))
    D = shortest_path(A, unweighted=True, directed=False).astype(np.int16)
    X = np.vstack([D, np.array([np.minimum(D[idx[a]], D[idx[b]]) for a, b in E])])
    return V, idx, X


def hard_pairs(X, H):
    """Rows (bool over vertices) of the H pairs with the fewest separators."""
    N = len(X)
    keep_rows, keep_cnt = [], []
    for i in range(N - 1):
        R = X[i] != X[i + 1:]
        cnt = R.sum(axis=1)
        sel = np.argsort(cnt, kind="stable")[:H]
        keep_rows.append(R[sel])
        keep_cnt.append(cnt[sel])
        if sum(len(c) for c in keep_cnt) > 4 * H:          # periodically trim
            rows = np.vstack(keep_rows)
            cnts = np.concatenate(keep_cnt)
            o = np.argsort(cnts, kind="stable")[:H]
            keep_rows, keep_cnt = [rows[o]], [cnts[o]]
    rows = np.vstack(keep_rows)
    cnts = np.concatenate(keep_cnt)
    o = np.argsort(cnts, kind="stable")[:H]
    return rows[o], cnts[o]


def search(M, cnt, S, depth, out):
    unc = ~M[:, S].any(axis=1)
    if not unc.any():
        out.append(tuple(S))
        return
    if depth == 0:
        return
    if depth == 1:
        cand = np.flatnonzero(M[unc].all(axis=0))
        out.extend(tuple(S) + (int(w),) for w in cand)
        return
    ui = np.flatnonzero(unc)
    r = ui[np.argmin(cnt[ui])]
    for w in np.flatnonzero(M[r]):
        search(M, cnt, S + [int(w)], depth - 1, out)


def is_mixed(X, S):
    R = X[:, list(S)]
    return len(np.unique(R, axis=0)) == len(R)


def no_mixed_4set(n, H=20000):
    V, idx, X = graph(n)
    M, cnt = hard_pairs(X, H)
    survivors = []
    for first in (("u", 1), ("v", 1)):
        search(M, cnt, [idx[first]], 3, survivors)
    survivors = {tuple(sorted(s)) for s in survivors}
    # A set of size < 4 that separates all hard pairs stops the search early, so its
    # supersets were not enumerated: extend it to every 4-set containing it.
    # (Being a mixed generator is monotone, so testing the 4-supersets suffices.)
    short = [s for s in survivors if len(s) < 4]
    cands = {s for s in survivors if len(s) == 4}
    for s in short:
        rest = [w for w in range(len(V)) if w not in s]
        for extra in combinations(rest, 4 - len(s)):
            cands.add(tuple(sorted(s + extra)))
    gens = [s for s in cands if is_mixed(X, s)]
    return gens, len(survivors), len(short), [[V[a] for a in s] for s in gens[:1]]


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        t = time.time()
        gens, nsurv, nshort, ex = no_mixed_4set(n)
        msg = ("no mixed metric generator of size <= 4 (exhaustive, exact)" if not gens
               else f"mixed generator of size <= 4 EXISTS, e.g. {ex[0]}")
        print(f"n={n}: {msg}; sets separating the hard-pair subset: {nsurv} ({nshort} of size < 4, extended)  ({time.time() - t:.0f}s)", flush=True)
