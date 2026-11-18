"""
Validation:  python -m mdtool.validate

1. Known published values (exact ILP must reproduce them).
2. Brute force from the textbook definitions on random small graphs,
   written independently of the covering-matrix formulation in core.py.
"""
from __future__ import annotations

import itertools
import math
import random

import networkx as nx

from .core import Study, prepare, solve
from .families import FAMILIES

KNOWN = [
    # (family, variant, formula(n), n range, source)
    ("cycle", "dim", lambda n: 2, range(3, 31), "folklore"),
    ("cycle", "edim", lambda n: 2, range(3, 31), "Kelenc, Tratnik, Yero (2018)"),
    ("cycle", "mdim", lambda n: 3, range(4, 31), "Kelenc, Kuziak, Taranenko, Yero (2017)"),
    ("cycle", "ftdim", lambda n: 3, range(5, 31), "Hernando et al. (2008); C_4 needs 4"),
    ("cycle", "sdim", lambda n: math.ceil(n / 2), range(3, 31), "Oellermann, Peters-Fransen (2007)"),
    ("prism", "dim", lambda n: 2 if n % 2 else 3, range(3, 31), "Caceres et al. (2007)"),
    ("antiprism", "dim", lambda n: 3, range(5, 31), "Javaid, Rahim, Ali (2008)"),
    ("petersen2", "dim", lambda n: 3, range(5, 31), "Javaid, Rahim, Ali (2008)"),
    ("wheel", "dim", lambda n: (2 * n + 2) // 5, range(7, 31), "Buczkowski et al. (2003)"),
    ("wheel", "edim", lambda n: n - 1, range(5, 31), "Kelenc, Tratnik, Yero (2018), Prop. 5"),
]


# ------------------------------------------------ definitions, written naively
def _d(G):
    return dict(nx.all_pairs_shortest_path_length(G))


def _rep_v(D, S, x):
    return tuple(D[x][w] for w in S)


def _rep_e(D, S, e):
    return tuple(min(D[e[0]][w], D[e[1]][w]) for w in S)


def naive_ok(G, D, S, variant):
    V, E = list(G.nodes()), list(G.edges())
    if variant in ("dim", "ldim"):
        pairs = itertools.combinations(V, 2) if variant == "dim" else E
        return all(_rep_v(D, S, x) != _rep_v(D, S, y) for x, y in pairs)
    if variant == "edim":
        reps = [_rep_e(D, S, e) for e in E]
        return len(set(reps)) == len(reps)
    if variant == "mdim":
        reps = [_rep_v(D, S, v) for v in V] + [_rep_e(D, S, e) for e in E]
        return len(set(reps)) == len(reps)
    if variant == "ftdim":   # definition: S - {v} resolving for every v in S
        return all(naive_ok(G, D, [w for w in S if w != v], "dim") for v in S) and len(S) >= 2
    if variant == "sdim":    # w strongly resolves u,v: u on a shortest v-w path or v on a shortest u-w path
        for u, v in itertools.combinations(V, 2):
            if not any(D[v][w] == D[v][u] + D[u][w] or D[u][w] == D[u][v] + D[v][w] for w in S):
                return False
        return True
    raise ValueError(variant)


def naive_value(G, variant):
    D, V = _d(G), sorted(G.nodes())
    for k in range(1, len(V) + 1):
        for S in itertools.combinations(V, k):
            if naive_ok(G, D, list(S), variant):
                return k
    return len(V)


def main():
    fails = 0
    print("1) Published values")
    for fam, var, f, rng, src in KNOWN:
        st = Study(FAMILIES[fam])
        bad = [n for n in rng if st.value(n, var)[0] != f(n)]
        fails += bool(bad)
        print(f"   {'OK  ' if not bad else 'FAIL'} {fam:10s} {var:6s} n={rng.start}..{rng.stop - 1}  [{src}]"
              + (f"  mismatch at n={bad}" if bad else ""))

    print("2) Brute force from definitions on random connected graphs")
    rnd = random.Random(7)
    tested = 0
    for trial in range(60):
        nv = rnd.randint(5, 9)
        G = nx.gnp_random_graph(nv, rnd.uniform(0.25, 0.6), seed=rnd.randint(0, 10 ** 6))
        if not nx.is_connected(G):
            continue
        for var in ("dim", "edim", "mdim", "ldim", "ftdim", "sdim"):
            a = solve(prepare(G, var))[0]
            b = naive_value(G, var)
            tested += 1
            if a != b:
                fails += 1
                print(f"   FAIL {var} on graph {sorted(G.edges())}: ILP {a}, brute force {b}")
    print(f"   {tested} graph/variant checks, {'all agree' if not fails else 'see failures above'}")
    print("ALL PASSED" if not fails else f"{fails} FAILURES")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
