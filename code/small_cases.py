"""
Exact mixed metric dimension of P(n,3) for small n by exhaustive search
(independent of the ILP). By rotation, a mixed resolving set may be assumed to
contain u_1 or v_1. For each size k we test every k-set containing u_1 or v_1.
"""
import itertools
import sys

import networkx as nx
import numpy as np

from check_bases import P


def rep_matrix(G):
    nodes = sorted(G.nodes(), key=lambda v: (v[0], v[1]))
    idx = {v: a for a, v in enumerate(nodes)}
    D = dict(nx.all_pairs_shortest_path_length(G))
    Dm = np.array([[D[x][w] for w in nodes] for x in nodes], dtype=np.int16)
    E = np.array([np.minimum(Dm[idx[x]], Dm[idx[y]]) for x, y in G.edges()], dtype=np.int16)
    return nodes, idx, np.vstack([Dm, E])          # elements x vertices


def mdim_exhaustive(n):
    G = P(n)
    nodes, idx, X = rep_matrix(G)
    starts = [idx[("u", 1)], idx[("v", 1)]]
    V = len(nodes)
    for k in range(2, 8):
        for s0 in starts:
            others = [a for a in range(V) if a != s0]
            for rest in itertools.combinations(others, k - 1):
                cols = (s0,) + rest
                R = X[:, cols]
                if len(np.unique(R, axis=0)) == len(R):
                    return k, [nodes[c] for c in cols]
    return None, None


if __name__ == "__main__":
    hi = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for n in range(7, hi + 1):
        k, S = mdim_exhaustive(n)
        print(f"n={n}: mdim={k}  basis={' '.join(f'{a}_{i}' for a, i in S)}", flush=True)
