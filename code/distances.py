"""
Closed-form distances in P(n,3)  (outer cycle u_i, inner vertices v_i, i in Z_n,
edges u_i u_{i+1}, u_i v_i, v_i v_{i+3}).

Infinite version (i in Z), offset t >= 0, t = 3q + r, r in {0,1,2}:
    F_uu(t) = min(t, q + 2 + r)
    F_uv(t) = q + 1 + r
    F_vv(t) = q + (0, 3, 4)[r]
Finite graph: d_n(X_i, Y_j) = min_{m in Z} F_XY(|j - i + m n|)
            = min(F_XY(t), F_XY(n - t)),  t = (j - i) mod n,
the second form proved for n >= 19 and checked below for every 8 <= n <= 200
(it fails only at n = 7, where a third lift is shorter).

Run:  python distances.py      (verifies against BFS for 7 <= n <= 200)
"""
import networkx as nx


def F(X, Y, t):
    t = abs(t)
    q, r = divmod(t, 3)
    if X == Y == "u":
        return min(t, q + 2 + r)
    if X != Y:
        return q + 1 + r
    return q + (0, 3, 4)[r]


def d(n, X, i, Y, j):
    t = (j - i) % n
    return min(F(X, Y, t), F(X, Y, n - t))


def P(n):
    G = nx.Graph()
    for i in range(n):
        G.add_edge(("u", i), ("u", (i + 1) % n))
        G.add_edge(("u", i), ("v", i))
        G.add_edge(("v", i), ("v", (i + 3) % n))
    return G


if __name__ == "__main__":
    G7, bad7 = P(7), []
    for X in "uv":
        for (Y, j), val in nx.single_source_shortest_path_length(G7, (X, 0)).items():
            if d(7, X, 0, Y, j) != val:
                bad7.append((X, 0, Y, j, val, d(7, X, 0, Y, j)))
    print(f"n=7: formula (4) fails for {len(bad7)} pairs from u_0/v_0, e.g. {bad7[:2]} (expected: fails)")
    bad = 0
    for n in range(8, 201):
        G = P(n)
        for X in "uv":  # rotation symmetry: sources u_0 and v_0 suffice
            dist = nx.single_source_shortest_path_length(G, (X, 0))
            for (Y, j), val in dist.items():
                if d(n, X, 0, Y, j) != val:
                    bad += 1
                    if bad < 10:
                        print("mismatch", n, X, Y, j, val, d(n, X, 0, Y, j))
    print("distance formula verified for all 8 <= n <= 200" if not bad else f"{bad} mismatches")
