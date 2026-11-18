"""
Direct check (from the definition, by BFS) that the bases used in the paper are
mixed resolving sets of P(n,3), for every n in the stated class up to N_MAX.

Vertices: ('u', i), ('v', i) with i = 1..n  (indices mod n).
Mixed resolving: all vertices AND edges have distinct distance vectors to W,
where d(w, xy) = min(d(w, x), d(w, y)).
"""
import sys
import networkx as nx


def P(n):
    G = nx.Graph()
    m = lambda i: (i - 1) % n + 1
    for i in range(1, n + 1):
        G.add_edge(("u", i), ("u", m(i + 1)))
        G.add_edge(("u", i), ("v", i))
        G.add_edge(("v", i), ("v", m(i + 3)))
    return G


def W5(n):            # all n >= 12
    return [("u", n // 2 - 1), ("u", (n + 1) // 2 + 2), ("v", 1), ("v", 2), ("v", n)]


def W0(n):            # n = 0 mod 6, n >= 12
    return [("u", 1), ("u", n // 2 + 1), ("v", 3), ("v", 5)]


def W3(n):            # n = 3 mod 6, n >= 15
    return [("u", 1), ("u", (n - 3) // 2), ("v", (n + 5) // 2), ("v", n - 3)]


def W4(n):            # n = 4 mod 6, n >= 10
    return [("u", n // 2 - 2), ("v", 1), ("v", 2), ("v", n - 3)]


def is_mixed_resolving(G, W):
    dist = [nx.single_source_shortest_path_length(G, w) for w in W]
    seen = set()
    for x in G.nodes():
        r = tuple(D[x] for D in dist)
        if r in seen:
            return False
        seen.add(r)
    for x, y in G.edges():
        r = tuple(min(D[x], D[y]) for D in dist)
        if r in seen:
            return False
        seen.add(r)
    return True


CASES = [("W5 (all n)", W5, None, 12), ("W0", W0, 0, 12), ("W3", W3, 3, 15), ("W4", W4, 4, 10)]

if __name__ == "__main__":
    N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    for name, f, res, n0 in CASES:
        bad = [n for n in range(n0, N_MAX + 1)
               if (res is None or n % 6 == res) and not is_mixed_resolving(P(n), f(n))]
        print(f"{name:12s} n0={n0}: {'OK for every n up to ' + str(N_MAX) if not bad else 'FAILS at ' + str(bad[:10])}")
