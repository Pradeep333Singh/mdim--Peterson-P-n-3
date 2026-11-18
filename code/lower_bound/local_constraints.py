"""
The local constraint family L for P(n,3), and tools for the shrinking argument.

Base j; elements at base j:
  vertices u_j, v_j ; edges a_j = u_j u_{j+1}, b_j = u_j v_j, c_j = v_j v_{j+3}.
L consists of
  VE: the six 'vertex vs incident edge' conditions at base j (Lemma 2.1):
      some w closer to u_{j+1} than u_j; closer to u_j than u_{j+1};
      closer to v_j than u_j; closer to u_j than v_j;
      closer to v_{j+3} than v_j; closer to v_j than v_{j+3}.
  EE: edges e_{j+d1} and f_{j+d2} distinguished, e, f in {a, b, c}, min(d1,d2) = 0, max <= R.
  VV: vertices X_j and Y_{j+delta} distinguished, 0 <= delta <= R.
  VN: vertex X_{j+d1} and a NON-incident edge e_{j+d2} distinguished, min(d1,d2) = 0, max <= R.
A set S satisfies L iff for every base j and every type, S meets the set C_type(j).
With R = 6 there are 6 + 57 + 25 + 72 = 160 types; all endpoints lie in [j, j+R+3].
"""
import numpy as np

import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from distances import d

R_DEFAULT = 6


def elem_dist(n, elem, Y, m):
    """distance from vertex Y_m to element (vertex ('V',X,i) or edge ('E',X,i,Y2,i2))."""
    if elem[0] == "V":
        _, X, i = elem
        return d(n, X, i, Y, m)
    _, X, i, X2, i2 = elem
    return min(d(n, X, i, Y, m), d(n, X2, i2, Y, m))


def edge(kind, j, n):
    M = lambda t: (t - 1) % n + 1
    if kind == "a":
        return ("E", "u", M(j), "u", M(j + 1))
    if kind == "b":
        return ("E", "u", M(j), "v", M(j))
    return ("E", "v", M(j), "v", M(j + 3))


def types(R=R_DEFAULT, RV=None, kinds=("VE", "EE", "VV", "VN")):
    """List of constraint types; each is a function (n, j) -> pair of 'distance callables'."""
    T = []
    # VE as strict-closer conditions: (target vertex b, other vertex a) meaning d(w,b) < d(w,a)
    VE = [(("u", 1), ("u", 0)), (("u", 0), ("u", 1)), (("v", 0), ("u", 0)),
          (("u", 0), ("v", 0)), (("v", 3), ("v", 0)), (("v", 0), ("v", 3))]
    for b, a in VE:
        T.append(("VE", b, a))
    seen = set()
    for e in "abc":
        for f in "abc":
            for dl in range(-R, R + 1):
                if dl == 0 and e == f:
                    continue
                key = tuple(sorted([(e, 0), (f, dl)]))
                # normalise so that the pair is translation-canonical
                base = min(key[0][1], key[1][1])
                key = tuple((x, o - base) for x, o in key)
                if key in seen:
                    continue
                seen.add(key)
                T.append(("EE",) + key)
    RV = R if RV is None else RV
    # vertex-vertex types: X_j vs Y_{j+dl}, 0 <= dl <= RV
    for dl in range(0, RV + 1):
        for X in "uv":
            for Y in "uv":
                if dl == 0 and X >= Y:
                    continue
                T.append(("VV", (X, 0), (Y, dl)))
    # vertex vs non-incident edge: X_{j+d1} vs e_{j+d2}, min(d1, d2) = 0, max <= R
    for X in "uv":
        for e in "abc":
            for dl in range(-RV, RV + 1):     # edge base minus vertex index
                inc = {("u", "a"): (0, -1), ("u", "b"): (0,), ("v", "b"): (0,), ("v", "c"): (0, -3)}.get((X, e), ())
                if dl in inc:
                    continue
                d1, d2 = (0, dl) if dl >= 0 else (-dl, 0)
                T.append(("VN", (X, d1), (e, d2)))
    return [t for t in T if t[0] in kinds]


def C_rows(n, R=R_DEFAULT, RV=None, kinds=("VE", "EE", "VV", "VN")):
    """Boolean matrix: rows = (type, base j), columns = vertices (u_1..u_n, v_1..v_n)."""
    verts = [("u", m) for m in range(1, n + 1)] + [("v", m) for m in range(1, n + 1)]
    rows = []
    for t in types(R, RV, kinds):
        for j in range(1, n + 1):
            if t[0] == "VE":
                (B, ob), (A, oa) = t[1], t[2]
                bi, ai = (j + ob - 1) % n + 1, (j + oa - 1) % n + 1
                rows.append([d(n, B, bi, Y, m) < d(n, A, ai, Y, m) for Y, m in verts])
            elif t[0] == "VV":
                (X1, o1), (X2, o2) = t[1], t[2]
                E1, E2 = ("V", X1, (j + o1 - 1) % n + 1), ("V", X2, (j + o2 - 1) % n + 1)
                rows.append([elem_dist(n, E1, Y, m) != elem_dist(n, E2, Y, m) for Y, m in verts])
            elif t[0] == "VN":
                (X1, o1), (e, oe) = t[1], t[2]
                E1, E2 = ("V", X1, (j + o1 - 1) % n + 1), edge(e, j + oe, n)
                rows.append([elem_dist(n, E1, Y, m) != elem_dist(n, E2, Y, m) for Y, m in verts])
            else:
                (e, oe), (f, of) = t[1], t[2]
                E1, E2 = edge(e, j + oe, n), edge(f, j + of, n)
                rows.append([elem_dist(n, E1, Y, m) != elem_dist(n, E2, Y, m) for Y, m in verts])
    return np.array(rows, dtype=bool), verts
