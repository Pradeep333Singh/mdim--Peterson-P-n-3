"""
Exact computation of metric-dimension variants.

Every variant is reduced to one covering problem:

    rows[p, w] = True  if vertex w resolves pair p (in the variant's sense)
    S is feasible  <=>  every pair p has at least r vertices of S with rows[p, w]

  variant   pairs                         'w resolves (x, y)'                         r
  dim       all vertex pairs              d(x,w) != d(y,w)                            1
  edim      all edge pairs                d(e,w) != d(f,w), d(e,w)=min over ends      1
  mdim      all pairs of vertices+edges   same, on the mixed element set              1
  ldim      adjacent vertex pairs         d(x,w) != d(y,w)                            1
  ftdim     all vertex pairs              d(x,w) != d(y,w)                            2
  ftedim    all edge pairs                d(e,w) != d(f,w)                            2
  sdim      all vertex pairs              x on a shortest y-w path or y on an x-w one 1

ftdim uses the standard equivalence: S is fault-tolerant resolving (S - {v}
resolving for every v in S) iff every pair is resolved by >= 2 vertices of S.

The minimum feasible |S| is found exactly with an integer linear program
(SciPy / HiGHS).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.csgraph import shortest_path

VARIANTS = {
    "dim": "metric dimension",
    "edim": "edge metric dimension",
    "mdim": "mixed metric dimension",
    "ldim": "local metric dimension",
    "ftdim": "fault-tolerant metric dimension",
    "ftedim": "fault-tolerant edge metric dimension",
    "sdim": "strong metric dimension",
}


@dataclass
class Instance:
    """A graph prepared for one variant."""
    G: nx.Graph
    nodes: List                 # column order (vertices that may be chosen)
    index: Dict                 # vertex -> column
    D: np.ndarray               # all-pairs distances (vertices x vertices)
    elements: List              # things being distinguished (vertices / edges)
    X: np.ndarray               # element-to-vertex distances (elements x vertices)
    pairs: np.ndarray           # (P, 2) element index pairs
    rows: np.ndarray            # (P, V) bool resolution matrix
    r: int                      # required multiplicity


def distance_matrix(G: nx.Graph):
    nodes = sorted(G.nodes())
    index = {v: k for k, v in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")
    D = shortest_path(A, unweighted=True, directed=False).astype(np.int32)
    return nodes, index, D


def _element_distances(G, nodes, index, D, kind):
    if kind == "vertex":
        return list(nodes), D
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    E = np.array([np.minimum(D[index[u]], D[index[v]]) for u, v in edges], dtype=np.int32)
    if kind == "edge":
        return edges, E
    return list(nodes) + edges, np.vstack([D, E])           # mixed


def _all_pairs(N):
    iu, ju = np.triu_indices(N, k=1)
    return np.stack([iu, ju], axis=1)


def prepare(G: nx.Graph, variant: str) -> Instance:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {list(VARIANTS)}")
    nodes, index, D = distance_matrix(G)
    kind = {"edim": "edge", "ftedim": "edge", "mdim": "mixed"}.get(variant, "vertex")
    elements, X = _element_distances(G, nodes, index, D, kind)
    r = 2 if variant.startswith("ft") else 1

    if variant == "ldim":
        pairs = np.array(sorted((min(index[u], index[v]), max(index[u], index[v]))
                                for u, v in G.edges()), dtype=np.int64)
    else:
        pairs = _all_pairs(len(elements))

    if variant == "sdim":
        u, v = pairs[:, 0], pairs[:, 1]
        duv = D[u, v][:, None]
        rows = (D[u] == duv + D[v]) | (D[v] == duv + D[u])
    else:
        rows = X[pairs[:, 0]] != X[pairs[:, 1]]
    return Instance(G, nodes, index, D, elements, X, pairs, rows, r)


def is_resolving(inst: Instance, S: Sequence) -> Tuple[bool, Optional[Tuple]]:
    """Check S (vertices) against the variant. Returns (ok, first unresolved pair)."""
    cols = [inst.index[v] for v in S]
    if not cols:
        return False, None
    hits = inst.rows[:, cols].sum(axis=1)
    bad = np.flatnonzero(hits < inst.r)
    if bad.size == 0:
        return True, None
    p = inst.pairs[bad[0]]
    return False, (inst.elements[p[0]], inst.elements[p[1]])


def solve(inst: Instance, time_limit: float = 300.0, fix: Sequence = ()) -> Tuple[int, List]:
    """Exact minimum resolving set via ILP. `fix` forces vertices into S."""
    rows = np.unique(inst.rows, axis=0)          # identical constraints collapse
    V = len(inst.nodes)
    A = sparse.csr_matrix(rows.astype(np.int8))
    lb_x = np.zeros(V)
    for v in fix:
        lb_x[inst.index[v]] = 1
    res = milp(c=np.ones(V),
               constraints=LinearConstraint(A, lb=inst.r, ub=np.inf),
               integrality=np.ones(V),
               bounds=Bounds(lb_x, np.ones(V)),
               options={"time_limit": time_limit, "disp": False})
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"ILP not solved to optimality (status {res.status}: {res.message})")
    S = [inst.nodes[k] for k in np.flatnonzero(res.x > 0.5)]
    ok, _ = is_resolving(inst, S)
    assert ok, "solver returned an infeasible set"
    return len(S), S


class Study:
    """Caches instances per (n, variant) for one family."""

    def __init__(self, family):
        self.family = family
        self._graphs: Dict[int, nx.Graph] = {}
        self._inst: Dict[Tuple[int, str], Instance] = {}

    def graph(self, n):
        if n not in self._graphs:
            self._graphs[n] = self.family.build(n)
        return self._graphs[n]

    def instance(self, n, variant):
        key = (n, variant)
        if key not in self._inst:
            self._inst[key] = prepare(self.graph(n), variant)
        return self._inst[key]

    def value(self, n, variant, time_limit=300.0):
        return solve(self.instance(n, variant), time_limit=time_limit)
