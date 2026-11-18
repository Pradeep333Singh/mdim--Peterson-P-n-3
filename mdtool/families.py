"""
Graph families for metric-dimension studies.

A *rotational family* is described by layers and edge rules, which covers
cycles, prisms, antiprisms, generalized Petersen graphs, circular ladders,
wheels and most convex-polytope families studied in the literature.

Layer spec:   name -> (m, c)      the layer has m*n + c vertices, indexed 0..size-1
              e.g. "a": (1, 0)  -> n vertices a_1..a_n
                   "b": (2, 0)  -> 2n vertices
                   "h": (0, 1)  -> a single hub vertex

Edge rule:    (la, lb, mult, off)
              for every i in layer la:  la_i  ~  lb_{(mult*i + off) mod |lb|}
              (0-based internally; vertices are printed 1-based, so a_i ~ a_{i+1})
              e.g. ("a","a",1,1)  outer cycle
                   ("a","b",1,0)  spokes a_i - b_i
                   ("b","b",1,2)  inner star polygon (P(n,2))
                   ("a","h",0,0)  every a_i joined to hub h

You can also write any family by hand as a Python function returning a
networkx graph whose nodes are (layer, index) tuples (index 0-based).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import networkx as nx

EdgeRule = Tuple[str, str, int, int]


@dataclass
class Family:
    name: str
    layers: Dict[str, Tuple[int, int]]          # name -> (m, c)
    rules: List[EdgeRule]
    min_n: int = 3
    description: str = ""
    builder: Callable[[int], nx.Graph] | None = None   # optional custom builder
    layer_order: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_order:
            self.layer_order = list(self.layers)

    def size(self, layer: str, n: int) -> int:
        m, c = self.layers[layer]
        return m * n + c

    def build(self, n: int) -> nx.Graph:
        if self.builder is not None:
            return self.builder(n)
        G = nx.Graph()
        for L in self.layer_order:
            for i in range(self.size(L, n)):
                G.add_node((L, i))
        for la, lb, mult, off in self.rules:
            sb = self.size(lb, n)
            for i in range(self.size(la, n)):
                u, v = (la, i), (lb, (mult * i + off) % sb)
                if u != v:
                    G.add_edge(u, v)
        if not nx.is_connected(G):
            raise ValueError(f"{self.name} is disconnected for n={n}")
        return G

    def edge_types(self, n: int):
        """Edges grouped by the rule that created them: {rule: [(i, edge), ...]}.
        Used for edge-representation tables. Duplicate edges keep their first rule."""
        seen, out = set(), {}
        for rule in self.rules:
            la, lb, mult, off = rule
            sb = self.size(lb, n)
            lst = []
            for i in range(self.size(la, n)):
                u, v = (la, i), (lb, (mult * i + off) % sb)
                if u == v:
                    continue
                key = frozenset((u, v))
                if key in seen:
                    continue
                seen.add(key)
                lst.append((i, (u, v)))
            out[rule] = lst
        return out

    def rotation_is_automorphism(self, n: int) -> bool:
        """Check that i -> i+m (per layer) is an automorphism, so a basis may be
        assumed WLOG to contain a vertex with a small index."""
        if self.builder is not None:
            return False
        G = self.build(n)

        def rot(v):
            L, i = v
            m, _ = self.layers[L]
            return (L, (i + m) % self.size(L, n)) if m else v

        return all(G.has_edge(rot(u), rot(v)) for u, v in G.edges())


def label(v) -> str:
    """LaTeX label for a vertex, 1-based: ('a', 0) -> a_{1}."""
    L, i = v
    return f"{L}_{{{i + 1}}}"


def rule_label(rule: EdgeRule) -> str:
    """LaTeX label for an edge type, e.g. ('a','a',1,1) -> a_i a_{i+1}."""
    la, lb, mult, off = rule
    if mult == 0:
        rhs = f"{lb}_{{{off + 1}}}" if off else f"{lb}"
    else:
        idx = "i" if mult == 1 else f"{mult}i"
        # 0-based target mult*i+off with i = j-1  ->  1-based index mult*j + (off + 1 - mult)
        off_adj = off + 1 - mult
        rhs = f"{lb}_{{{idx}{'+' + str(off_adj) if off_adj > 0 else ('-' + str(-off_adj) if off_adj < 0 else '')}}}"
    return f"{la}_i{rhs}"


# ---------------------------------------------------------------------------
# Built-in families (several have known metric dimension, used for validation)
# ---------------------------------------------------------------------------
FAMILIES: Dict[str, Family] = {}


def register(f: Family) -> Family:
    FAMILIES[f.name] = f
    return f


register(Family("cycle", {"a": (1, 0)}, [("a", "a", 1, 1)], min_n=3,
                description="Cycle C_n"))

register(Family("prism", {"a": (1, 0), "b": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 1), ("a", "b", 1, 0)], min_n=3,
                description="Prism D_n = C_n x P_2"))

register(Family("antiprism", {"a": (1, 0), "b": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 1), ("a", "b", 1, 0), ("a", "b", 1, 1)],
                min_n=3, description="Antiprism A_n"))

register(Family("petersen2", {"a": (1, 0), "b": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 2), ("a", "b", 1, 0)], min_n=5,
                description="Generalized Petersen graph P(n,2)"))

register(Family("wheel", {"a": (1, 0), "h": (0, 1)},
                [("a", "a", 1, 1), ("a", "h", 0, 0)], min_n=3,
                description="Wheel W_n = C_n + K_1"))

# Convex polytope R_n (Imran et al.): two n-cycles and a middle 2n... here the
# common 'prism with pendant layer' style used in many papers: three n-layers.
register(Family("triple_prism", {"a": (1, 0), "b": (1, 0), "c": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 1), ("c", "c", 1, 1),
                 ("a", "b", 1, 0), ("b", "c", 1, 0)], min_n=3,
                description="Three concentric n-cycles joined by spokes (C_n x P_3)"))

register(Family("antiprism_spokes", {"a": (1, 0), "b": (1, 0), "c": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 1), ("c", "c", 1, 1),
                 ("a", "b", 1, 0), ("a", "b", 1, 1), ("b", "c", 1, 0)], min_n=3,
                description="Antiprism with an extra outer n-cycle attached by spokes"))


def parse_family_spec(text: str) -> Family:
    """Build a family from a one-line spec, e.g.
    'name=myfam; a:1,0 b:2,0; a-a:1,1 b-b:1,1 a-b:2,0 a-b:2,1'
    """
    parts = [p.strip() for p in text.split(";")]
    name = parts[0].split("=", 1)[1].strip() if parts[0].startswith("name=") else "custom"
    layer_part, rule_part = parts[-2], parts[-1]
    layers = {}
    for tok in layer_part.split():
        L, mc = tok.split(":")
        m, c = (int(x) for x in mc.split(","))
        layers[L] = (m, c)
    rules = []
    for tok in rule_part.split():
        ab, mo = tok.split(":")
        la, lb = ab.split("-")
        mult, off = (int(x) for x in mo.split(","))
        rules.append((la, lb, mult, off))
    return Family(name, layers, rules, min_n=3, description="custom spec")

register(Family("petersen3", {"a": (1, 0), "b": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 3), ("a", "b", 1, 0)], min_n=7,
                description="Generalized Petersen graph P(n,3)"))

register(Family("petersen4", {"a": (1, 0), "b": (1, 0)},
                [("a", "a", 1, 1), ("b", "b", 1, 4), ("a", "b", 1, 0)], min_n=9,
                description="Generalized Petersen graph P(n,4)"))
