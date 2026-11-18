"""
Computer verification that P(n,3) has NO mixed resolving set of size 4
(so mdim >= 5) for n = 1, 2, 5 (mod 6), n in a given range.
Independent cross-check only; the proof in the paper uses Section 5 and exact searches.

By rotational symmetry, if a mixed resolving 4-set exists, a rotation of it
contains u_1 or v_1. So it suffices to show that the integer program
    minimise sum x_w  s.t. every pair of elements is distinguished, x_{u_1} = 1
(and the same with x_{v_1} = 1) has optimum >= 5.
HiGHS proves optimality (or we ask only whether value <= 4 is feasible).

Usage: python lower_bound_check.py N_LO N_HI
"""
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

import os  # noqa: E402
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, ".."), os.path.join(_here, "..", "..")]   # folder containing mdtool
from mdtool.core import prepare           # noqa: E402
from check_bases import P                  # noqa: E402


def min_with_fixed(n, fixed):
    G = P(n)
    inst = prepare(G, "mdim")
    rows = np.unique(inst.rows, axis=0)
    V = len(inst.nodes)
    A = sparse.csr_matrix(rows.astype(np.int8))
    lb = np.zeros(V)
    lb[inst.index[fixed]] = 1
    # feasibility of a set of size <= 4 containing `fixed`
    cons = [LinearConstraint(A, lb=1, ub=np.inf), LinearConstraint(np.ones((1, V)), lb=0, ub=4)]
    res = milp(c=np.zeros(V), constraints=cons, integrality=np.ones(V),
               bounds=Bounds(lb, np.ones(V)), options={"time_limit": 3600})
    return res.status   # 0 = a 4-set exists, 2 = infeasible (no 4-set), other = unknown


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    for n in range(lo, hi + 1):
        if n % 6 not in (1, 2, 5):
            continue
        t = time.time()
        st = [min_with_fixed(n, ("u", 1)), min_with_fixed(n, ("v", 1))]
        verdict = "no mixed resolving 4-set (mdim >= 5)" if st == [2, 2] else f"STATUS {st}"
        print(f"n={n}: {verdict}  ({time.time() - t:.0f}s)", flush=True)
