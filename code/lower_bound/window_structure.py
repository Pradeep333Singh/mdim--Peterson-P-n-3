"""
Certify the 'window structure' of every local constraint type, for n = 6k + p, p in {1,2,5}:

For each type tau and layer L in {u,v}, let M_k(o) = [vertex L_m lies in C_tau(j)] where
o = m - j (mod n). We check that, for every k in [K_LO, K_HI],
  (i)  M_k(o) for o in [-15, 15]         depends only on o            (window at the vertex)
  (ii) M_k(3k + c) for c in [-18, 18]    depends only on c            (window at the antipode)
  (iii) outside these windows M_k is 3-periodic inside each of the two arcs.
Together with the hand argument (distances are F(shorter arc) and F(a+3) = F(a)+1 there),
this gives the windows used in the shrinking lemma for all k >= K_LO.
"""
import sys

import numpy as np

from local_constraints import C_rows, types

NT = len(types())


def patterns(n):
    rows, verts = C_rows(n)
    out = {}
    for t in range(NT):
        block = rows[t * n:(t + 1) * n]
        for L, off in (("u", 0), ("v", n)):
            col = block[:, off]                      # vertex L_1 ; row j-1 -> o = 1 - j
            out[(t, L)] = np.array([bool(col[(1 - o - 1) % n]) for o in range(n)])
    return out


if __name__ == "__main__":
    K_LO, K_HI = int(sys.argv[1]), int(sys.argv[2])
    for p in (1, 2, 5):
        ref0 = ref1 = None
        bad = 0
        for k in range(K_LO, K_HI + 1):
            n = 6 * k + p
            P = patterns(n)
            w0 = {key: tuple(M[o % n] for o in range(-15, 16)) for key, M in P.items()}
            w1 = {key: tuple(M[(3 * k + c) % n] for c in range(-18, 19)) for key, M in P.items()}
            if ref0 is None:
                ref0, ref1 = w0, w1
            if w0 != ref0 or w1 != ref1:
                bad += 1
                print("window pattern differs at k =", k)
            for key, M in P.items():
                arcs = [range(16, 3 * k - 18), range(3 * k + 19, n - 15)]
                for arc in arcs:
                    a = list(arc)
                    for o in a:
                        if o + 3 in arc and M[o] != M[o + 3]:
                            bad += 1
                            print("not periodic", p, k, key, o)
                            break
        print(f"p={p}: k in [{K_LO},{K_HI}] (n {6*K_LO+p}..{6*K_HI+p}): "
              + ("window patterns identical for every k and periodic outside windows" if not bad else f"{bad} problems"),
              flush=True)
