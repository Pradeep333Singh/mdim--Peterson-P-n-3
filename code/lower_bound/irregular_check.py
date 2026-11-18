"""Certify the window patterns (Lemma 5.1) and the irregular-set bound (eq. irregular)
for every k in [K_LO, K_HI], p in {1,2,5}."""
import sys
from window_structure import patterns
K_LO, K_HI = int(sys.argv[1]), int(sys.argv[2])
for p in (1, 2, 5):
    ref = None; problems = 0
    for k in range(K_LO, K_HI + 1):
        n = 6 * k + p
        P = patterns(n)
        w = {key: (tuple(M[o % n] for o in range(-15, 16)), tuple(M[(3*k + c) % n] for c in range(-18, 19)),
                   tuple(M[(n - c) % n] for c in (16, 17, 18)), tuple(M[c] for c in (16, 17, 18)))
             for key, M in P.items()}
        if ref is None: ref = w
        if w != ref:
            problems += 1; print("window values differ at", p, k)
        irr = set()
        for M in P.values():
            for o in range(n):
                if M[o] != M[(o + 3) % n] or M[o] != M[(o - 3) % n]:
                    irr.add(o)
        allowed = {(c % n) for c in range(-5, 13)} | {(3*k + c) % n for c in range(-5, 16)}
        if not irr <= allowed:
            problems += 1; print("irregular offsets outside bound at", p, k, sorted(irr - allowed)[:10])
    print(f"p={p}: k in [{K_LO},{K_HI}]: " + ("windows identical for all k; irregular set within [-5,12] U [3k-5,3k+15]" if not problems else f"{problems} PROBLEMS"), flush=True)
