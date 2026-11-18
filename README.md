# Mixed metric dimension of the generalized Petersen graph P(n,3)

Code and computation logs accompanying the paper

> P. Singh, *The mixed metric dimension of the generalized Petersen graph P(n,3)*, submitted.

The paper proves that, for n ≥ 19, mdim(P(n,3)) = 4 when n ≡ 0, 3, 4 (mod 6) and
mdim(P(n,3)) = 5 when n ≡ 1, 2, 5 (mod 6), and gives the values for 7 ≤ n ≤ 18.
Several steps of the proof are exact computations; this repository contains the programs
for all of them and the logs of the runs reported in the paper.

## Contents

- `code/` — verification programs and logs; `code/README.md` maps each script to the
  lemma, proposition or table it establishes.
- `code/lower_bound/` — the computations for the lower bound (Section 5 of the paper).
- `mdtool/` — a small package for metric-dimension computations that the scripts import.

## Running the checks

Requirements: Python 3.10 or later with `numpy`, `scipy` and `networkx`
(the runs in the paper used Python 3.13, NumPy 2.5, SciPy 1.18, NetworkX 3.7).

```
pip install -r requirements.txt
cd code
python verify_all.py            # main checks (~10 min)
python verify_all.py --lower    # also the lower bound of Section 5 (~70 min)
python verify_all.py --exact 301   # independent exhaustive search up to n = 301 (~35 min)
```

All computations that the proofs rely on use exact integer or rational arithmetic.

## License

MIT, see `LICENSE`.
