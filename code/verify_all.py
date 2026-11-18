"""
Re-run every computation the paper relies on and print a summary.

    python verify_all.py            (about 15 minutes, without the long ILP range)
    python verify_all.py --exact 301 (also the exact search for mixed 4-sets, 7 <= n <= 301; ~35 min)
    python verify_all.py --ilp 120  (independent HiGHS cross-check of the lower bound up to n = 120; ~1 hour)
    python verify_all.py --lower    (also the proof for n = 1, 2, 5 mod 6: windows + exact search; ~70 min)
"""
import subprocess
import sys
import time

STEPS = [
    ("Distance formula (Lemma 3.2) vs BFS, 8 <= n <= 200", ["python3", "distances.py"]),
    ("Generators W0, W3, W4, W* by BFS, all admissible n <= 3000", ["python3", "check_bases.py", "3000"]),
    ("Exact values and bases for 7 <= n <= 30 (exhaustive search)", ["python3", "small_cases.py", "30"]),
    ("Tables valid for all k (Section 6.4)", ["python3", "tables_allk.py"]),
    ("Tables + exact distinctness certificates (Table 2)", ["python3", "build_tex_parts.py"]),
    ("Negative controls (bases fail on the wrong residues; 3-sets fail)", ["python3", "negative_controls.py"]),
    ("Exact search for mixed 4-sets, 7 <= n <= 40 (independent check, quick part)",
     ["python3", "exact_mixed.py"] + [str(n) for n in range(7, 41)]),
]

if __name__ == "__main__":
    if "--lower" in sys.argv:   # proof of the lower bound for n = 1, 2, 5 (mod 6), Section 5 (~70 min)
        STEPS += [("Shrinking lemma, single vertex, every block position (Lemma 5.3)",
                   ["python3", "lower_bound/shrink_exhaustive.py"]),
                  ("Shrinking lemma, sampled 4-sets, every admissible block (Lemma 5.3)",
                   ["python3", "lower_bound/test_shrink.py", "1"]),
                  ("Window patterns and irregular set, k = 12..30 (Lemmas 5.1, 5.2)",
                   ["python3", "lower_bound/irregular_check.py", "12", "30"]),
                  ("Controls: 4-sets satisfying L exist where mdim = 4",
                   ["python3", "lower_bound/exact_search.py", "24", "27", "28", "30", "34", "40"]),
                  ("No 4-set satisfies L, n = 1,2,5 mod 6, 19 <= n <= 236 (Proposition 5.5, exact)",
                   ["python3", "lower_bound/exact_search.py"] + [str(n) for n in range(19, 237) if n % 6 in (1, 2, 5)])]
    if "--exact" in sys.argv:
        hi = int(sys.argv[sys.argv.index("--exact") + 1])
        STEPS.append((f"Exact search for mixed 4-sets, 41 <= n <= {hi} (independent check of Theorem 1.1)",
                      ["python3", "exact_mixed.py"] + [str(n) for n in range(41, hi + 1)]))
    if "--ilp" in sys.argv:
        hi = sys.argv[sys.argv.index("--ilp") + 1]
        STEPS.append((f"No mixed 4-set for n = 1,2,5 mod 6, n <= {hi} (HiGHS cross-check)",
                      ["python3", "lower_bound_check.py", "7", hi]))
    for title, cmd in STEPS:
        t = time.time()
        print(f"\n=== {title}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip()[-3000:])
        if r.returncode:
            print("FAILED:", r.stderr[-2000:])
            sys.exit(1)
        print(f"--- done in {time.time() - t:.0f}s")
    print("\nALL CHECKS COMPLETED")
