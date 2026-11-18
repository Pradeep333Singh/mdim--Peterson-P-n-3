# Code for "The mixed metric dimension of the generalized Petersen graph P(n,3)"

Requirements: Python 3.10+, `networkx`, `numpy`, `scipy` (HiGHS solver), plus the `mdtool`
package (folder next to this one; it provides the table fitting and the ILP set-up).

```
pip install networkx numpy scipy
# mdtool must sit in this folder's parent or grandparent (the scripts add both to sys.path),
# or be on PYTHONPATH
python verify_all.py              # all checks except the long ILP range (~10 min)
python verify_all.py --exact 301  # also the exact search for mixed 4-sets up to n = 301 (~35 min)
python verify_all.py --ilp 120    # independent HiGHS cross-check of the lower bound (~1 hour)
python verify_all.py --lower      # also the proof for n = 1, 2, 5 (mod 6) (~70 min)
```

| file | what it establishes (paper reference) |
|---|---|
| `distances.py` | closed-form distances, compared with BFS for 8 <= n <= 200 (Lemma 3.2) |
| `check_bases.py` | W0, W3, W4, W* are mixed generators, BFS, every admissible n <= 3000 |
| `small_cases.py` | exact mdim and a basis for 7 <= n <= 30 by exhaustive search (Table 2) |
| `tables_p3.py` | representation tables from the distance formula (Appendix A) |
| `distinct.py` | exact proof that table rows never coincide, for all k (Lemma 4.2, Table 2) |
| `tables_allk.py` | tables equal the distance formula for all k (Section 6.4) |
| `exact_mixed.py` | independent exact exhaustive search (does not use Section 5): a mixed generator of size <= 4 exists iff n = 0,3,4 (mod 6), for 19 <= n <= 301; log in `exact_mixed_log.txt` |
| `negative_controls.py` | sanity checks: the bases fail on the wrong residue classes; 3-sets fail |
| `lower_bound_check.py` | further HiGHS cross-check of the lower bound for n <= 120 (not used in the proofs) |
| `build_tex_parts.py` | writes `output/appendix_tables.tex` and `output/distinct_stats.json` (compare with the copies in `tex/` and here) |
| `lower_bound_log.txt`, `small_cases_log.txt`, `distinct_stats.json` | outputs of the runs used in the paper |
| `lower_bound/local_constraints.py` | the local family L (160 constraint types) (Section 5.1) |
| `lower_bound/irregular_check.py` | window values identical for k = 12..30, p = 1,2,5; irregular set bound (Lemmas 5.1, 5.2) |
| `lower_bound/exact_search.py` | exact exhaustive search: no 4-set satisfies L, n = 1,2,5 mod 6, 19..236 (Prop. 5.5); log `exact_search_log.txt`. `exact_search_log_v3_L63.txt` is the earlier run with only the 63 incident vertex-edge and edge-edge types (n = 2,5 mod 6, 23..230), quoted in the remark of Section 5 |
| `lower_bound/shrink_exhaustive.py` | exhaustive single-vertex test of the shrinking lemma, every block position (Lemma 5.3) |
| `lower_bound/test_shrink.py` | test of the shrinking lemma over every admissible block, with a control on non-admissible blocks (Lemma 5.3) |
| `lower_bound/remark_check.py` | reproduces the facts quoted in the remark at the end of Section 5 (log `remark_check_log.txt`) |
