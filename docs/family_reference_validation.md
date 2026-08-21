# Family-reference validation

The frozen family graph uses the CAugBERT 0.99 / CLaMP 0.99 conservative union.

| Check | Result |
|---|---:|
| Upstream benchmark precision | 0.9133 |
| Upstream benchmark recall | 0.2830 |
| Upstream benchmark F1 | 0.4322 |
| Symbolic same-family advantage across 100 experimental pairs | 0.9208 |
| 95% bootstrap interval | [0.8781, 0.9578] |
| Positive-family fraction | 100% |
| Structurally non-exact families | 95 |
| Structurally non-exact advantage | 0.9166 |
| Graph-noise simulation rows | 2,900 |

The symbolic check used 10,000 paired-family bootstrap draws with seed 20260818.
No completed independent human labels are included in the frozen study.
