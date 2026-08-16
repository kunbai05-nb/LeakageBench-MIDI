# Family-aware splitting

Random file splitting assumes files are independent, but same-work families violate that assumption. Assign complete family components atomically to prevent known component crossings while matching target file/token ratios.

For family size (k), train probability (p_train), test probability (p_test), and all other splits combined as (p_other):

```text
P_cross(k) = 1 - (1-p_train)^k - (1-p_test)^k + p_other^k
```

Conditioned on one designated test member:

```text
P(train sibling | one test member, k) = 1 - (1-p_train)^(k-1)
```

Both increase with family size. Project reports use the bounded statement **zero known cross-split family overlap under the adopted family definition**; unknown family relations may remain.
