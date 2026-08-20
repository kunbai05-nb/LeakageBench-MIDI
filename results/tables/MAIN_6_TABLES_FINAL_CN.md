# LeakageBench-MIDI 六张正文表：作者审阅版

# 表1 Dataset leakage landscape：数据集与泄漏图景

**研究问题：** LMD 的 family structure 如何转化为随机文件划分中的同作品暴露，以及为什么简单去重仍不等于 family-level isolation。

### Panel A：Family structure

| Statistic | Value |
| --- | --- |
| Frozen downstream identities | 178,561 |
| Total family components | 140,427 |
| Multi-member families | 20,797 |
| Files in multi-member families | 58,931 |
| Largest family | 66 |

### Panel B：Random split leakage

| Split | Test-family contamination | Test-file contamination | Repeats |
| --- | --- | --- | --- |
| 80/10/10 | 27.66% | 29.77% | 1,000 |
| 90/5/5 | 30.30% | — | 1,000 |
| 90/10 | 29.34% | — | 1,000 |

### Panel C：Why simple dedup is insufficient

| Treatment | Files retained | Known cross-split families | Key implication |
| --- | --- | --- | --- |
| File-level split | 4,300 | 177 | Substantial same-work exposure remains |
| Exact/canonical/track-order dedup | 4,263 | 161 | Reduced, but not eliminated |
| Family-level isolation | 4,300 | 0 | Known overlap eliminated |

*脚注：178,561 是 frozen downstream identity universe，不是 official raw LMD count。177 → 161 → 0 表示 known cross-split families；零表示 adopted family definition 下的 zero KNOWN cross-split overlap。*

---

# 表2 Evaluation inflation and causal controls：评测膨胀与替代解释控制

**研究问题：** 主效应是否能在不同结构定义下保持，并且不是由 generic extra-data exposure 或广义泛化改进单独解释。

### Panel A：Legacy confirmatory + normalized robustness

| Analysis / Protocol | Contrast | Signed effect | Relative effect | 95% CI | Families / Seeds | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| Initial Transformer-L confirmatory | family-exposed − clean | -0.11532 | 13.62% | [-0.13244, -0.09896] | 100 / 3 | 初始确认性证据 |
| Normalized structural robustness | normalized family-exposed − clean | -0.11485 | 13.47% | [-0.13946, -0.09208] | 50 / — | 稳健性 |

### Panel B：Phase-2 causal control

| Condition / contrast | Frozen mean NLL | Signed effect | Relative effect | 95% CI | Reading |
| --- | --- | --- | --- | --- | --- |
| Clean | 0.7977 | — | — | — | 参考条件 |
| Unrelated donor | 0.7909 | — | — | — | 对照条件 |
| Same-family donor | 0.5889 | — | — | — | 主条件 |
| **Same − Unrelated** | — | **-0.20199** | **25.54%** | [-0.22872, -0.17671] | **主结果** |
| Same − Clean | — | -0.20876 | 26.17% | [-0.23631, -0.18249] | 对照 |
| Unrelated − Clean | — | -0.00677 | 0.85% | [-0.01275, -0.00229] | 对照 |

### Panel C：Generalization control

| Analysis / Protocol | Signed effect | Relative effect | 95% CI | Reading |
| --- | --- | --- | --- | --- |
| Family-disjoint clean-validation | 0.00106 | — | [-0.00085, 0.00302] | 泛化控制 |

*脚注：Phase-2 signed contrast = same-family − unrelated；负值表示 family-specific exposure 下 receiver NLL 更低。NLL lower 不等于音乐质量提高。*

---

# 表3 Model dependence：模型容量与架构/范式依赖

**研究问题：** family-specific effect 是否随 Transformer 容量或 model paradigm 改变。不同模型的 raw objective 不横向比较。

### Panel A：Transformer capacity

| Model | Signed effect | Relative effect | 95% CI |
| --- | --- | --- | --- |
| Transformer-S | -0.07759 | 8.57% | [-0.08954, -0.06633] |
| Transformer-M | -0.08571 | 10.40% | [-0.09882, -0.07293] |
| Transformer-L | -0.11532 | 13.62% | [-0.13265, -0.09911] |
| Capacity trend | -0.07847 | — | [-0.12124, -0.03741] |

### Panel B：Architecture / paradigm

| Model | Protocol / estimand | Signed effect | Relative effect | 95% CI | Reading |
| --- | --- | --- | --- | --- | --- |
| Transformer-L legacy | Legacy; family-exposed − clean | -0.11532 | 13.62% | [-0.13244, -0.09896] | 初始确认性 |
| TCN | Legacy; family-exposed − clean | -0.03172 | 2.53% | [-0.03778, -0.02635] | 弱同方向支持 |
| Transformer-L Phase-2 | Phase-2; same-family − unrelated | -0.20199 | 25.54% | [-0.22872, -0.17671] | **主结果** |
| Conditional VAE | Phase-2; same-family − unrelated | -0.00905 | — | [-0.01436, -0.00400] | 小但可靠 |
| Latent Diffusion | Phase-2; same-family − unrelated | -0.00000 | — | [-0.00002, 0.00001] | 未检测到显著效应 |

*脚注：capacity trend 仅表示 tested Transformer family 内的趋势。lockfile 未提供冻结 Params 字段，因此本版不推断参数量。 Transformer-L 的 legacy 与 Phase-2 CI 来自不同 result_id，按各自 estimand 保留。*

---

# 表4 Generated-music consequences：生成音乐后果

**研究问题：** family exposure 是否改变生成分布与 receiver 的对齐，以及是否造成 broad surface-statistical drift。

### Panel A：Receiver alignment

| Metric | Unrelated | Same-family | Δ | 95% CI | Holm p / status |
| --- | --- | --- | --- | --- | --- |
| Pitch-class JSD | 0.2226 | 0.1941 | -0.02854 | [-0.03926, -0.01846] | p < .001 / p = 0.0168 |
| Pitch-interval JSD | 0.4527 | 0.4255 | -0.02721 | [-0.03671, -0.01801] | p < .001 / p = 0.0168 |
| IOI Wasserstein | 0.5654 | 0.4890 | -0.07641 | [-0.14687, -0.02007] | p = 0.0028 / p = 0.2788；次要/非确认性 |

### Panel B：Representative surface controls

| Metric | Unrelated | Same-family | Δ | 95% CI | Reading |
| --- | --- | --- | --- | --- | --- |
| Note density | 19.6237 | 19.7046 | 0.08099 | [-0.22604, 0.38135] | 代表性 null |
| Polyphony | 3.4567 | 3.5105 | 0.05386 | [-0.01814, 0.12857] | 代表性 null |
| Pitch range | 35.1133 | 35.4020 | 0.28867 | [-0.28935, 0.90467] | 代表性 null |
| Qualified-note ratio | 0.9885 | 0.9893 | 0.00086 | [-0.00162, 0.00300] | 代表性 null |
| Onset density | 0.3976 | 0.3970 | -0.00060 | [-0.00836, 0.00760] | 代表性 null |

*脚注：负的 JSD effect 表示生成分布更接近 held-out receiver。IOI 为次要、非确认性结果；copying/extraction 已从主文移除，保留在 supplementary。*

---

# 表5 Mechanism diagnostics：机制诊断

**研究问题：** family-specific effect 是否在 shared region 中更强，以及 relatedness 是否呈现稳定 association。

### Panel A：Token localization（common paired family support）

| Scale | Shared effect | Nonshared effect | Shared − Nonshared | 95% CI | p | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| 4-event | -0.15528 | -0.03713 | -0.11814 | [-0.22714, -0.02907] | p = 0.0064 | shared localization supported |
| 8-event | -0.16390 | -0.16658 | 0.00268 | [-0.11069, 0.17190] | p = 0.9209 | no significant differential |
| 16-event | -0.16823 | -0.17483 | 0.00660 | [-0.08951, 0.16462] | p = 0.9821 | no significant differential |

### Panel B：Relatedness

| Metric | Estimate | 95% CI | p | Reading |
| --- | --- | --- | --- | --- |
| Spearman rho | 0.40907 | [0.21086, 0.57844] | — | moderate positive association |
| Robust Huber standardized coefficient | 0.02707 | [0.00893, 0.04190] | — | adjusted positive association |

*脚注：Panel A 的 shared、nonshared 与 differential 均来自 common paired family support。Relatedness 仅作 association 解读。*

---

# 表6 Mitigation and external evidence：泄漏缓解与外部证据

**研究问题：** family-aware split 的数据代价，以及 incomplete/noisy family inference 下的 practical robustness。

### Panel A：Mitigation trade-off（4,300-file cohort）

| Method | Files retained | Token loss | Known cross-split families | Reassignment | Reading |
| --- | --- | --- | --- | --- | --- |
| S0 File-level split | 4,300 | 0.00% | 177 | 0.00% | 文件划分基线 |
| S1 Dedup | 4,263 | 0.80% | 161 | 0.00% | 简单去重仍有残余 |
| **S2 Family-aware split** | **4,300** | **0.00%** | **0** | **25.95%** | **采用方案** |
| Delete-all multi-family baseline | 3,300 | 22.84% | — | — | 反事实高代价基线 |

### Panel B1：Incomplete relation graph（FULL-UNIVERSE SIMULATION）

| Condition | Residual known cross-split families | Empirical interval | Simulated adopted-edge pairwise recall | Reading |
| --- | --- | --- | --- | --- |
| File-level baseline | 5033.34 | [4932.38, 5120.52] | — | baseline |
| Perfect reference graph | 0.00 | — | 1.0000 | upper bound |
| simulated recall 0.95 | 343.92 | [311.43, 376] | 0.9234 | high-recall region |
| simulated recall 0.90 | 675.26 | [635.48, 716.62] | 0.8494 | degraded |
| simulated recall 0.50 | 2880.01 | [2799.85, 2948.15] | 0.3585 | further degraded |

### Panel B2：False-positive relation stress test（FULL-UNIVERSE SIMULATION）

| FP injection | Relation precision | Over-merged components | Split-ratio error | Reading |
| --- | --- | --- | --- | --- |
| 1% | 98.31% | 381.77 | 0.0000011 | conservative practical region |
| 5% | 91.44% | 1921.82 | 0.0000011 | false-positive stress test |

*脚注：Panel A 是 4,300-file cohort；Panel B1/B2 是 full-universe simulation。reference graph != complete real-world ground truth；simulated adopted-edge recall != real detector recall；conservative practical region 只是 simulation-derived recommendation，不是 real detector recall 的声明。*
