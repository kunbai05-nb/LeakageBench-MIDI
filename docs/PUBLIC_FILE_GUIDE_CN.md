# LeakageBench-MIDI 公开仓库逐文件导览

本文档解释 GitHub 公开版中每个文件的职责，面向审稿人、复现研究者和后续维护者。这里的“复现”分为四类：

- **软件复现**：验证 family graph、泄漏审计、component-aware split 等核心实现。
- **合成流程复现**：使用仓库现场生成的合成 MIDI 验证端到端流程。
- **统计强复现**：从匿名的 family/seed 级分析单元重新计算论文数值、置信区间和稳健性结果。
- **冻结产物校验**：验证论文最终表格、图和公开结果导出的哈希与一致性。

本仓库不包含原始 MIDI、token 序列、训练/测试私有清单、模型权重或内部服务器证据，因此不声称仅凭 GitHub 可以从原始语料重新训练全部模型。公开版的核心强声明是：在不暴露受限数据的前提下，可以独立重算论文的公开数值结果。

## 1. 仓库根目录与自动化

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `.github/workflows/reproducibility.yml` | GitHub Actions 自动化定义。在每次 push、pull request 或手动触发时，创建干净的 Ubuntu/Python 3.10.12 环境，安装锁定依赖，运行一键强复现，并上传逐字段审计产物。它把本地“能跑”升级为托管环境持续验证。 | CI、维护者、审稿人查看绿色构建状态。 |
| `.gitignore` | 阻止缓存、虚拟环境、原始 MIDI、音频、checkpoint、日志、归档和合成生成物进入版本控制；同时显式允许安全的 `clean_test_nll_rows.csv` 匿名标量文件。 | 发布安全边界。 |
| `.zenodo.json` | Zenodo GitHub 集成元数据，使用官方支持的字段保存正式标题、v1.1.1 版本、作者、ORCID、单位、许可证和关键词；Zenodo 会在 release 归档时分配 DOI。 | GitHub–Zenodo 长期归档。 |
| `CHANGELOG.md` | 记录公开 release 的版本变化、兼容性变化和公开材料增减，帮助引用者确认自己使用的是哪一版。 | 版本追踪。 |
| `CITATION.cff` | 机器可读引用元数据，GitHub 可据此生成 Cite this repository 信息。 | 论文引用与软件归档。 |
| `LICENSE` | MIT 软件许可证；只覆盖本仓库代码和文档，不赋予第三方音乐数据再分发权。 | 法律边界。 |
| `README.md` | 公开仓库总入口，说明研究问题、公开边界、安装、合成演示、强统计复现、一键流程、测试方法和目录结构。 | 所有使用者首先阅读。 |
| `environment.yml` | Conda 环境规格，提供与 `pip` 路线并列的环境创建方式，并固定关键科学计算依赖。 | Conda 用户。 |
| `pyproject.toml` | Python 包元数据、Python 版本要求、运行依赖、测试可选依赖和包发现规则；支持 `pip install -e .`。 | 安装、打包、CI。 |
| `requirements-lock.txt` | 本 release 验证过的精确依赖版本，包括 NumPy、SciPy、scikit-learn、Mido、PyTorch 和 pytest。用于最大限度减少环境漂移。 | 严格环境复现。 |

## 2. 配置

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `configs/README.md` | 解释公开配置的范围、冻结原则和如何将配置传给公开脚本。 | 配置导航。 |
| `configs/release_rc1/protocol.json` | 机器可读的 release-candidate 协议配置，保存公开可披露的 split、阈值和分析参数；用于避免运行时随意改动方法。 | 方法核对和脚本输入。 |

## 3. 方法、数据与复现文档

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `docs/METHODS_PUBLIC.md` | 最终 Methods 的公开版，描述研究对象、reference known-family 定义、数据处理、模型条件、统计单位和科学边界；删除了内部路径及受限身份信息。 | 审稿人与方法复核。 |
| `docs/PUBLIC_FILE_GUIDE_CN.md` | 当前逐文件导览。它解释公开树中每个文件的职责，并帮助读者区分代码、匿名分析单元、冻结统计摘要和最终展示产物。 | 仓库导航与维护交接。 |
| `docs/REPRODUCIBILITY.md` | 定义四级复现能力，说明一键命令、逐字段审计、哪些统计从 family 行重新计算、哪些冻结推断字段通过充分统计量校验，以及不能仅凭 GitHub 完成的训练环节。 | 复现声明的权威入口。 |
| `docs/data_policy.md` | 说明哪些数据可公开、哪些因版权或隐私边界不公开，以及用户自行取得第三方数据时应遵守的条件。 | 数据合规。 |
| `docs/data_provenance.md` | 公开级 provenance：解释数值材料来自哪些冻结分析阶段、如何匿名化、如何以哈希绑定；不包含内部服务器路径和私有 registry。 | 证据链核查。 |
| `docs/datasets.md` | 描述论文涉及的数据集角色、公开获取责任和本仓库不随附原始音乐数据的原因。 | 数据准备。 |
| `docs/model_artifacts.md` | 说明模型定义与 checkpoint 的区别、GitHub 版是否提供权重，以及未来 companion artifact 应满足的发布条件。 | 权重和推理边界。 |
| `docs/splitting_protocol.md` | 专门说明 file-level split、known-family component split、泄漏度量和不完整 family inference 的报告方式。 | split 实现与审查。 |

## 4. 合成示例

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `examples/synthetic_demo/README.md` | 解释合成示例的目的、运行方法和预期产物，强调合成文件不是论文数据。 | 新用户快速验证。 |
| `examples/synthetic_demo/spec.json` | 合成 demo 的机器可读规格，定义少量虚构作品、变体和预期 family 关系，供生成脚本稳定地产生测试输入。 | 合成端到端流程输入。 |

## 5. 核心 Python 包

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `leakagebench_midi/__init__.py` | 包入口，暴露公共 API 和版本相关信息，使脚本和外部用户能稳定导入核心函数。 | 软件复现。 |
| `leakagebench_midi/core.py` | 核心数据结构与算法：family relation 图、连通分量、split 泄漏审计、component-atomic 分配和相关验证逻辑。 | 论文 mitigation 方法的主要实现。 |
| `leakagebench_midi/structural.py` | MIDI 结构标准化、结构特征与 pair 分类相关逻辑，用于区分 exact、规范化后等价和结构非精确关系。 | 结构控制分析。 |
| `leakagebench_midi/models/__init__.py` | 模型子包入口，集中导出公开模型组件，避免调用者依赖内部文件布局。 | 推理代码组织。 |
| `leakagebench_midi/models/loader.py` | 安全加载公开支持的模型结构和外部权重，校验 checkpoint 元数据与形状；仓库自身不附带权重。 | 用户自行取得权重后的推理。 |
| `leakagebench_midi/models/tcn.py` | 论文使用的 TCN 类模型结构公开实现，用于架构核对和合成权重加载测试。 | 模型定义复核。 |
| `leakagebench_midi/models/tokenizer.py` | 符号音乐事件 tokenizer 的公开实现和词表接口。只提供转换逻辑，不包含论文 token 序列。 | 数据预处理接口。 |
| `leakagebench_midi/models/transformer.py` | Transformer 模型结构公开实现，包括配置到模块的构造逻辑。 | 模型定义复核。 |

## 6. 强复现数据说明与完整性

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `reproduction/README.md` | 强复现数据包入口，说明目录结构、匿名化原则、运行命令及 family ID 不可与原始身份反查的边界。 | 统计复现导航。 |
| `reproduction/DATA_DICTIONARY.md` | 逐个数据表解释字段、统计单位、condition 含义和可执行的聚合方式；是读懂 CSV/JSON 的主要文档。 | 独立统计分析。 |
| `reproduction/LIMITATIONS.md` | 记录公开最小充分数据不能支持的主张，例如原始 MIDI 检查、训练重跑和未知 family 真值评估，防止把强统计复现写成全链条数据复现。 | 声明边界。 |
| `reproduction/PUBLIC_REPRODUCTION_MANIFEST.json` | 列出 reproduction 子树每个文件的字节数、SHA-256、行数和敏感内容标志。强复现脚本运行前先验证它，检测数据损坏或替换。 | 完整性入口。 |

## 7. 匿名分析单元：`reproduction/data/`

这些文件是从冻结内部证据中提取的最小充分标量。`family_id` 是 release 专用的单向伪匿名标识，不是原始 family hash；文件中没有 MIDI、token 序列或本地路径。

| 文件 | 作用 | 主要字段或分析 |
|---|---|---|
| `reproduction/data/capacity_nll_rows.csv` | Transformer 容量分析的 family × seed × condition NLL 行。用于重算不同模型规模下 same-family 暴露效应及容量趋势。 | `model`、`seed`、`condition`、`family_id`、`nll`、`token_count`。 |
| `reproduction/data/clean_test_nll_rows.csv` | family-disjoint clean-test 泛化控制的匿名 NLL 行，用于检验泄漏收益是否延伸到未暴露 family。名称中的 clean test 是实验条件，不是私有测试清单。 | 3 个 seed、clean 与 family-leak 条件、193 个 family。 |
| `reproduction/data/cross_paradigm_nll_rows.csv` | TCN、Conditional VAE、Latent Diffusion 等跨生成范式的 family/seed 级目标值。不同范式的可比指标统一映射到公开字段。 | `paradigm`、`metric`、`condition`、`family_id`。 |
| `reproduction/data/generation_family_metrics.csv` | 生成样本的 family 级复制与相似性诊断，如 shared span、Copy@τ 和 exact multi-bar extraction。 | `model`、`condition`、`family_id`、`metric`、`value`。 |
| `reproduction/data/imperfect_inference_runs.csv` | 不完整/有噪声 inferred family graph 的 2,900 次 CPU 模拟逐 run 结果。用于重算 FN、FP、bounded FP 和 combined noise 的 trade-off。 | recall/FP 目标与实测值、residual known leakage、split distortion、component 膨胀、运行时间和内存。 |
| `reproduction/data/legacy_nll_rows.csv` | 初始/legacy 确认实验的模型、seed、relation stratum 和 family 级 NLL。用于重算 Transformer-L 主效应及 structural strata。 | `relation_stratum` 支持 normalized structurally-nonexact 分析。 |
| `reproduction/data/lmd_family_size_distribution.csv` | reference known-family graph 的 component size 频数表。通过稀疏 size distribution 重算文件数、family 数、多成员 family 和最大组件，无需发布身份清单。 | `component_size`、`family_count`、`file_count`。 |
| `reproduction/data/lmd_monte_carlo_runs.json` | 文件级随机 split 的 seed-level Monte Carlo 结果及覆盖信息，用于重算 80/10/10 等划分下测试 family/file 污染率。 | `seed_level` 是逐 seed 模拟，`aggregate` 是便于核验的汇总。 |
| `reproduction/data/mechanism_family_rows.csv` | 机制分析的 family 级 clean/unrelated/same-family NLL、泄漏 gain 和多种 donor–receiver 结构相似度。 | 支持 relation stratum、相关性和局部机制检验。 |
| `reproduction/data/musical_family_metrics.csv` | 音乐结果分析的 family × model × condition 指标长表，覆盖音高、节奏、和声、结构与 surface sanity 指标。 | 用于 family bootstrap、三条件对比和多重检验。 |
| `reproduction/data/normalized_subset_rows.csv` | 结构标准化后 pair 分类的 family 级结果和事件数，用来验证主效应不只来自 exact duplicate。 | 原始/标准化分类、是否改变、donor/receiver note events。 |
| `reproduction/data/pdmx_family_deltas.csv` | PDMX 外部数据验证的每 family、每 seed treated-control 差值与方向标记。 | family-equal-weight 外部效应和负方向比例。 |
| `reproduction/data/pdmx_nll_rows.csv` | PDMX 外部验证的 seed × condition × family NLL 与 token count 原子行。 | 重算 treated/control 均值和固定 seed 效应。 |
| `reproduction/data/phase2_nll_rows.csv` | Clean、Unrelated donor、Same-family donor 三条件受控实验的 family/seed NLL。 | 论文 Phase-2 主对比和 family bootstrap 的核心输入。 |
| `reproduction/data/relatedness_features.csv` | 每个 family 的 leakage gain 与预先冻结的 pitch、interval、rhythm、duration、结构编辑、shared subsequence 和 token n-gram 特征。 | family relatedness 机制相关分析。 |
| `reproduction/data/token_localization_family_rows.csv` | 4、8、16-event 尺度的 shared/nonshared token 区域 NLL 差异。 | 重算局部效应是否集中于 shared region。 |

## 8. 冻结非识别统计审计：`reproduction/frozen/`

这组文件不是第二套“结果真值”，而是公开的非识别统计审计层。family 原子行足够时，脚本直接重算；若正式分析涉及冻结 multiplicity/display chain 或预注册 bootstrap 输出，则同时用这些文件核验具体推断字段，并在逐字段审计中标明来源。

| 文件 | 作用 | 核验内容 |
|---|---|---|
| `reproduction/frozen/capacity_trend_summary.json` | 冻结模型容量趋势估计。 | slope、95% CI、双侧 p 值、bootstrap 次数和解释限制。 |
| `reproduction/frozen/census_summary.json` | reference graph census 的冻结汇总。 | 178,561 文件身份、component 数、多成员 family、边数和最大组件等。 |
| `reproduction/frozen/confirmatory_summary.json` | legacy confirmatory 实验的 H1/H2/H3、seed 效应和 bootstrap 摘要。 | 主张状态与未访问 clean test 的边界。 |
| `reproduction/frozen/cross_paradigm_summary.json` | 跨架构/范式分析的冻结设计和结果入口。 | 范式数、family 数、seed 数、正式与 development-only 状态。 |
| `reproduction/frozen/generation_statistics_summary.json` | 生成复制诊断的正式统计摘要。 | family bootstrap、primary contrast、聚合单位和各指标结果。 |
| `reproduction/frozen/imperfect_inference_condition_summary.csv` | inferred graph 噪声模拟按 condition 汇总的 mean、median、2.5% 和 97.5% empirical interval。 | FN curve、FP trade-off、combined grid 和 component distortion。 |
| `reproduction/frozen/imperfect_inference_summary.json` | imperfect-inference 实验的总体元数据和 paired-seed 差异。 | reference 定义、seed 数、总运行时间、formal 文件未改变。 |
| `reproduction/frozen/localization_summary.json` | token localization 分析的冻结总览。 | family 数、局部尺度、统计单位和状态。 |
| `reproduction/frozen/mitigation_data_cost_summary.json` | family-aware split 与 delete-all multi-member 反事实的数据成本比较。 | 保留文件、分配策略和解释边界。 |
| `reproduction/frozen/mitigation_summary.json` | S0 file split、S1 简单 dedup、S2 reference-family-aware split 的冻结比较。 | known cross-split family、候选文件/family 和 mitigation 效果。 |
| `reproduction/frozen/musical_bootstrap_summary.json` | 音乐指标各 model × metric × contrast 的完整 family-bootstrap 结果映射。 | 均值效应、区间和原始 p 值等。 |
| `reproduction/frozen/musical_canonical_holm_results.csv` | 论文 canonical 音乐指标的域内与全局 Holm 校正表。 | raw p、domain Holm p、global Holm p、效应和 CI。 |
| `reproduction/frozen/musical_condition_summary.csv` | 各模型、条件和音乐指标的描述统计。 | mean、median、family 数和 null 方向。 |
| `reproduction/frozen/musical_holm_summary.json` | 按 distributional、harmony、rhythm、structure、surface、tonal 域组织的 Holm 结果。 | 多重比较家族边界。 |
| `reproduction/frozen/musical_paired_effects.csv` | clean/unrelated/same-family 配对效应的主汇总表。 | 平均/中位效应、CI、原始 p、标准化和相对效应。 |
| `reproduction/frozen/musical_three_condition_effects.csv` | 三训练条件全部成对 contrast 的 family 级统计。 | family 数、均值/中位效应、CI 和 raw p。 |
| `reproduction/frozen/normalized_structural_summary.json` | normalized structurally-nonexact 子集的冻结效应摘要。 | family 数、分类规则、seed-wise 效应、CI 和相对 NLL 改善。 |
| `reproduction/frozen/pdmx_summary.json` | PDMX 外部验证的冻结统计摘要。 | treated/control delta、family-cluster bootstrap、τ 和方向比例。 |
| `reproduction/frozen/phase2_summary.json` | Phase-2 受控三条件实验的冻结主统计。 | primary contrast、bootstrap seed/unit、比较结果和选择性报告防护。 |
| `reproduction/frozen/relatedness_summary.json` | relatedness 特征分析的冻结结果。 | 预先冻结指标、seed-aware sensitivity 和解释限制。 |
| `reproduction/frozen/transformer_scale_summary.json` | Transformer-S/M/L 与 TCN-384 的规模效应汇总。 | 每种架构的 effect、relative effect 和区间。 |

## 9. 最终结果索引与公开 v2 导出

| 文件 | 作用 | 复现阶段与使用者 |
|---|---|---|
| `results/RESULTS_MANIFEST.json` | 最终公开结果产物清单及 SHA-256。`reproduce_public_results.py` 依据它验证表格、图和 v2 导出未被改动。 | 冻结产物完整性。 |
| `results/manuscript_results_v2_public.csv` | 将 `manuscript_results_v2` 展平为逐字段 CSV，移除内部 source path、registry 和私有 provenance。 | 人工审阅、电子表格比较。 |
| `results/manuscript_results_v2_public.json` | 保留层级结构的公开 v2 结果，是强复现脚本逐字段比对的目标。 | 机器读取和论文数值核验。 |
| `results/manuscript_results_v2_public.sha256` | 公开 CSV/JSON 导出的冻结哈希记录，防止无痕修改。 | 完整性验证。 |

## 10. 最终图与图源数据

| 文件 | 作用 | 内容 |
|---|---|---|
| `results/figures/fig1_leakage_landscape.svg` | 主图 1 的可缩放矢量图。 | reference graph census 与随机文件 split 的 leakage landscape。 |
| `results/figures/fig2_phase2_core.svg` | 主图 2。 | Clean、Unrelated donor、Same-family donor 的受控 Phase-2 核心效应。 |
| `results/figures/fig3_model_dependence.svg` | 主图 3。 | Transformer 容量趋势和跨架构/生成范式差异。 |
| `results/figures/fig4_musical_alignment.svg` | 主图 4。 | 生成音乐向 receiver 靠近的选择性 musical metrics。 |
| `results/figures/fig5_mechanism_diagnostics.svg` | 主图 5。 | local shared-region、relatedness 与复制诊断等机制结果。 |
| `results/figures/fig6_imperfect_inference_robustness.svg` | 主图 6。 | inferred family graph recall/false-positive 噪声下的 residual leakage 与 split distortion。 |
| `results/figures/figure_capacity.csv` | 容量图的公开数值源表。 | 各模型规模效应及区间，便于重绘而无需解析 SVG。 |
| `results/figures/figure_external.csv` | 外部验证图的数值源表。 | PDMX 等外部结果。 |
| `results/figures/figure_mitigation.csv` | mitigation 图的数值源表。 | file split、dedup、family-aware split 和数据成本。 |
| `results/figures/figure_prevalence.csv` | prevalence 图的数值源表。 | census 与 random split contamination 指标。 |

## 11. 最终表格

| 文件 | 作用 | 内容 |
|---|---|---|
| `results/tables/MAIN_6_TABLES_FINAL_CN.md` | 六张最终主表的中文 Markdown 汇编，便于论文写作核对和复制。 | 全部主结果的可读版。 |
| `results/tables/table1_final.svg` | 最终表 1 矢量排版。 | reference family census 与随机 split prevalence。 |
| `results/tables/table2_final.svg` | 最终表 2 矢量排版。 | 主确认与 Phase-2 受控效果。 |
| `results/tables/table3_final.svg` | 最终表 3 矢量排版。 | 架构、容量与外部验证。 |
| `results/tables/table4_final.svg` | 最终表 4 矢量排版。 | 音乐生成指标与选择性对齐。 |
| `results/tables/table5_final.svg` | 最终表 5 矢量排版。 | 机制、局部化和复制诊断。 |
| `results/tables/table6_final.svg` | 最终表 6 矢量排版。 | mitigation 与不完整 inference robustness。 |
| `results/tables/table_architecture_capacity.csv` | 架构/容量结果的机器可读表。 | 模型、效应、相对效应和不确定区间。 |
| `results/tables/table_confirmatory.csv` | confirmatory 与 Phase-2 核心结果表。 | 预注册/冻结对比的公开数值。 |
| `results/tables/table_lmd_census.csv` | LMD reference graph census 表。 | 文件、family、multi-member component 和 leakage prevalence。 |
| `results/tables/table_mitigation.csv` | mitigation 比较表。 | S0/S1/S2 known leakage、保留文件和重新分配成本。 |
| `results/tables/table_pdmx_external.csv` | PDMX 外部验证表。 | treated/control 外部效应、CI 和方向统计。 |

## 12. 公开命令行脚本

| 文件 | 作用 | 典型输入与输出 |
|---|---|---|
| `scripts/analyze_family_effects.py` | 对 family/condition 级目标值计算配对效应和汇总统计。 | 输入公开格式的标量表，输出 family-equal-weight 分析。 |
| `scripts/audit_split.py` | 用已给定 family relation 审计 train/validation/test split 的 known cross-split family。 | 输入文件到 split 的映射和 family map，输出污染计数/比例。 |
| `scripts/build_family_map.py` | 从公开格式的 relation 边构造 connected components 和 family map。 | 输入 pairwise relation；输出 component 归属，不提供 universal detector。 |
| `scripts/build_family_split.py` | 将整个 inferred/reference component 原子地分配到 train/validation/test，并尽量保持目标比例。 | 输入 family map 与目标比例；输出 component-aware split。 |
| `scripts/classify_pair_structure.py` | 对 MIDI pair 做结构标准化与关系分层，支持 exact/canonical/structurally-nonexact 控制。 | 输入用户自备 MIDI pair；输出结构类别和摘要。 |
| `scripts/construct_contamination.py` | 构造 Clean、Unrelated donor 和 Same-family donor 的受控实验清单。 | 输入用户自备 family/split 元数据；输出条件清单，不附带论文 token。 |
| `scripts/generate_synthetic_demo.py` | 按 `examples/synthetic_demo/spec.json` 现场生成无版权风险的小型合成 MIDI 与关系文件。 | 输出到用户指定的临时/demo 目录。 |
| `scripts/reproduce_all.sh` | 审稿人一键入口：重算论文统计、校验并复制最终表图、运行全部公开 tests。 | 一个输出目录；CPU-only、无网络数据下载。 |
| `scripts/reproduce_paper_statistics.py` | 强统计复现主程序。先验证 reproduction manifest，再从匿名分析单元重算效应/区间/曲线，与公开 v2 逐字段对比并生成审计。 | 输出 `REPRODUCED_FIELD_AUDIT.csv`、`REPRODUCTION_STATUS.json`、`REPRODUCED_RESULTS.md`。 |
| `scripts/reproduce_public_results.py` | 验证 `RESULTS_MANIFEST.json` 与 25 个冻结结果产物，可将验证后的表图复制到独立输出目录。 | `--verify` 或 `--output`。它校验排版产物，不代替统计重算。 |
| `scripts/run_leakage_census.py` | 在用户自备 file universe/family map 上运行 census 与 split contamination 汇总。 | 输出 component 数、size distribution 和 known leakage。 |
| `scripts/run_synthetic_demo.sh` | 合成端到端流程包装器，依次生成虚构 MIDI、构建 family map、split 并审计。 | 用于安装后的最快 smoke test。 |
| `scripts/validate_environment.py` | 检查 Python/依赖版本、CPU/GPU 可见性和必要模块导入，帮助解释环境差异。 | 输出环境验证报告，不修改实验结果。 |

## 13. 公开测试

| 文件 | 作用 | 覆盖风险 |
|---|---|---|
| `tests/test_model_loader.py` | 用临时合成权重测试模型 loader、配置校验和错误 checkpoint 拒绝逻辑。 | 模型定义与加载兼容性；不需要正式权重。 |
| `tests/test_public_workflows.py` | 测试公开 CLI 和合成端到端工作流。 | 命令行连接错误、输出契约和 split 审计。 |
| `tests/test_release_safety.py` | 检查公开树不包含原始 MIDI、checkpoint、危险目录、绝对路径或明显敏感文件。 | 发布边界回归。 |
| `tests/test_strong_reproduction_bundle.py` | 验证强复现 manifest 哈希、伪匿名 family ID 形状、无 token-sequence 字段及 CPU-only 脚本约束。 | 匿名统计包完整性与敏感内容防护。 |
| `tests/test_validation_and_normalization.py` | 测试输入验证、component split 约束和结构标准化边界条件。 | 算法正确性与异常输入。 |

## 14. 推荐阅读顺序

审稿人或首次复现者建议依次阅读：

1. `README.md`：理解研究与公开边界。
2. `docs/REPRODUCIBILITY.md`：确认仓库能复现到什么层级。
3. `docs/METHODS_PUBLIC.md` 与 `docs/splitting_protocol.md`：核对方法。
4. `reproduction/DATA_DICTIONARY.md`：理解匿名分析单元。
5. 运行 `bash scripts/reproduce_all.sh ./_reproduced_release`。
6. 查看生成的 `REPRODUCTION_STATUS.json` 和 `REPRODUCED_FIELD_AUDIT.csv`。
7. 对照 `results/` 中的最终表图和 `results/manuscript_results_v2_public.json`。

如果目标是把方法用于一个新数据集，应从 `scripts/build_family_map.py`、`scripts/build_family_split.py` 和 `scripts/audit_split.py` 开始，并始终将输出描述为 adopted detector/reference relation 下的 **known-family leakage**，而不是对世界上所有真实 same-work relation 的完备识别。
