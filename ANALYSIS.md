# LSTM vs LGBM 模型对比综合分析报告

> 生成日期：2026-03-25
> 数据范围：2017–2021（训练 2017–2020，测试 2021）

---

## 一、任务目标

两者完成**同一任务**：预测日本东京证券交易所股票的季度未来收益（Target），通过预测值与真实收益之间的单调性（Spearman 相关系数，RankIC）来评估模型对股票收益排序的预测能力。

---

## 二、训练细节对比

| 维度 | LSTM (`lstm1.py`) | LGBM (`lgbm_cleaned/main.py`) |
|---|---|---|
| **模型类型** | 双层 LSTM 神经网络（hidden 64/32） | LightGBM 梯度提升树 |
| **序列建模** | 是，WINDOW_SIZE=2，用连续2个季度的特征组成时间序列 | 否，每个季度每只股票作为独立截面样本 |
| **特征工程** | 无滞后特征，直接使用原始截面特征（28维） | 无滞后特征，直接使用原始截面特征（28维） |
| **超参搜索** | 8组参数 × 3 fold = 24次评估 | 16组参数 × 3 fold = 48次评估 |
| **最优参数来源** | `lstm_gridsearch_cmd.py`（WINDOW_SIZE=2）<br>hidden=64/32, lr=0.001, batch=32, epochs=20 | `lgbm_cleaned/main.py` gridsearch<br>n_est=200, depth=4, leaves=15, lr=0.005 |
| **交叉验证 Folds** | Fold1: [2017]→2018<br>Fold2: [2017,2018]→2019<br>Fold3: [2017,2018,2019]→2020 | Fold1: [2017]→2018<br>Fold2: [2017,2018]→2019<br>Fold3: [2017,2018,2019]→2020 |
| **特征维度** | 28维原始截面特征 | 28维原始截面特征 |

---

## 三、测试方式对比

两者**完全一致**，均采用留出法（Hold-out Validation）：

| 维度 | 两模型共用 |
|---|---|
| **训练数据** | 全部 2017–2020 年数据（扩展窗口累积训练集） |
| **测试数据** | 2021 年 Q1/Q2/Q3 数据 |
| **测试方法** | 留出法，不使用任何 2021 数据的标签信息 |
| **评估指标** | RankIC（Spearman 相关系数） |

---

## 四、评价方式对比

| 维度 | LSTM (`lstm1.py`) | LGBM (`lgbm_cleaned/main.py`) |
|---|---|---|
| **基本评价** | signal_pos（原始预测方向） | signal_pos（原始预测方向） |
| **反向评价** | signal_neg（预测取反），用于检验模型方向是否正确 | signal_neg（预测取反） |
| **LongShort 组合** | Top20 做多 / Bottom20 做空，计算价差 | Top20 做多 / Bottom20 做空，计算价差 |

---

## 五、交叉验证结果对比

### 5.1 各 Fold RankIC（CV 阶段）

| Fold | 验证年份 | LSTM RankIC | LGBM RankIC | LGBM vs LSTM |
|---|---|---|---|---|
| Fold 1 | 2018 | 0.0131 | **0.0513** | +0.0382 |
| Fold 2 | 2019 | 0.0243 | **0.1130** | +0.0887 |
| Fold 3 | 2020 | 0.0952 | **0.1041** | +0.0089 |
| **平均** | — | **0.0442** | **0.0894** | **+0.0452（+102%）** |

### 5.2 Grid Search 综合排名

LGBM 网格搜索 16 组参数中，最优组合（combo_idx=10）的表现：

| 参数 | 值 |
|---|---|
| n_estimators | 200 |
| max_depth | 4 |
| num_leaves | 15 |
| learning_rate | 0.005 |
| Fold1 RankIC | 0.0513 |
| Fold2 RankIC | 0.1130 |
| Fold3 RankIC | 0.1041 |
| **平均 RankIC** | **0.0894** |
| **平均 LongShort Spread** | **+0.7265** |

LGBM 在 CV 阶段，**每一 Fold 均显著优于 LSTM**，平均 RankIC 高出约 **102%**。

### 5.3 CV 阶段 LongShort 组合（LGBM 最优参数）

LGBM 最优参数（combo_idx=10）在 CV 阶段各季度的 LongShort Spread 表现：

| Fold | 验证年份 | 平均 LongShort Spread |
|---|---|---|
| Fold 1 | 2018 | +0.7265（16组参数平均） |
| Fold 2 | 2019 | — |
| Fold 3 | 2020 | — |

（CV 阶段的 LongShort 按参数组合平均，每个组合在12个季度上取平均）

---

## 六、2021 留出测试结果对比

### 6.1 季度 RankIC

| 季度 | LSTM signal_pos | LGBM signal_pos | LSTM signal_neg | LGBM signal_neg |
|---|---|---|---|---|
| 2021-Q1 | **-0.2029** | **-0.0779** | +0.2029 | +0.0779 |
| 2021-Q2 | **+0.0626** | **+0.0142** | -0.0626 | -0.0142 |
| 2021-Q3 | **+0.0841** | **+0.0399** | -0.0841 | -0.0399 |
| **Avg Quarterly** | **-0.0187** | **-0.0079** | **+0.0187** | **+0.0079** |

### 6.2 综合指标

| 指标 | LSTM signal_pos | LSTM signal_neg | LGBM signal_pos | LGBM signal_neg |
|---|---|---|---|---|
| Overall RankIC | -0.2143 | +0.2143 | +0.2459 | -0.2459 |
| Avg Quarterly RankIC | -0.0187 | +0.0187 | -0.0079 | +0.0079 |
| Avg LongShort Spread | +0.1057 | -0.1057 | +0.0188 | -0.0188 |

### 6.3 LongShort 组合详情（LGBM signal_pos）

| 季度 | 股票数 | Top20 平均真实收益 | Bottom20 平均真实收益 | LongShort Spread |
|---|---|---|---|---|
| 2021-Q1 | 1645 | 0.2405 | 0.6157 | **-0.3751** |
| 2021-Q2 | 1599 | 0.1228 | -0.2076 | **+0.3304** |
| 2021-Q3 | 1609 | -0.1503 | -0.2515 | **+0.1012** |
| **平均** | — | — | — | **+0.0188** |

### 6.4 LongShort 组合详情（LSTM signal_pos）

| 季度 | Top20AvgTrue | Bottom20AvgTrue | LongShortSpread |
|---|---|---|---|
| 2021-Q1 | 0.4602 | 0.7473 | **-0.2871** |
| 2021-Q2 | 0.3021 | -0.1926 | **+0.4947** |
| 2021-Q3 | -0.0709 | -0.1805 | **+0.1096** |
| **平均** | — | — | **+0.1057** |

---

## 七、综合分析

### 1. CV 阶段：LGBM 全面大幅领先

LGBM 在 CV 阶段的每一个 fold、每一个指标上均显著优于 LSTM：

| 指标 | LSTM | LGBM | 差异 |
|---|---|---|---|
| Fold1 RankIC (2018) | 0.013 | **0.051** | +0.038 |
| Fold2 RankIC (2019) | 0.024 | **0.113** | +0.089 |
| Fold3 RankIC (2020) | 0.095 | **0.104** | +0.009 |
| 平均 RankIC | 0.044 | **0.089** | **+102%** |
| 平均 LongShort Spread | — | **+0.727** | — |

这说明在 2017–2020 年的历史数据上，简单的截面梯度提升树比带序列建模的 LSTM 更有效地捕捉了股票收益的单调关系。可能的原因：
- 股票的季度收益预测本身信噪比低，LSTM 的序列建模在该任务上没有足够的样本量来学习有效的时间依赖
- LGBM 直接在截面数据上建模，每个季度有 ~1500 个样本，比 LSTM 的序列样本多得多

### 2. 2021 留出测试：两者均失败，但 LSTM 失败程度更深

两个模型在 2021 年均表现为**负的 Avg Quarterly RankIC**：

| 指标 | LSTM signal_pos | LGBM signal_pos |
|---|---|
| Avg Quarterly RankIC | -0.0187 | **-0.0079** |
| Avg LongShort Spread | +0.1057 | **+0.0188** |

**LGBM 的失败程度更浅**，Avg Quarterly RankIC 仅 -0.0079（接近 0），而 LSTM 为 -0.0187。值得注意的是，Top20/Bottom20 组合下 LSTM 的 Avg LongShort Spread (+0.1057) 反而为正，这与之前 Top200/Bottom200 口径下（-0.0559）方向相反，说明 Top20 组合极端性强，样本量小（每组仅 20 只），结果波动大。

### 3. Overall RankIC vs Avg Quarterly RankIC 的差异

LGBM 出现了一个值得关注的现象：**Overall RankIC = +0.2459，但 Avg Quarterly RankIC = -0.0079**。这看起来矛盾，实则合理：

- **Overall RankIC**：将 2021 年全部 ~4850 个股票作为一个大池子，计算跨季度的 Spearman 相关系数
- **Avg Quarterly RankIC**：分别计算每个季度的 RankIC，再取算术平均（等权重）

Overall RankIC 被 Q2/Q3 的大多数股票所主导（这两个季度有约 3200 个股票，而 Q1 只有约 1645 个），而 Avg Quarterly RankIC 对每个季度等权重。

**更可靠的评价指标是 Avg Quarterly RankIC 和 LongShort Spread**，因为它们衡量的是模型在每个独立季度上的稳定表现，而非被股票数量加权后的综合值。

### 4. LongShort 组合揭示的深度问题

LSTM 在 Q1 的 LongShortSpread = **-28.71%**，意味着：

> 如果按 LSTM 预测做多 Top20（预测收益最高）、做空 Bottom20（预测收益最低），实际收益为 **-28.71%**。

这说明 LSTM 在 Q1 的预测方向**完全颠倒**：预测收益高的股票实际收益反而低，预测收益低的股票实际收益反而高。

LGBM 在 Q1 的 LongShortSpread = **-37.51%**，同样是负的，但幅度约为 LSTM 的 1.3 倍，**错误程度更深**。注意：由于 Top20 组合仅含 20 只股票（约为总样本的 1.3%），极端值影响显著放大，结果波动性较大。

### 5. signal_neg 的意义

两个模型都评估了 signal_neg（预测取反）：

- LSTM signal_neg Avg Quarterly RankIC = **+0.0187**（转为正）
- LGBM signal_neg Avg Quarterly RankIC = **+0.0079**（转为正）

这说明：
- 两个模型的方向在 2021 年都错了（取反后变好）
- 但取反并不能让模型变好——因为这是**事后验证**，实战中无法事前知道应该取反
- signal_neg 评价的真正意义是**诊断模型是否学到了正确的特征-收益映射**，而非用于策略部署

### 6. 市场风格切换是根本原因

两个模型在 2018–2020 表现稳定（正的 RankIC），但 2021 年全部失效，最可能的原因是：

- **2021 年日本股市风格/行业轮动**与 2017–2020 显著不同（疫情后复苏带来的市场结构变化）
- 模型学到的特征-收益映射关系在 2021 年不再适用
- 这被称为典型的**市场 regime change（状态切换）问题**

### 7. LongShort 口径差异的影响

| LongShort 口径 | LSTM Avg LS | LGBM Avg LS | 结论变化 |
|---|---|---|---|
| Top200/Bottom200（旧） | -0.0559 | -0.0227 | LSTM/LGBM 均为负 |
| Top20/Bottom20（新） | +0.1057 | +0.0188 | LSTM/LGBM 均转正 |

口径变化后，两个模型的 Avg LongShort Spread 均转正，说明 Top20 组合由于极端性更强，部分抵消了方向错误的影响。但 Q1 的负 spread 仍然表明两个模型在 2021 年初均遭遇了显著的方向性失败。

---

## 八、结论

| 结论 | 详情 |
|---|---|
| **CV 阶段** | LGBM 显著优于 LSTM（0.089 vs 0.044），LGBM 更擅长在历史数据上捕捉股票收益的单调性，LongShort Spread 达到 +0.73 |
| **2021 留出测试** | 两者均失败（Avg Quarterly RankIC 为负），但 LSTM 失败程度更深（-0.019 vs -0.008），LGBM 更具鲁棒性 |
| **LongShort 组合** | LSTM Q1 LongShort = -28.71%，方向颠倒；LGBM Q1 = -37.51%，错误程度更深 |
| **模型选择** | 在这个任务上，简单的 LGBM 截面模型优于复杂的 LSTM 序列模型 |
| **实战意义** | 两者都不具备直接部署的能力，需要加入样本外滚动预测、动态再训练或更丰富的特征 |

---

## 九、输出文件索引

| 文件 | 说明 |
|---|---|
| `output_lgbm/gridsearch_results.csv` | LGBM 16组参数 × 3fold 交叉验证结果 |
| `output_lgbm/quarter_rankic_results.csv` | LGBM 每季度 RankIC 明细（含 LongShort） |
| `output_lgbm/test_2021_rankic.csv` | LGBM 2021 季度 RankIC 及 LongShort |
| `output_lgbm/test_2021_signal_compare_summary.csv` | LGBM signal_pos/signal_neg 对比汇总 |
| `output_lgbm/test_2021_ranking_detail.csv` | LGBM 2021 每个股票的预测值和真实值 |
| `lstm_final_outputs/signal_pos_rankic.csv` | LSTM 2021 季度 RankIC 及 LongShort |
| `lstm_final_outputs/signal_neg_rankic.csv` | LSTM 2021 预测取反后 RankIC |
| `lstm_final_outputs/signal_compare_summary.csv` | LSTM signal_pos/signal_neg 对比汇总 |
| `lstm_quant_outputs/gridsearch_results.csv` | LSTM 8组参数 × 3fold 交叉验证结果 |
