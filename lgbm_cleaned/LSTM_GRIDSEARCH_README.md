# LSTM GridSearch 脚本功能说明

## 概述

`lstm_gridsearch_cmd1.py` 是一个基于季度数据（cleaned_data）的 LSTM 模型超参数搜索脚本。

## 数据特点

- **数据来源**: cleaned_data/ 目录下的季度数据
- **数据频率**: 每季度更新一次（3个月）
- **预测目标**: 预测下一季度的股票收益率 (Target)
- **输入特征**: 所有非 ID/时间/目标的列

## 核心流程

### 1. 数据准备
```
- 读取 train_dataset_clean.csv 和 test_dataset.csv
- 按 SecuritiesCode 和 Quarter 排序
- 特征列：除 ID、Quarter、Target 外的所有列
- 缺失值处理：先用 ffill/bfill，再用 0 填充
```

### 2. 序列构建
```
- window_size = 3: 用前3个季度的数据预测下1个季度
- 每个样本: (3季度特征) -> (1季度目标)
- 输出: X (samples, 3, num_features), y (samples, 1), meta (股票代码、季度信息)
```

### 3. LSTM 模型结构
```
SimpleLSTM:
  - LSTM(input_size, hidden_size=hidden1, batch_first=True)
  - Linear(hidden1 -> hidden2)
  - ReLU
  - Linear(hidden2 -> 1)
```

### 4. 超参数搜索
```
参数网格:
  - hidden1: [32, 64]
  - hidden2: [16, 32]
  - learning_rate: [0.001, 0.0005]
  - batch_size: [32]
  - epochs: [20]

共 2×2×2×1×1 = 8 种组合
```

### 5. 验证策略 (Expanding Window)
```
Fold 1: Train [2017] -> Val [2018]
Fold 2: Train [2017, 2018] -> Val [2019]
Fold 3: Train [2017, 2018, 2019] -> Val [2020]
```

### 6. 评估指标
```
- RankIC: Spearman相关系数
- 按每个季度计算 rankIC
- 返回该 fold 的平均 rankIC
```

### 7. 输出文件
```
保存到 SAVE_DIR (lstm_quant_outputs):
- gridsearch_results.csv: 所有参数组合的汇总结果
- quarter_rankic_results.csv: 每个季度级别的 rankIC 详情

输出字段:
- gridsearch_results: combo_idx, hidden1, hidden2, lr, batch_size, epochs, fold1/2/3_rankic, avg_rankic
- quarter_rankic: combo_idx, fold_idx, train_years, val_year, params, LabelQuarter, n_stocks, rankIC
```

## 关键函数

| 函数 | 功能 |
|------|------|
| `SeqDataset` | PyTorch Dataset 类 |
| `SimpleLSTM` | LSTM 模型定义 |
| `build_sequences` | 构建时间序列样本 |
| `calc_rankic` | 计算 RankIC (Spearman) |
| `train_one_fold` | 单个 fold 的训练和验证 |
| `main` | 主程序入口 |

## 使用说明

```bash
python lstm_gridsearch_cmd1.py
```

## 相关文件

- `cleaned_data/train_dataset_clean.csv` - 训练数据
- `cleaned_data/test_dataset.csv` - 测试数据
- `lstm_quant_outputs/` - 输出目录（需手动创建或修改 SAVE_DIR）
