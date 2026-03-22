# Stock Embedding Transformer - 修改日志

## 文件信息
- **源文件**: itransformer_model.py
- **目标文件**: new_transformer_embedding.py
- **创建日期**: 2026-03-07

---

## 核心修改要点

### 1. 显存优化 (最重要)

**问题**:
- iTransformer 将"股票×特征"作为变元进行注意力计算
- 复杂度: O(N²)，其中 N = 股票数 × 特征数
- 2000 股票 × 41 特征 = 82,000 变元
- 注意力矩阵: 82,000² ≈ 67 亿元素 → OOM

**解决方案**:
- 每只股票作为独立样本处理
- 注意力仅在时间步 (seq_len) 上计算
- 复杂度: O(T²)，其中 T = 序列长度 (20)
- 注意力矩阵: 20² = 400 元素

**效果**:
- 显存需求从 O(N²×D) 降为 O(T²×D)
- batch_size 从 4 提升到 256
- 支持 2000+ 股票高效训练

---

### 2. 输入格式改变

**iTransformer (旧)**:
```
输入: (batch, seq_len, stocks × features)
    = (batch, 20, 2000 × 41) = (batch, 20, 82000)
输出: 每批次所有股票的预测
```

**Stock Embedding Transformer (新)**:
```
输入: (batch, seq_len, features) + stock_ids
    = (batch, 20, 41) + (batch,)
    其中每行是一只股票的一个序列

输出: (batch,) - 每只股票的 30 日收益预测
```

---

### 3. 股票嵌入 (Stock Embedding)

**新增层**: `nn.Embedding(num_stocks, embed_dim)`

**作用**:
- 将股票代码映射为 embed_dim 维向量
- 让模型能区分不同股票的特性
- 学习每只股票的"个性"

**实现**:
```python
# 1. Stock Embedding: (batch,) -> (batch, 1, embed_dim)
stock_emb = self.stock_embedding(stock_ids).unsqueeze(1)

# 2. 特征投影: (batch, seq_len, num_features) -> (batch, seq_len, d_model - embed_dim)
x_feat = self.feature_proj(x)

# 3. 拼接: (batch, seq_len, d_model)
stock_emb = stock_emb.expand(-1, x.size(1), -1)  # 广播到所有时间步
x = torch.cat([x_feat, stock_emb], dim=-1)
```

---

### 4. 模型架构

```python
class StockEmbeddingTransformer(nn.Module):
    def __init__(self, num_stocks, num_features, embed_dim=64, d_model=64, ...):
        # Stock Embedding
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)

        # 特征投影
        self.feature_proj = nn.Linear(num_features, d_model - embed_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, ...)

        # Transformer 编码器 (仅时间维度注意力)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, ...)
        self.transformer = nn.TransformerEncoder(encoder_layer, ...)

        # 输出投影
        self.fc = nn.Sequential(...)
```

---

### 5. 数据加载

**新函数**: `create_transformer_sequences()`

**返回**:
- X: `(total_samples, seq_len, num_features)` - 每行是一只股票
- y: `(total_samples,)` - 30 日收益
- stock_ids: `(total_samples,)` - 股票索引
- dates_arr: 日期数组

**对比**:
- 旧: 返回 `(X, y, dates_arr, num_features)`，X 形状 `(samples, seq_len, stocks×features)`
- 新: 返回 `(X, y, stock_ids, dates_arr, num_features)`，X 形状 `(samples, seq_len, features)`

---

### 6. 训练流程

**修改点**:
1. DataLoader 现在返回 `(batch_X, batch_y, batch_stock_ids)` 三个 tensor
2. 模型前向传播接受两个输入: `model(x, stock_ids)`
3. 训练循环处理 stock_ids

**保持不变**:
- Expanding Window 训练策略 (4 轮)
- 特征归一化 (仅用训练集统计量)
- 目标归一化 (全局归一化)
- 评估指标

---

## 超参数对比

| 参数 | iTransformer | StockEmbeddingTransformer |
|------|--------------|---------------------------|
| batch_size | 4 | 256 |
| 注意力复杂度 | O(N²) | O(T²) |
| 2000股票显存 | OOM | 可运行 |
| embed_dim | N/A | 64 |

---

## 评估指标 (保持不变)

- **RMSE**: 预测均方根误差
- **Spearman**: 预测与真实收益的 Spearman 相关系数
- **Hit Ratio**: 预测方向准确率
- **Portfolio Sharpe**: 每日调仓组合的夏普比率
- **Kaggle Sharpe**: Kaggle 官方评估的夏普比率

---

## 训练轮次 (保持不变)

- Round 1: Train 2017 → Validate 2018
- Round 2: Train 2017-2018 → Validate 2019
- Round 3: Train 2017-2019 → Validate 2020
- Final: Train 2017-2020 → Pred 2021

---

## 风险与注意事项

1. **Stock Embedding 冷启动**: 新股票可能 embedding 质量低
2. **时序信息丢失**: 不再使用股票间横截面关系
3. **训练效率**: 每轮需要遍历更多样本（但单轮更快）
4. **兼容性**: 确保评估指标与 Kaggle 官方一致

---

## 代码位置

- **模型**: `StockEmbeddingTransformer` 类 (new_transformer_embedding.py:214)
- **数据加载**: `create_transformer_sequences` 函数 (new_transformer_embedding.py:323)
- **训练**: `train_stock_transformer_model` 函数 (new_transformer_embedding.py:456)
- **预测**: `predict_stock_transformer` 函数 (new_transformer_embedding.py:492)
- **主函数**: `predict_with_stock_transformer` 函数 (new_transformer_embedding.py:510)

---

## 运行方式

```bash
cd d:\code\Competition\jpx-tokyo-stock-exchange-prediction
python new_transformer_embedding.py
```

输出:
- `output_stock_embedding_transformer/predictions.csv`
- `output_stock_embedding_transformer/metrics.csv`
