# 虚拟细胞系统：三个核心模型与所用数据

## 汇报摘要

- 整套系统由三个相互衔接的模块组成：GeneJEPA 把单细胞基因表达压缩为 768 维细胞状态；ST-A 根据对照细胞状态和药物/剂量信息，预测处理后的 768 维细胞状态；Decoder 再把预测状态还原为 5,000 个基因的表达值。
- 三个模块使用同一套 GeneJEPA latent 作为接口。ST-A 不直接预测基因表达，Decoder 也不负责学习药物扰动；这种分工把“理解细胞状态”“预测状态变化”“还原基因表达”拆成了三个独立问题。
- 数据来自 Tahoe-100M。本地完整数据为 95,624,334 个细胞；Experiment 1 从中形成 56,993 个可用处理条件，并缓存了 30,839,089 个唯一细胞的 GeneJEPA embedding，供 ST-A 和 Decoder 使用。

## 一、总体流程

```text
Gene expression
      ↓
GeneJEPA：把单细胞表达编码为 cell latent
      ↓  [768]
ST-A + drug/dose：预测药物处理后的细胞状态
      ↓  [256, 768]
Decoder：把 predicted treated latent 还原为基因表达
      ↓
Predicted treated gene expression [256, 5000]
```

三个模块的职责边界如下：

1. **GeneJEPA**：学习单细胞转录组的通用表示，把不同长度的稀疏基因表达压缩成统一的 768 维向量。
2. **ST-A**：学习药物和剂量如何改变一组细胞的 latent 分布，输出预测的 treated cell latent set。
3. **GeneJEPA Decoder**：把每个 768 维 latent 解码成 5,000 个目标基因的表达值，形成可解释的预测表达矩阵。

为避免数据规模口径混淆，本文区分三类数量：Tahoe-100M 的完整数据规模、GeneJEPA checkpoint 的实际训练候选池，以及 Experiment 1 正式缓存并供下游模型读取的细胞数。

---

## 二、GeneJEPA：把基因表达编码成细胞状态

### 2.1 模型信息

| 项目 | 内容 |
|---|---|
| 模型名称 | GeneJEPA（Gene Joint-Embedding Predictive Architecture） |
| 主要作用 | 从单细胞 RNA 表达中学习通用细胞状态表示，为扰动预测和表达解码提供统一 latent 空间 |
| 模型类型 | 自监督 JEPA；Perceiver-style gene-set encoder；Student + EMA Teacher + Predictor |
| 输入 | 一个细胞中出现的基因 token ID，以及相应的表达 counts；每个细胞的有效基因数可以不同 |
| 输入表示 | 基因身份 embedding 与表达值 Fourier features 合并，形成 768 维 gene token representation |
| 主体结构 | 512 个 learned latent tokens，通过 cross-attention 读取基因集合，再经过 12 个 latent Transformer blocks；6 个 attention heads |
| 输出 | 每个物理细胞一个 cell embedding |
| 输出维度 | 768 |
| 当前使用分支 | 冻结的 Epoch25 EMA Teacher，调用 `get_embedding(use_teacher=True)` |
| 当前 checkpoint | `scjepa-epoch=25-val_loss=0.179.ckpt` |

GeneJEPA 内部先把每个有效基因变成 768 维 token，再由 512 个 learned latent tokens 汇总整组基因信息；Transformer 处理后对 512 个 latent tokens 求平均，得到一个 768 维单细胞 embedding。

训练时，模型把一个细胞的基因集合分成可见 context 和被遮挡 target。Student 根据 context 预测 target 的 latent representation，EMA Teacher 提供稳定的 target 表示。训练目标由 JEPA 表示预测的 cosine loss，以及抑制 representation collapse 的 variance/covariance regularization 组成。

### 2.2 使用的数据

| 数据口径 | 数量或说明 |
|---|---|
| 数据来源 | Hugging Face `vevotx/Tahoe-100M` 单细胞药物扰动图谱 |
| 本地完整细胞数 | 95,624,334 |
| 数据分片 | 3,388 个 Parquet shards |
| 完整数据中的 cell lines | 50 |
| 完整数据中的 drug labels | 380，包含 379 个药物和 DMSO 对照 |
| 实验板 | 14 个 plates |
| Gene vocabulary | 62,710 个基因 |
| 当前 checkpoint 的训练候选池 | 固定 seed 42，从训练池选取 843 个 shards，共 23,793,195 个物理细胞 |
| 每 epoch 训练取样 | 700,000 个细胞 |
| 验证取样 | 10,000 个细胞 |
| 最终正式 embedding cache | 30,839,089 个唯一细胞，每个为 768 维 float32 latent |

GeneJEPA 的输入不是固定长度的全基因矩阵，而是每个细胞实际出现的稀疏基因列表。输入阶段去除 Tahoe 记录中的前导 sentinel，按照 62,710-gene vocabulary 映射 gene token ID，对表达 counts 做一次 `log1p`，再使用 Tahoe 全局 mean/std 标准化。模型最终输出保留正负值的原始 768 维 latent，不进行 centering、whitening 或 L2 normalization。

当前下游正式 cache 共包含：

- 28,639,959 个 treated cells；
- 2,199,130 个 eligible DMSO control cells；
- 合计 30,839,089 个唯一细胞。

每个物理细胞只提取一次 GeneJEPA embedding，ST-A 和 Decoder 通过全局 embedding index 复用这些结果。

---

## 三、ST-A：预测药物处理后的细胞 latent 分布

### 3.1 模型信息

| 项目 | 内容 |
|---|---|
| 模型名称 | ST-A（State Transition — Absolute） |
| 主要作用 | 根据一组对照细胞及药物/剂量条件，直接预测处理后一组细胞的 latent 分布 |
| 模型类型 | 基于官方 STATE `StateTransitionPerturbationModel` 的 Transformer-based perturbation model |
| Control 输入 | `ctrl_cell_emb [B, 256, 768]`，为冻结 GeneJEPA 生成的原始 control latent |
| 药物/剂量输入 | `pert_emb [B, 256, 380]`，同一 condition 的 256 个细胞复制相同的 perturbation vector |
| Perturbation encoding | 379 维 drug one-hot + 1 维由 train split 拟合的标准化 `log10(dose_uM)` |
| 输出 | `predicted treated latent [B, 256, 768]` |
| 输出方式 | Absolute prediction：直接输出 `Zpred`，不与输入 control latent 做 residual 相加 |
| Final activation | Identity，输出保留 signed latent |
| 主要 loss | Energy distance；每个 set 计算一个 Energy，再在 batch 内求平均 |
| 参数规模 | 总参数 101,560,320；可训练参数 76,984,320 |

这里的 **256** 表示一个 cell set 中包含 256 个细胞，**768** 表示每个细胞的 GeneJEPA latent 维度。`B` 表示一个 batch 中并行处理的 condition 数量。

ST-A 先分别把 768 维 control latent 和 380 维 perturbation vector 投影到 768 维 hidden space，再交给双向 Transformer backbone 建模细胞集合。正式配置使用 8 个 Transformer layers、12 个 attention heads、768 hidden size 和 3,072 intermediate size，最后通过 768→768 的线性层输出每个预测细胞的 latent。

### 3.2 正式训练参数

| 参数 | 正式配置 |
|---|---:|
| Set size | 256 control cells + 256 treated target cells / condition |
| GPU 数 | 2 |
| Batch size | 64 conditions / GPU |
| Micro global batch | 128 conditions |
| Gradient accumulation | 2 |
| Effective global batch | 256 conditions |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Precision | BF16 autocast forward；Energy loss 使用 FP32 |
| 训练上限 | 30 epochs |
| Best epoch | 28 |
| 最终模型 | ST-A v2 `best.pt`（best_epoch=28） |

### 3.3 使用的数据

ST-A 使用 Tahoe-100M 中能够组成“同一实验上下文的 DMSO control → drug-treated”配对数据。进入 Experiment 1 的条件必须同时拥有足够的 control 和 treated cells，以支持两侧各无放回抽取 256 个细胞。

| 数据项 | 数量 |
|---|---:|
| Eligible cell lines | 48 |
| Drugs | 379 |
| Doses | 3：0.05、0.5、5.0 μM |
| Plates | 14 |
| Eligible conditions | 56,993 |
| `(cell_line_id, drug)` edges | 17,174 |
| Eligible treated source cells | 92,703,975 |
| 实际缓存 treated cells | 28,639,959，按 condition 最多固定缓存 512 个 |
| 实际缓存 DMSO cells | 2,199,130，使用全部 eligible controls |
| 正式 cache 总数 | 30,839,089 个唯一细胞 |

Train/validation/test 按 `(cell_line_id, drug)` edge 划分。一个 edge 下的所有 dose、plate、sample 和 plate6/plate14 replicate 必须进入同一个 split，因此同一个生物关系不会跨 split 泄漏。

| Split | Conditions | Edges | Drugs | Cell lines | Cached treated cells |
|---|---:|---:|---:|---:|---:|
| Train | 45,652 | 13,740 | 379 | 48 | 22,941,936 |
| Validation | 5,657 | 1,717 | 377 | 47 | 2,841,724 |
| Test | 5,684 | 1,717 | 377 | 47 | 2,856,299 |
| **合计** | **56,993** | **17,174** | **379** | **48** | **28,639,959** |

DMSO cache 按 control pool 全局去重并被多个 condition 复用，因此 2,199,130 个 DMSO cells 不再重复分摊到 train/validation/test 三行中。

一个 ST-A 训练样本的构成是：

```text
同一实验上下文中的 256 个 DMSO control cells
        +
该 condition 的 drug one-hot 与标准化 dose（复制到 256 个细胞）
        ↓
ST-A 预测 256 个 treated cell latents
        ↕ Energy distance
同一 condition 中真实抽取的 256 个 treated cell latents
```

每次构造 set 时，control 和 treated 两侧都在各自 pool 内无放回抽取；不同训练 epoch 的 set 可以重合，从而在不重复提取 embedding 的前提下产生不同的细胞组合。

### 3.4 三个数据术语

- **Condition**：一个具体处理条件，包含特定的 cell line、drug、dose 以及对应的 plate/sample 实验上下文，并匹配相应的 DMSO control pool。
- **Edge**：一个 `(cell_line_id, drug)` 组合；同一 edge 下可以包含多个 dose、plate 和实验重复。
- **Held-out**：validation/test 中没有在 train 出现过的 `(cell line, drug)` 组合；但该 drug 和该 cell line 各自都在 train 的其他组合中出现过，因此这里考察的是“见过药物 + 见过细胞系”的新组合泛化，而不是 unseen-drug zero-shot。

---

## 四、GeneJEPA Decoder：把 latent 还原为基因表达

### 4.1 模型信息

| 项目 | 内容 |
|---|---|
| 模型名称 | GeneJEPA Decoder v1 |
| 主要作用 | 把每个 GeneJEPA cell latent 解码为可解释的基因表达向量 |
| 输入 | 一个物理细胞的原始、signed、冻结 GeneJEPA Epoch25 EMA Teacher latent |
| 输入维度 | 768 |
| 输出 | 该细胞在固定 gene panel 上的预测表达 |
| 输出维度 | 5,000 genes |
| 整体架构 | `768 → 1024 → 1024 → 512 → 5000` 的 MLP |
| Hidden block | Linear → LayerNorm → GELU → Dropout(0.1) |
| Final activation | Softplus |
| Loss | 在 `log1p(CP10000)` 表达空间计算 MSE |
| 参数规模 | 4,931,976 个可训练参数 |

Decoder 是逐细胞模型：一个 768 维 latent 对应一个 5,000 维表达向量。它不输入 drug、dose、cell line 或 plate；药物信息已经由 ST-A 反映在 predicted treated latent 中。

### 4.2 正式训练参数

| 参数 | 正式配置 |
|---|---:|
| GPU 数 | 2 |
| Batch size | 1,024 cells / GPU |
| Global batch | 2,048 cells |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Precision | BF16 autocast forward；prediction、target 和 MSE 使用 FP32 |
| 训练上限 | 10 epochs |
| Best epoch | 9 |
| 最终模型 | GeneJEPA Decoder v1 `best.pt`（best_epoch=9） |

### 4.3 使用的数据

Decoder 的 latent 输入直接来自正式 GeneJEPA Epoch25 embedding cache，不重新运行 GeneJEPA，也不对 latent 做 centering、whitening、L2 normalization 或非负裁剪。

训练 target 是每个物理细胞的 `log1p(CP10000)` 表达：先在完整的 62,710-gene mapped counts 上完成 library-size normalization 和 `log1p`，然后按照固定 gene panel 取出 5,000 个基因。

5,000-gene panel 只使用 **train split 的 treated physical cells** 确定：在 22,941,936 个训练 treated cells 上计算 62,710 个候选基因的总体方差，包含零表达，按方差从高到低选择前 5,000 个基因；方差相同时按 GeneJEPA gene index 排序。Validation 和 test 数据不参与 gene panel 选择。

| Split | Treated physical cells | Decoder 用途 |
|---|---:|---|
| Train | 22,941,936 | 拟合 Decoder；同时用于确定 5,000-gene panel |
| Validation | 2,841,724 | 计算完整 validation MSE，并选择 best checkpoint |
| Test | 2,856,299 | 保留用于独立测试，不参与训练、panel 或 checkpoint 选择 |
| **合计** | **28,639,959** | 全部为 treated physical cells |

Decoder 拟合中不使用 DMSO cells。它与 Experiment 1 使用同一套冻结 edge split：train、validation、test 的 treated physical cells 互不重叠，且同一 `(cell_line_id, drug)` edge 的所有 dose、plate 和 replicate 保持在同一 split。

---

## 五、三个模型对比

| 项目 | GeneJEPA | ST-A | Decoder |
|---|---|---|---|
| 主要作用 | 把单细胞基因表达编码为通用细胞状态 | 根据 control + drug/dose 预测 treated 状态分布 | 把 treated latent 还原为基因表达 |
| 模型类型 | 自监督 JEPA；Perceiver-style encoder；Student/EMA Teacher | 双向 Transformer-based perturbation model | 逐细胞 MLP decoder |
| 输入 | 稀疏 gene IDs + expression counts | Control latent set + drug/dose vector | 单细胞 GeneJEPA latent |
| 输入维度 | 可变长度 gene set；每个 gene token 768 维，vocabulary 62,710 | `[B,256,768]` + `[B,256,380]` | `[N,768]` |
| 输出 | Cell latent | Predicted treated latent set | Predicted treated gene expression |
| 输出维度 | `[N,768]` | `[B,256,768]` | `[N,5000]` |
| 使用数据 | Tahoe-100M；当前 checkpoint 使用固定 quarter-shard 训练候选池 | Experiment 1 配对的 DMSO/control 与 drug-treated sets | Experiment 1 treated physical cells及其表达 target |
| 数据规模 | 完整数据 95,624,334 cells；训练候选池 23,793,195；正式提取 30,839,089 embeddings | 56,993 conditions；17,174 edges；30,839,089 cached cells | 28,639,959 treated physical cells；5,000-gene targets |
| Loss | JEPA cosine prediction + variance/covariance anti-collapse regularization | Energy distance | MSE |
| 关键训练参数 | 12 blocks、6 heads、batch 92、700,000 train samples/epoch、lr 1e-4、最多 30 epochs | S=256、64 conditions/GPU×2、accumulation=2、effective global batch=256、lr 1e-4、30 epochs | 1,024 cells/GPU×2、global batch=2,048、lr 3e-4、10 epochs、BF16 forward/FP32 MSE |
| 最终模型 | Epoch25 EMA Teacher | ST-A v2 best checkpoint，best_epoch=28 | Decoder v1 best checkpoint，best_epoch=9 |

## 六、数据规模总表

| 模块 | Cells | Genes | Cell lines | Drugs | Doses | Conditions | Edges | Train / validation / test |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GeneJEPA | Tahoe 完整数据 95,624,334；checkpoint 训练候选池 23,793,195；最终正式 embedding 30,839,089 | 62,710 vocabulary | 50（完整 Tahoe） | 380 labels，含 DMSO | N/A：模型不使用 dose | N/A | N/A | 每 epoch 700,000 train samples；10,000 validation samples；无独立 test 配置 |
| ST-A | 30,839,089 cached unique cells：28,639,959 treated + 2,199,130 DMSO | N/A：输入输出均为 768 维 latent | 48 | 379 | 3 | 56,993 | 17,174 | Conditions：45,652 / 5,657 / 5,684；Edges：13,740 / 1,717 / 1,717 |
| Decoder | 28,639,959 treated physical cells；不使用 DMSO | 5,000 outputs；候选 universe 62,710 | 48（train 48，validation/test 各 47） | 379（train 379，validation/test 各 377） | 3 | 来源条件 56,993 | 来源 edges 17,174 | Physical cells：22,941,936 / 2,841,724 / 2,856,299 |

## 七、一句话理解整套系统

GeneJEPA 负责把高维、稀疏的单细胞表达压缩成统一的细胞状态；ST-A 在这个状态空间中学习药物和剂量导致的群体变化；Decoder 再把预测状态翻译回固定 5,000-gene panel 的表达，从而形成从对照细胞到预测处理后基因表达的完整虚拟细胞链路。
