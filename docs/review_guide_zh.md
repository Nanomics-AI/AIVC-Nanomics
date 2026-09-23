# AIVC 主线代码阅读指南

这份文档不是实验结果报告，而是一张“代码地图”：说明项目最终要做什么、三个模型模块怎样连接，以及建议按什么顺序阅读代码。

## 1. 这个项目最终想做什么

给定一组未处理的 control 细胞，以及药物和剂量信息，我们希望预测这些细胞经过处理后的基因表达状态。

完整流程是：

```text
Tahoe 单细胞稀疏表达数据
        ↓
GeneJEPA 预处理
        ↓
我们训练的 GeneJEPA Epoch25 EMA Teacher
        ↓
每个细胞得到一个 768 维 latent
        ↓
ST-A 读取 control latent 和药物/剂量
        ↓
预测 treated latent
        ↓
Decoder v1
        ↓
预测处理后的 5000 个基因表达值
```

## 2. 三个模型模块分别做什么

### GeneJEPA：把一个细胞压缩成 768 维表示

一个细胞原本是“哪些基因出现、每个基因有多少 count”的稀疏列表。GeneJEPA 把这份长度不固定的列表编码成一个固定长度的 `[768]` 向量。

模型内部有 512 个 learned latent tokens，每个 token 宽度为 768；它们经过 12 个 transformer block 后再求平均，最终每个细胞只输出一个 `[768]` 向量。这里的 512 是模型内部 token 数，不是一次输入 512 个细胞，也不是为每个细胞保存 `[512,768]`。

### ST-A：预测药物处理后的 latent

ST-A 一次处理一个 256-cell set。它看到 control 细胞的 latent，以及这组细胞共同的药物和剂量特征，然后直接预测处理后细胞所在的 latent 位置。

ST-A 输出仍然是 signed latent，不做 ReLU 截断：

```text
输入 control latent： [B, 256, 768]
药物/剂量特征：       [B, 256, 380]
输出 treated latent： [B, 256, 768]
```

380 维 perturbation feature 由 379 维药物 one-hot 和 1 维标准化后的 `log10(dose_uM)` 组成。同一个 set 的 256 个细胞使用同一份药物/剂量信息。

### Decoder：把 latent 还原成可解释的基因表达

Decoder v1 对每个细胞分别工作：

```text
[768] → [1024] → [1024] → [512] → [5000]
```

输出对应冻结的 5000-gene panel，目标空间是 `log1p(CP10K)`。这一步的 CP10K 是 Decoder 的监督目标处理，不是 GeneJEPA 的输入预处理。

## 3. 一个细胞从输入到输出经历什么

### 3.1 GeneJEPA 输入

真实代码在 [`genejepa/data.py`](../genejepa/data.py)。处理顺序是：

```text
原始稀疏 counts
→ 如果首项是 sentinel，则移除首项
→ 把 Tahoe gene token ID 映射到 GeneJEPA vocabulary
→ log1p(count)
→ 使用冻结的 global mean/std 做标准化
```

这里不会再次做 CP10K，也不会重复 `log1p`。

### 3.2 GeneJEPA 编码

[`genejepa/tokenizer.py`](../genejepa/tokenizer.py) 把 gene identity 和连续表达值组合成 token embedding。

[`genejepa/models.py`](../genejepa/models.py) 让 512 个 learned latent tokens 对细胞的基因 token 做 cross-attention，再经过 transformer、`final_norm` 和 mean pooling，输出一个 768 维 cell embedding。

正式提取使用 Epoch25 EMA Teacher，而不是训练中的 student。入口在 [`tahoe_genejepa_embedding.py`](../perturbation_scripts/tahoe_genejepa_embedding.py)。

### 3.3 embedding cache

为了避免同一个物理细胞反复运行 GeneJEPA，正式流程先把每个 unique cell 提取一次，再按全局 `embedding_index` 写入 `[N,768] float32` cache。

主线脚本按顺序是：

1. [`prepare_tahoe_experiment1_manifests.py`](../perturbation_scripts/prepare_tahoe_experiment1_manifests.py)：固定 condition split、drug vocabulary 和 dose transform。
2. [`plan_tahoe_experiment1_full_cache.py`](../perturbation_scripts/plan_tahoe_experiment1_full_cache.py)：生成稳定 cell locator 和 worker plan。
3. [`extract_tahoe_experiment1_full_cache_worker.py`](../perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py)：可恢复地提取 embedding。
4. [`merge_tahoe_experiment1_full_cache.py`](../perturbation_scripts/merge_tahoe_experiment1_full_cache.py)：按全局 index 合并 worker 输出。
5. [`tahoe_experiment1_latent_data.py`](../perturbation_scripts/tahoe_experiment1_latent_data.py)：mmap cache，并动态无放回抽取 256-cell set。

cache 中的 latent 保持原始 signed float32，不做 centering、whitening 或 L2 normalization。

## 4. GeneJEPA 代码建议怎么看

按下面顺序读最容易：

1. [`genejepa/configs.py`](../genejepa/configs.py)：先看 `d=768`、`latents_L=512`、12 blocks、6 heads 和训练配置。
2. [`genejepa/tokenizer.py`](../genejepa/tokenizer.py)：理解 gene ID 和 expression value 怎样进入同一个 token。
3. [`genejepa/models.py`](../genejepa/models.py)：看 cross-attention、latent transformer、EMA Teacher 和 mean pooling。
4. [`genejepa/data.py`](../genejepa/data.py)：核对 Tahoe 预处理只执行一次。
5. [`genejepa/train.py`](../genejepa/train.py)：看 student、EMA Teacher、loss、optimizer 和 checkpoint 流程。

## 5. ST-A 代码建议怎么看

先读 [`tahoe_experiment1_latent_data.py`](../perturbation_scripts/tahoe_experiment1_latent_data.py)，确认 DataLoader 输出：

```text
ctrl_cell_emb  [B,256,768]
pert_cell_emb  [B,256,768]
pert_emb       [B,256,380]
```

再读 [`run_tahoe_experiment1_st_a.py`](../perturbation_scripts/run_tahoe_experiment1_st_a.py)：

- basal encoder 把 768 维 control latent 投影到 hidden space；
- perturbation encoder 把 380 维 drug/dose 特征投影到相同宽度；
- bidirectional Llama backbone 在 256 个 cell token 上建模；
- `project_out` 直接产生 absolute treated latent；
- 训练目标是真实 treated set，loss 是 Energy distance。

最后看 [`state_genejepa_st_a_compat.patch`](../patches/state_genejepa_st_a_compat.patch)。这个 patch 只解决一件事：允许最终激活为 identity，从而保留 GeneJEPA latent 的负值。

## 6. Decoder 代码建议怎么看

先读 [`tahoe_decoder_v1_data.py`](../perturbation_scripts/tahoe_decoder_v1_data.py)：它通过稳定 locator 找回与 latent 对应的同一个物理细胞，并构造：

```text
原始 sparse counts
→ 对该细胞所有成功映射到 GeneJEPA vocabulary 的基因求 library size
→ 使用完整 mapped-gene library 做 CP10K
→ log1p
→ 最后选择冻结的 5000-gene panel
```

因此，5000-gene panel 只决定 Decoder 最终监督和输出哪些基因，不会决定
CP10K 的 library-size denominator。

再读 [`run_genejepa_decoder_v1.py`](../perturbation_scripts/run_genejepa_decoder_v1.py)，重点看 `build_decoder()`、训练 DataLoader、MSE、checkpoint 和 validation。

冻结的 gene panel 与 contract 位于：

- [`results/genejepa_decoder_v1_gene_panel.csv`](../results/genejepa_decoder_v1_gene_panel.csv)
- [`results/genejepa_decoder_v1_gene_panel.json`](../results/genejepa_decoder_v1_gene_panel.json)
- [`results/genejepa_decoder_v1_contract.json`](../results/genejepa_decoder_v1_contract.json)

## 7. 关键张量尺寸汇总

| 阶段 | 张量尺寸 | 含义 |
| --- | --- | --- |
| GeneJEPA 内部 latent array | `[B,512,768]` | 每个细胞内部的 512 个 learned tokens |
| GeneJEPA cell embedding | `[B,768]` | 对 512 tokens 求平均后的单细胞表示 |
| ST-A control input | `[B,256,768]` | 每个 condition 的 256 个 control 细胞 |
| drug/dose input | `[B,256,380]` | 379-d drug one-hot + 1-d dose |
| ST-A output | `[B,256,768]` | 预测的 treated cell set |
| Decoder input | `[N,768]` | 每个预测细胞的 latent |
| Decoder output | `[N,5000]` | 预测的 5000-gene `log1p(CP10K)` 表达 |

## 8. 最短阅读路线

如果只想先抓住主线，按这个顺序即可：

```text
README.md
→ genejepa/configs.py
→ genejepa/data.py
→ genejepa/models.py
→ tahoe_genejepa_embedding.py
→ tahoe_experiment1_latent_data.py
→ run_tahoe_experiment1_st_a.py
→ tahoe_decoder_v1_data.py
→ run_genejepa_decoder_v1.py
```

数据、embedding cache 和 checkpoint 不在 GitHub。它们的本地路径和 SHA-256 见 [`provenance.md`](provenance.md)。完整的精简前代码历史仍保存在 `project-code-audit-20260923` 分支。
