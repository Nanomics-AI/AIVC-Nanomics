# 技术阅读指南

## 1. 这个项目现在预测什么？

当前正式主线预测的是：给定一种细胞背景、药物和剂量，先预测药物处理后的细胞群 latent，再预测该 condition 的 Top20 基因表达变化量。

```text
Tahoe 真实单细胞
  → Author GeneJEPA Epoch49 EMA Teacher
  → 每个细胞一个 768 维 latent
  → ST-A v2 预测 treated latent set
  → D2 比较 control/treated 两个 latent set
  → signed Top20 expression delta
```

本仓库保留的是 Phase II / Phase III 正式实验实际运行过的原始源码，不是为 GitHub 重新整理出的另一套实现。

## 2. 建议阅读顺序

1. Author GeneJEPA 接入：
   - `perturbation_scripts/phase2_author_genejepa.py`
   - `perturbation_scripts/phase1_author_genejepa.py`
   - `perturbation_scripts/author_genejepa_epoch49_hd100_cache.py`
2. Tahoe preprocessing 和 physical-cell cache：
   - `perturbation_scripts/phase2_stav2_data.py`
   - `perturbation_scripts/extract_tahoe_experiment1_full_cache_worker.py`
   - `perturbation_scripts/extract_tahoe_latent_audit_embeddings.py`
   - `genejepa/configs.py`
   - `genejepa/data.py`
3. ST-A v2 Dataset 和 sampler：
   - `perturbation_scripts/phase2_stav2_dataset.py`
   - `perturbation_scripts/tahoe_experiment1_latent_data.py`
4. ST-A v2 模型与 runner：
   - `perturbation_scripts/phase2_stav2_model.py`
   - `perturbation_scripts/run_phase2_stav2.py`
5. D2 data/target：
   - `perturbation_scripts/phase3_set_decoder_data.py`
   - `perturbation_scripts/tahoe_decoder_v1_data.py`
6. D2 模型与 runner：
   - `perturbation_scripts/phase3_set_decoder_model.py`
   - `perturbation_scripts/run_phase3_set_decoder.py`
7. 完整推理与正式评价：
   - `perturbation_scripts/evaluate_phase3_set_decoder.py`

## 3. 关键张量尺寸

单个细胞经过 Author GeneJEPA：

```text
final_norm tokens [512,768]
→ 对 512 个 token 做 mean pooling
→ cell latent [768]
```

一个 condition：

```text
256 cells → [256,768]
```

ST-A v2：

```text
control [B,256,768]
→ predicted treated [B,256,768]
```

D2：

```text
control [B,256,768]
treated [B,256,768]
→ signed Top20 delta [B,20]
```

## 4. Author GeneJEPA 做什么？

这里使用的是外部上游
[BiostateAI/GeneJEPA](https://github.com/BiostateAI/GeneJEPA)，固定 commit：

```text
a2f4d7218b17f2f52cc5f1cc94420c8ef1ae3265
```

正式 checkpoint 是 Epoch49 EMA Teacher。其结构为 24 个 transformer block、12 个 attention head、hidden dim 768、512 个 latent token。正式 extraction 捕获 EMA Teacher 的 `final_norm [B,512,768]`，再执行一次 `mean(dim=1)`，保存 signed float32 `[B,768]`。

Tahoe 输入只执行一次 sentinel handling、gene-token mapping、`log1p` 和 frozen global mean/std normalization。输出 latent 不做 centering、whitening、L2 normalization 或 ReLU。

## 5. set 是怎么抽的？

每个 treated condition 和 matched DMSO control pool 最多缓存 512 个 physical cells。每次训练/评价抽取 256 个。共享 sampler 使用：

```text
seed + epoch + pair_id + side
```

生成稳定 seed，再用 PCG64 在 set 内无放回抽样。不同 epoch 之间可以重合。`phase2_cell_index` 是 locator、Author embedding 与 Top20 expression cache 之间的行对齐契约。

## 6. ST-A v2 做什么？

ST-A v2 的输入是 control latent set `[B,256,768]`。药物使用：

```text
Embedding(379,768)
```

剂量使用：

```text
dose_scaled = log1p(dose_uM) / log1p(5)
```

conditioning 为：

```text
control latent + drug embedding × normalized dose
```

输出是 signed、absolute treated latent set `[B,256,768]`。它不预测 residual。训练 loss 为 Energy distance：`SamplesLoss(loss="energy", blur=0.05)`。

正式训练入口：

```text
perturbation_scripts/run_phase2_stav2.py train
```

## 7. D2 做什么？

D2 不是“对每个单细胞做绝对表达量解码”的 decoder。它同时接收 control set 和 treated set。

两个 set 共用同一个 encoder：

```text
Linear 768→256
TransformerEncoder ×2
  d_model=256
  nhead=8
  FFN=1024
  dropout=0
  norm_first=True
  无 positional encoding
scalar attention pooling
```

得到：

```text
h_control
h_treated
```

然后计算：

```text
h_treated - h_control
```

最后通过 `256→256→128→20` readout，输出 signed Top20 delta。

`phase3_set_decoder_model.py` 原文件同时包含 D1 和 D2，因为它就是正式 validation-only architecture comparison 使用的源码。当前选中的路线是 D2，没有为了 GitHub 新建 D2-only 实现。

正式训练入口：

```text
perturbation_scripts/run_phase3_set_decoder.py train --variant d2
```

## 8. Top20 target 如何产生？

正式 target 复用 `tahoe_decoder_v1_data.py` 中的函数：

```text
全部 mapped raw gene counts
→ 用全部 mapped genes 计算 library size
→ CP10K
→ log1p
→ 按 frozen Top20 panel 取列
```

因此 Top20 panel 不参与 CP10K 分母的计算。GT 是：

```text
GT delta = mean(real treated Top20) - mean(matched real control Top20)
```

## 9. 训练 D2 时为什么不用 ST-A？

D2 训练时使用真实配对数据：

```text
real control latent set
+ real treated latent set
→ D2
→ Top20 delta
```

这样 D2 学习 set-level latent difference 到 expression delta 的映射，不把 ST-A 的预测误差混进 decoder training target。

## 10. 真正完整推理时 ST-A 怎么接回 D2？

```text
real control + drug/dose
→ ST-A v2
→ predicted treated latent set

real control + predicted treated
→ D2
→ Top20 delta
```

正式入口为：

```text
perturbation_scripts/evaluate_phase3_set_decoder.py evaluate
```

同一个 evaluator 还会计算 D1 和 frozen Old Decoder reference。`run_phase1_top20_decoder.py` 之所以保留，是因为正式 evaluator 直接 import 其中的 `build_decoder()`；Old Decoder 只是同评价器 reference，不是当前最终 decoder。

## 11. 为什么代码里还有历史文件名？

当前仓库尽量保留实际正式实验运行过的原始代码，不为了 GitHub 展示重新重构。因此部分依赖仍带有早期实验名称，例如 `hd100`、`audit`、`b0`、`phase1` 或 `decoder_v1`。这些名称对应开发历史；文件本身被保留，是因为当前正式 Phase II/III 源码仍真实调用其中的函数。原样保留可以保证 GitHub 中接受审阅的代码与实际实验代码一致。

两个被保留的原文件还包含只属于早期命令分支的 ARC7 延迟 import。当前 Phase II/III 入口不会执行这些分支，因此独立 ARC7 脚本仍按范围要求排除。正式 Author 路线只调用 `author_genejepa_epoch49_hd100_cache.py` 的 loader/configuration；正式 evaluator 只调用 `run_phase1_top20_decoder.py` 的 `build_decoder()`。代码中的 external `genejepa.train` 会在 `register_author_config_aliases()` 切换 package source 后，从固定 commit 的 Author GeneJEPA checkout 解析。

## 12. 哪些内容不在 GitHub？

Tahoe parquet、condition tables、physical-cell locator plans、Author/ST-A/D2 checkpoint、embedding cache、Top20 expression cache、日志和正式 evaluation outputs 都是外部 runtime artifacts。预期路径和 SHA 见 `docs/provenance.md`。
