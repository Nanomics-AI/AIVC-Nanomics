# GeneJEPA 复现任务交接说明（给 Codex）

> 项目根目录：`/mnt/c/SH/AIVC/GeneJEPA-main`  
> 环境：Windows 11 + WSL，使用 `uv`  
> 当前阶段：数据准备已经完成，准备开始修改并复现实验。

## 1. Codex 开始前先做什么

请先阅读：

- `AGENTS.md`
- `README.md`
- `genejepa/configs.py`
- `genejepa/data.py`
- `genejepa/models.py`
- `genejepa/train.py`
- `genejepa/callbacks.py`

先审计、再修改，不要根据经验猜参数。

第一次改代码前，请先明确：

- “number of latent blocks / D”在当前代码里的准确参数名和位置。
- attention heads 的准确参数名和位置。
- 训练入口、DataConfig / ExperimentConfig / model 的构造流程。
- 是否存在自动加载 checkpoint、resume、预训练权重的逻辑。
- validation loss 的准确 metric 名称、计算位置和 logging 位置。
- best checkpoint 是按哪个 metric 保存。
- `train_samples`、数据文件数、epoch/step 数之间的关系。
- validation split 如何划分，缩小训练数据时如何保证 validation 逻辑不被破坏。
- 首次训练是否会生成 `global_stats.json`，如果会，确认只读取本地 Tahoe 数据。

先给出审计报告和计划修改清单，再实施。

## 2. 当前项目状态

目录大体如下：

```text
GeneJEPA-main/
├── AGENTS.md
├── README.md
├── download Tahoe datasets/
│   ├── download_tahoe_watchdog_v3.py
│   └── verify_tahoe.py
├── genejepa/
│   ├── callbacks.py
│   ├── configs.py
│   ├── data.py
│   ├── models.py
│   ├── tokenizer.py
│   └── train.py
├── hf_data_cache/
│   ├── data/
│   │   └── data/
│   │       ├── train-00000-of-03388.parquet
│   │       ├── ...
│   │       └── train-03387-of-03388.parquet
│   ├── local_file_manifest.json
│   ├── tahoe_download_plan.json
│   ├── tahoe_download_state.json
│   └── ...
├── pyproject.toml
└── uv.lock
```

Tahoe-100M 已经完整下载到本地。

数据准备阶段使用过两个辅助脚本：

```text
download Tahoe datasets/download_tahoe_watchdog_v3.py
download Tahoe datasets/verify_tahoe.py
```

后续训练阶段不要因为没有必要而修改它们。

## 3. 已完成事项

以下已经完成，不要重复做：

- Tahoe-100M 全量 parquet 已下载。
- 3388 个 `data/*.parquet` shard + `gene_metadata.parquet` 已就位。
- 已生成 `hf_data_cache/local_file_manifest.json`。
- 已做数据完整性验证。
- 已确认 GeneJEPA 的 `prepare_data()` 能识别本地 manifest，因此不会重新下载。

所以：

**不要重新下载 Tahoe-100M。**

**不要删除、移动、重命名 `hf_data_cache` 中的数据。**

项目应从：

```bash
cd /mnt/c/SH/AIVC/GeneJEPA-main
```

启动训练。

如果后续训练代码试图重新大规模下载 Hugging Face 数据，应先停止，检查工作目录和 manifest，而不是重新下载。

## 4. 复现实验目标

目标不是加载作者公开权重做 inference，而是：

**从随机初始化开始，自己训练一套缩小版 GeneJEPA，完整跑通 train + validation。**

当前指导要求：

- 使用原工作大约 **1/4 的训练数据**
- latent blocks / D：目标 `24 -> 12`
- attention heads：目标 `12 -> 6`
- **不能使用作者预训练权重**
- **不能用作者 checkpoint 初始化**
- 自己产生 checkpoint
- 重点关注 **validation loss curve**
- 先 smoke test，再正式训练

注意：Codex 必须先从本地代码确认实际默认值和准确参数名，不能直接假设。

## 5. weights / checkpoints 约束

希望流程是：

```text
随机初始化
    ↓
训练 Student / Predictor
    ↓
Teacher 按原 GeneJEPA EMA 逻辑更新
    ↓
产生我们自己的 checkpoint
```

禁止：

```text
作者 checkpoint / 作者权重
    ↓
load_state_dict / load_from_checkpoint
    ↓
继续训练
```

请审计：

- `ckpt_path`
- `resume`
- `load_from_checkpoint`
- `torch.load`
- Hugging Face 权重加载
- callback 自动恢复逻辑

可以保留“从我们自己的 checkpoint 恢复”的能力，但 fresh run 默认必须随机初始化。

## 6. “1/4 数据”尚未最终决定如何实现

不要删除 3/4 parquet，也不要未经分析就直接取前 847 个文件。

先检查 `genejepa/data.py`，明确：

- 一个 parquet 大约多少 cells
- Dataset 是否 streaming
- 是否用 `train_samples` 控制训练样本量
- 每个 epoch 是否重新 shuffle
- validation 使用哪些文件
- train/val 是否按文件切分
- 取 1/4 samples 与取 1/4 files 的区别
- 哪种最接近“训练数据规模缩成原工作 1/4”的实验意图

优先原则：

- 如果已有 `train_samples` / `max_samples` / `num_samples` / `steps_per_epoch` 一类可靠参数，优先通过配置限制训练样本量。
- 如果只能按 parquet 控制，validation 必须保持固定。
- 训练文件子集必须可复现，使用固定 seed。
- 不要默认“取前 1/4 文件”而不解释原因。

先给建议和理由，再修改。

## 7. 模型规模修改目标

确认代码参数后实现：

```text
latent blocks / D = 12
attention heads    = 6
```

其他参数先尽量保持作者默认值。

如果因为实现约束必须连带修改：

- hidden dimension
- latent dimension
- head dimension
- predictor dimension
- batch size
- gradient accumulation

必须解释原因，不要静默修改。

特别检查类似：

```text
embedding_dim % num_heads == 0
```

这样的约束。

## 8. 先做 smoke test

不要一上来直接跑 1/4 数据长训练。

smoke test 要验证完整链路：

```text
本地 Tahoe parquet
      ↓
DataLoader
      ↓
tokenizer / preprocessing
      ↓
context / target masking
      ↓
Student Encoder
      ↓
Predictor
      ↓
Teacher Encoder
      ↓
JEPA loss + 防 collapse 项
      ↓
backward
      ↓
Student + Predictor 更新
      ↓
Teacher EMA
      ↓
validation
      ↓
checkpoint
```

smoke test 使用：

- 很少 samples / steps
- 1 个或很少 epoch
- 小 batch
- 保留 validation
- 保留 checkpoint 和日志

必须确认：

- 读的是本地数据
- 没有重新下载
- forward 正常
- loss 非 NaN/Inf
- backward 正常
- teacher EMA 正常
- validation 能跑
- checkpoint 能保存
- fresh run 没有加载旧权重

## 9. validation 是重点

至少保留/确认：

- `train_loss`
- `val_loss`
- JEPA prediction loss
- variance regularization
- covariance regularization
- 如果现有代码有 embedding std / variance / collapse metric，也保留
- learning rate
- epoch
- global step

最终需要能够画：

```text
x-axis: epoch 或 global step
y-axis: validation loss
```

最好同时保存 train loss curve。

优先复用现有 TensorBoard / CSV / W&B logger。若现有代码无法方便导出，可加轻量 CSV logger，但不要为了画图大改训练框架。

## 10. checkpoint 要求

正式实验至少希望保存：

- last checkpoint
- best validation checkpoint
- Student weights
- Predictor weights
- Teacher / EMA state
- optimizer state
- scheduler state（若有）
- epoch / global_step

如果当前 Lightning checkpoint 已自动包含这些，不要重复造轮子。

请确认 best checkpoint 是否按 validation loss 保存。

## 11. 实验可复现性

正式实验请记录：

```text
random seed
数据子集规则
实际 train samples
validation 数据规模
D / latent blocks
num_heads
hidden / latent dim
batch size
gradient accumulation
learning rate
weight decay
epochs
optimizer
scheduler
EMA decay
masking 参数
loss 权重
precision
GPU 数量
```

运行开始时最好保存完整 config。

建议每个实验使用独立输出目录，例如：

```text
outputs/
└── genejepa_d12_h6_quarter/
    ├── config.json
    ├── checkpoints/
    ├── logs/
    └── metrics.csv
```

具体目录风格可按项目现有实现调整。

## 12. global_stats.json

如果第一次训练出现 `global_stats.json not found`，并开始读取 Tahoe parquet 计算 mean/std：

这通常是本地 normalization statistics 计算，不等于重新下载。

可以让它完成，但请确认：

- 数据来自本地 parquet
- 不会触发全量重新下载
- 生成后能被后续运行复用

## 13. 修改原则

1. 最小修改。
2. 保留作者原始训练逻辑。
3. 实验参数尽量通过 config / CLI 暴露，不要散落硬编码。
4. 不破坏全量原配置的运行能力。
5. 新增实验配置优于直接改死默认值。
6. 不动已验证的 Tahoe parquet。
7. 不重新下载数据。
8. 不加载作者预训练权重。
9. 每次关键修改说明“改了什么、为什么”。
10. 正式长训练前必须 smoke test。

如果要改多个文件，先列计划，例如：

```text
configs.py
- 增加 quarter reproduction 配置

data.py
- 增加可复现的数据量限制
- 保持 validation split 不变

train.py
- 显式 fresh / resume 行为
- 保存配置

callbacks.py
- 确认 best val checkpoint 和 metrics
```

## 14. Codex 推荐执行顺序

### Phase A：代码审计

阅读：

```text
AGENTS.md
README.md
genejepa/configs.py
genejepa/data.py
genejepa/models.py
genejepa/train.py
genejepa/callbacks.py
```

输出：

1. 当前训练入口
2. 当前默认模型配置
3. D / heads 对应参数
4. 当前 train/val 数据划分
5. 数据量控制机制
6. checkpoint / resume 行为
7. val_loss 记录与 best checkpoint 行为
8. 需要修改的文件清单

先不要跑长训练。

### Phase B：实现复现实验配置

目标：

```text
D=12
heads=6
1/4 training data
random initialization
fixed validation
own checkpoints
metrics logging
```

### Phase C：静态检查

至少：

```bash
uv run python -m compileall genejepa
```

并做必要的 import / config 构造检查。

### Phase D：smoke test

记录：

- 实际命令
- GPU 显存占用
- train loss
- val loss
- checkpoint
- metrics/logs
- 是否重新下载
- 是否有 NaN/Inf

### Phase E：正式 1/4 数据训练

确认 smoke test 后再跑。

重点保存：

```text
validation loss curve
train loss curve
best checkpoint
last checkpoint
完整 config
```

## 15. 不确定事项的处理原则

以下不要自行拍脑袋决定：

- “1/4 数据”按 cells、samples、files 还是 steps
- 是否改 batch size
- validation 是否也缩成 1/4
- 是否改 hidden dim
- 是否改 masking ratio
- 是否改 EMA decay
- 是否改 loss 权重
- 是否使用作者 checkpoint

遇到这些问题：

**先根据代码给出分析，再询问用户。**

默认倾向：

- validation 保持原逻辑和固定规模
- 只缩训练数据
- 模型主要只改 D 与 heads
- 其余尽量保持原设置
- 不加载任何作者权重

## 16. 给 Codex 的第一条任务

请先不要修改代码，也不要启动长训练。

阅读 `AGENTS.md`、`README.md`、`genejepa/configs.py`、`genejepa/data.py`、`genejepa/models.py`、`genejepa/train.py`、`genejepa/callbacks.py`，然后给我一份“GeneJEPA 复现实验代码审计报告”，明确：

1. D=24 和 heads=12 在当前代码中的准确参数及位置；
2. 应如何改成 D=12、heads=6；
3. 当前训练/验证 split 的实现；
4. “约 1/4 原训练数据”最合理的代码实现方式；
5. 如何保证 fresh run 不加载作者权重或旧 checkpoint；
6. validation loss 的计算和记录位置；
7. best checkpoint 是否按 val loss 保存；
8. smoke test 的最小安全运行方案；
9. 预计需要修改哪些文件。

审计完成后先等我确认，再进行正式代码修改。

## 17. 一句话总结

**数据阶段已经完成；接下来只做：基于本地 Tahoe-100M，从随机初始化开始，用约 1/4 训练数据、D=12、heads=6 复现 GeneJEPA，并重点观察 validation loss curve。**
