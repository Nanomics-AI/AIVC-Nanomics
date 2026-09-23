GeneJEPA 复现实验过程记录（内部）

1. 复现实验目标

本次工作的目标不是直接加载 GeneJEPA 作者发布的预训练权重进行推理，而是从随机初始化开始，自行训练一套缩小规模的 GeneJEPA，并完整跑通训练、验证和 checkpoint 保存流程。

根据复现要求，最终确定的主要实验约束为：

- 数据：Tahoe-100M；
- 训练数据规模：原 training pool 的固定 25%；
- GeneJEPA latent blocks：24 → 12；
- attention heads：12 → 6；
- embedding dimension d=768 保持不变；
- latent 数量 L=512 保持不变；
- 不加载作者 checkpoint；
- 不使用作者预训练权重初始化；
- 从随机权重开始训练；
- 重点观察 validation loss curve；
- 在正式训练前先完成单卡、双卡和小规模 smoke test。
这一实验原则在前期交接文档中也明确为：从随机初始化开始，用约 1/4 训练数据、D=12、heads=6 复现 GeneJEPA，并重点关注 validation loss。

---

2. Tahoe-100M 数据下载阶段
2.1 遇到的问题：数据规模大、网络不稳定、重新连接代价高

Tahoe-100M 由大量 parquet shard 构成。GeneJEPA 使用的数据目录最终包含：

- 3388 个 data/*.parquet；
- gene_metadata.parquet；
- 本地 local_file_manifest.json。
直接使用官方 prepare_data() 下载时，主要遇到两个问题：

1. 网络速度不稳定，需要能够自动判断长时间低速并重新连接；
2. 每次程序重启后，不希望再次从第 1 个文件开始逐个执行 hf_hub_download() 检查。
因此没有单纯依靠一次性的官方自动下载，而是在官方 DataModule 的基础上增加了下载 watchdog。

---

2.2 下载脚本逐步改进

最终形成了 download_tahoe_watchdog_v3.py。

主要改动包括：

### 断点记录

程序维护：

tahoe_download_plan.json
tahoe_download_state.json
tahoe_download_status.json

只有在：

- hf_hub_download() 正常返回；
- 最终文件真实存在；
- 文件大小有效；
- 如果远端 size 可用，则本地 size 与远端一致；
这些条件满足后，才把该文件登记为“完整下载”。

这样重新连接时可以直接从：

next_data_index

继续，而不是重新从第一个 parquet 文件检查。

### 下载速度监控

最终确定的低速判断逻辑为：

每 15 秒计算一次实际新增数据对应的平均速度

连续 20 个 15 秒区间
= 连续 5 分钟

每个区间都 < 1 MiB/s
→ 自动断开并重连

只要任意一个区间 >= 1 MiB/s
→ 连续低速计数立即清零

而且只有真正处于 downloading 阶段时才测速，在本地文件检查、已有文件跳过等阶段不会错误触发低速重连。

最终下载程序能够：

异常断开
  ↓
记录当前完成位置
  ↓
重新连接
  ↓
直接从下一个未完成 shard 继续

而不是重复遍历几千个已经完成的文件。

---

3. 数据完整性验证
3.1 建立独立验证脚本
下载完成以后，没有直接开始训练，而是增加：
verify_tahoe.py

检查内容包括：

- manifest 是否存在；
- data_files 数量是否为 3388；
- metadata 是否存在；
- 下载 state 与 manifest 是否一致；
- parquet footer 是否可读取；
- deep 模式下读取 parquet row groups / 数据页。
验证脚本明确把 GeneJEPA 当前预期的 parquet shard 数固定为：

EXPECTED_GENEJEPA_DATA_FILES = 3388

---

3.2 发现一个损坏的 parquet

虽然文件已经存在于磁盘上，但进一步完整扫描 validation parquet 时发现：

train-00000-of-03388.parquet

无法正常反序列化 parquet page，出现类似：

Couldn't deserialize thrift
TProtocolException: Invalid data
Deserializing page header failed

说明：

“文件存在”或者“文件大小看起来正常”，并不能绝对证明 parquet 内部数据页完整。

处理方式是：

1. 定位损坏 shard；
2. 只重新下载这一份 parquet；
3. 替换原文件；
4. 再次执行 parquet 完整扫描；
5. 确认 validation 数据可以正常读取。
这一阶段也确定了一个后续原则：

长训练前应优先确认数据完整性，而不是等训练数小时后才因为坏 shard 崩溃。

---

4. 防止 GeneJEPA 重复下载数据

下载完成后生成：

hf_data_cache/local_file_manifest.json

随后确认：

dm.prepare_data()

检测到 manifest 后会直接：

Local file manifest found...
Skipping downloads.

所以后续训练不再重新访问 Hugging Face 下载整个 Tahoe-100M。当前正式训练日志也确认继续使用本地 manifest。

因此从数据阶段结束以后，正式训练全部直接读取：

hf_data_cache/

中的本地 parquet。

---

5. 第一次训练前的模型缩减

原 GeneJEPA 模型规模较大，因此按照复现要求修改：

d = 768
latents_L = 512

blocks_D = 12
heads_h = 6

主要只缩减：

latent Transformer blocks
24 → 12

attention heads
12 → 6

其他关键结构参数尽量保持作者设置不变，例如：

mask_ratio = 0.45
EMA start decay = 0.992
EMA end decay = 0.9995
EMA warmup steps = 2000

predictor_depth = 3

这样可以减少实验变量，让本次实验主要研究：

缩小模型
+
缩小训练数据

而不是同时修改大量超参数。

---

6. Global normalization statistics

第一次训练时还没有：

hf_data_cache/global_stats.json

因此 GeneJEPA 从训练文件中读取约 100 万个 cell 进行全局统计。

最终计算得到：

mean ≈ 0.8255
std  ≈ 0.3135

整个统计过程涉及约：

1,250,056,733 个 expression values

随后保存为：

hf_data_cache/global_stats.json

后面的训练全部直接加载该文件，不再重复计算。

正式 25% 数据实验中，我们讨论过是否应该只使用 25% subset 重新计算 global mean/std。

最终决定：

保留原有的 global normalization statistics。

原因是它只作为固定的数据预处理统计量使用；这样不同实验之间 normalization 保持一致，只改变真正参与模型训练的数据规模。

因此当前实验口径应准确描述为：

模型训练使用固定 25% Tahoe training subset；表达值 normalization 使用此前计算得到的固定 Tahoe global statistics。

---

7. 单 GPU smoke test

正式处理双 GPU 之前，首先用单张 RTX A6000 测试：

CUDA_VISIBLE_DEVICES=0

smoke 配置大致为：

batch_size = 92
train_samples = 1840
max_epochs = 1
validation_num_batches = 4

D = 12
heads = 6

单卡有效 batch size 为：

92 × accumulate_grad_batches 2
= 184

训练能够完成：

forward
backward
validation
checkpoint

证明：

GeneJEPA 模型本身、Tahoe 数据、bf16、loss、Teacher EMA、validation 以及 checkpoint 流程在单卡环境下能够正常工作。

单卡成功以后，问题才被缩小到“双卡通信和 DDP”。

---

8. 双 GPU NCCL 问题

硬件环境：

2 × NVIDIA RTX A6000
约 48 GB VRAM / GPU

PyTorch：

torch 2.7.1+cu126
CUDA available = True

但是进一步检查发现：

torch.cuda.can_device_access_peer(0,1) = False
torch.cuda.can_device_access_peer(1,0) = False

同时：

nvidia-smi topo -m

无法正常输出 topology matrix。

后来进一步确认服务器本身也没有安装物理 NVLink Bridge。

因此当前两张 A6000 并不能通过 NVLink/P2P 直接互联。

---

8.1 NCCL 最小测试失败

为了把“GeneJEPA 问题”和“GPU 通信问题”分离，我们单独编写了：

/tmp/test_nccl.py

只执行最简单的：

GPU0: x = 1
GPU1: x = 2

NCCL all_reduce

预期：
GPU0 = 3
GPU1 = 3

默认 NCCL 初始化失败：

NCCL WARN Cuda failure 999 'unknown error'
include/alloc.h:237

即使只关闭：

NCCL_CUMEM_HOST_ENABLE=0

仍然失败。

这说明问题不是 GeneJEPA 模型本身，因为最简单的 NCCL collective 都无法完成。

---

8.2 NCCL 最终解决方案

进一步关闭 NCCL device-side cuMem allocation：

NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0

之后 NCCL communicator 能够正常初始化，并完成：

1 + 2 → all_reduce → 3

因此当前双卡训练固定使用：

NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0

作为当前：

WSL
+
PyTorch 2.7.1
+
NCCL 2.26.x
+
2 × A6000

环境的兼容性 workaround。

当前没有使用 NVLink/P2P，而由 NCCL 使用可用的 fallback 通信路径。

---

9. 双卡 DDP 的第二个问题：Hugging Face IterableDataset.shard()

NCCL 解决以后，Lightning DDP 已经能够初始化，但又出现新的报错：

AttributeError:
'IterableDataset' object has no attribute 'shard'

问题来自原来的 DDP 数据分片代码：

hf_dataset = hf_dataset.shard(
    num_shards=world_size,
    index=rank
)

实际环境中的 Hugging Face：

datasets = 2.19.2

这个版本的：

datasets.IterableDataset

没有公开 .shard() 方法。

---

9.1 解决方式：在 parquet 文件列表层做 DDP 分片

不再创建 IterableDataset 后调用：

hf_dataset.shard(...)

而是先按 rank 分 parquet 文件：

file_list = file_list[rank::world_size]

再分别：

load_dataset(
    "parquet",
    data_files=file_list,
    streaming=True
)

逻辑变成：

training parquet files
        ↓
DDP rank-level split
     ↙       ↘
rank 0      rank 1
  ↓           ↓
GPU0         GPU1

最终双卡 smoke test 能够正常完成训练和验证。当前 data.py 也继续采用这种 rank-level 文件分片方法。

---

10. 正式定义“25% training data”

最开始需要解决的一个实验设计问题是：

“1/4 数据”究竟应该理解为少跑 1/4 steps，还是实际固定 1/4 数据集？

最终决定：

真实固定 training data pool，而不是仅仅减少 train_samples。

当前 Tahoe 划分为：

总 parquet shards = 3388

validation = 16
training pool = 3372

所以 training pool 的严格 25% 为：

3372 / 4
= 843 shards

因此：

16 个 validation shards
保持固定

从 3372 个 training shards 中
用 seed=42 一次性选出 843 个

并把结果保存为：

hf_data_cache/train_subset_quarter_seed42.json

之后所有正式实验都读取这同一份 manifest，而不是每次启动时重新随机抽 25%。

data.py 会验证：

- validation shard 是否仍与冻结 manifest 一致；
- selected training files 是否全部属于 training pool；
- 是否有重复 shard；
- 最终 training subset 是否正确加载。
当前正式训练日志已经确认：

Fixed training subset enabled:
843/3372 shards (25.00%)

---

11. 修复训练数据 shuffle 的可重复性

原代码使用：

shuffle_seed = int(time.time()) + rank

这意味着：

即使实验都写着 random seed=42，只要启动时间不同，训练数据顺序仍会改变。

对于正式复现不够严谨。

因此改为：

shuffle_seed = random_seed + epoch

即：

epoch 0 → seed 42
epoch 1 → seed 43
epoch 2 → seed 44
...

并设置：

reload_dataloaders_every_n_epochs=1

保证每个 epoch 重新创建 DataLoader，从而让新的 epoch seed 真正生效。

当前代码会显式输出：

Train shuffle:
epoch=X
seed=42+X
rank=...

随后专门运行 2-epoch smoke test，确认：

epoch 0 → seed 42
epoch 1 → seed 43

并且两个 epoch 均正常完成 train、validation 和 checkpoint。

---

12. DataLoader worker 性能问题

正式训练初期观察：

GPU utilization：
100% → 0% → 100% → 0%

同时日志提示：

train_dataloader does not have many workers

进一步检查发现：

虽然：

DataConfig.num_workers = 8

但实际 DataLoader 被代码强制设置成：

num_workers = 0

也就是说 GPU 在训练过程中可能需要等待 CPU 读取、解析和 collate 下一批 Tahoe 数据。

---

13. Hugging Face worker 分片兼容性

当前：

datasets version = 2.19.2

检查发现：

IterableDataset.has shard = False

但 Hugging Face 自身的 IterableDataset.__iter__() 已经会读取：

get_worker_info()
worker_info

因此：

HF 2.19.2 会自己处理 DataLoader worker 的 streaming shard 分配，没有必要再手动执行 dataset_instance.shard()。

所以删除了自定义 worker .shard() 逻辑，让 Hugging Face 内部负责 worker-level sharding。当前 Tahoe100MDataset.__iter__() 就采用这一方式。

---

14. 将实际 worker 数改为运行时参数

为了避免每次测试不同 worker 数都修改 Python 文件，增加：

GENEJEPA_TRAIN_WORKERS=N

例如：

GENEJEPA_TRAIN_WORKERS=2

实际 DataLoader worker 由环境变量决定。

而：

DataConfig.num_workers = 8

仍然保留。

这里有一个当前代码设计上的特殊点：

DataConfig.num_workers=8

不仅代表 worker 配置，还参与：

min_files_for_parallelism = num_workers * 2

从而影响 validation shard 数。

因此目前：

DataConfig.num_workers = 8

主要继续用于保持：

validation = 16 shards

而真正训练时的 DataLoader worker 数由：

GENEJEPA_TRAIN_WORKERS

控制。

这是目前代码中两个需要明确区分的概念。

---

15. Worker benchmark

为了确定实际最佳 worker 数，不直接凭经验使用 8，而是进行了：

0 / 2 / 4 / 8 workers

四组 benchmark。

每组统一：

200 raw training batches
1 epoch
2 GPUs
batch_size=92
相同模型
相同数据
相同随机种子

结果大致为：

Training workers200 batch 时间稳态吞吐


0
≈10:01
≈0.379 batch/s
2
≈8:50
≈0.439 batch/s
4
≈9:04
≈0.435 batch/s
8
≈9:05
≈0.431 batch/s

从 0 → 2 后吞吐提高约 16%，说明之前 DataLoader 确实是 GPU 等待的重要原因之一。

但继续增加到 4、8 并没有进一步提高速度。

因此正式训练选择：

GENEJEPA_TRAIN_WORKERS=2

worker=2 benchmark 使用了固定的 843/3372 subset 和双 GPU 配置。
worker=4 和 worker=8 也进行了相同条件下的测试。
worker=0 则作为 baseline。

---

16. persistent\_workers 与跨 epoch reload

多 worker benchmark 时 Lightning 提示：

persistent_workers=True
+
reload_dataloaders_every_n_epochs=1

可能导致长期训练不稳定。

正式训练又必须保留：

reload_dataloaders_every_n_epochs=1

因为需要实现：

epoch 0 seed=42
epoch 1 seed=43
...

因此最终修改为：

persistent_workers=False
prefetch_factor=2

然后专门运行 2-epoch worker=2 smoke test。

结果确认：

Epoch 0
worker=2
seed=42

Epoch 1
重新创建 DataLoader
worker=2
seed=43

两轮均正常结束。

至此 DataLoader 方案冻结。

---

17. 一次 validation split 不一致问题

在 worker 调试过程中曾将：

DataConfig.num_workers

误改为 2。

由于原代码存在：

min_files_for_parallelism
=
DataConfig.num_workers * 2

validation 于是从：

16 shards

变成：

4 shards

固定的 25% manifest 检查立即报错：

Validation shard list does not match
the frozen subset manifest.

这个报错反而证明数据冻结检查正常工作，没有允许实验数据划分悄悄改变。

最终明确：

DataConfig.num_workers = 8
→ 保持 16 validation shards

GENEJEPA_TRAIN_WORKERS = 2
→ 实际训练 DataLoader workers

两者不能混淆。

---

18. 最初 Formal Run 1 配置（后续已调整）

经过全部 smoke test 和性能测试后，Formal Run 1 最终配置为：

## 模型

d = 768
latents_L = 512
blocks_D = 12
heads_h = 6

mask_ratio = 0.45

EMA:
start = 0.992
end = 0.9995
warmup = 2000 steps

## 训练

batch_size = 92 / GPU
GPU 数 = 2
gradient accumulation = 2

effective batch size
= 92 × 2 × 2
= 368

learning_rate = 1e-4
weight_decay = 2e-4
max_epochs = 50

## 数据

固定 validation = 16 shards

原 training pool = 3372 shards

正式 training subset
= 843 shards
= 25%

subset:
train_subset_quarter_seed42.json

## 每 epoch 训练预算

train_samples = 1,000,000

双卡对应：

5435 raw batches / epoch

2718 optimizer updates / epoch

50 epochs
→ 135900 optimizer updates

正式训练日志已确认这一 schedule。

## Validation

validation_num_batches = 55

大约对应：

92 × 2 GPUs × 55
≈ 10,120 validation cells

即与目标的约 10k validation samples 数量级基本对应。

## DataLoader

GENEJEPA_TRAIN_WORKERS = 2

persistent_workers = False
prefetch_factor = 2

validation num_workers = 0

## 计算

2 × NVIDIA RTX A6000
bf16 mixed precision

## NCCL

NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0

## W&B

当前使用：

WANDB_MODE=offline

## Checkpoint

正式目录：

checkpoints/
genejepa_quarter_d12_h6_seed42_run1

并在正式启动前确保不存在需要自动 resume 的旧 checkpoint，从随机初始化开始训练。

---

19. Formal Run 1 启动命令

当前正式训练使用：

GENEJEPA_TRAIN_WORKERS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_CUMEM_ENABLE=0 \
NCCL_CUMEM_HOST_ENABLE=0 \
WANDB_MODE=offline \
uv run -m genejepa.train \
2>&1 | tee logs/genejepa_quarter_d12_h6_seed42_run1.log

当前日志已经确认：

2 devices detected

effective batch size = 368

16 validation shards

3372 original training shards

843/3372 fixed training subset
= 25%

5435 raw batches/epoch

2718 optimizer updates/epoch

135900 total optimizer updates

55 validation batches

并已经正式进入训练。

---

20. 到8.13为止实际解决的问题总结

整个 GeneJEPA 复现过程中已经依次处理了：

1. Tahoe-100M 下载速度不稳定；
2. 网络断开后重复从第一个文件检查；
3. 下载进度无法可靠记录；
4. 下载测速把“检查已有文件”误判为低速；
5. 下载完成以后缺少系统性 parquet 完整性检查；
6. 检出并重新下载损坏 parquet；
7. 防止 GeneJEPA 训练阶段重新下载 Tahoe；
8. 第一次生成 global normalization statistics；
9. 完成单 GPU smoke test；
10. 双 GPU NCCL CUDA 999；
11. WSL 下两卡 P2P 不可用；
12. 使用 NCCL_CUMEM_ENABLE=0 workaround 打通 NCCL；
13. Hugging Face IterableDataset.shard() API 不兼容；
14. 改成 DDP rank-level parquet 文件分片；
15. 成功完成双 GPU smoke test；
16. 明确定义“25% training data”；
17. 固定 843 个 training shards；
18. 固定 16 个 validation shards；
19. 修复依赖启动时间的 shuffle；
20. 实现 seed + epoch 可复现 shuffle；
21. 验证跨 epoch DataLoader reload；
22. 发现 num_workers=0 带来的 GPU 等数据问题；
23. 修复 HF streaming 多 worker 兼容；
24. 增加 GENEJEPA_TRAIN_WORKERS 运行时参数；
25. benchmark 0/2/4/8 workers；
26. 最终确定 training workers=2；
27. 处理 persistent_workers 与 epoch reload 的兼容问题；
28. 运行 2-epoch 跨 epoch smoke；
29. 恢复 50 epoch Formal Run 1；
30. 当前正式进入 GeneJEPA 复现训练阶段。
---

21. 截至 2026-08-13 的阶段状态（历史记录）

截至目前，环境和训练基础设施已经基本冻结。

状态可以概括为：

Tahoe 数据下载                 ✅
数据完整性验证                 ✅
本地 manifest                  ✅
单卡训练                       ✅
双卡 NCCL                      ✅
双卡 DDP                       ✅
固定 25% training subset       ✅
固定 validation                ✅
随机初始化                     ✅
可复现 epoch shuffle           ✅
DataLoader worker benchmark    ✅
跨 epoch worker 验证           ✅
正式 50 epoch Run 1            ▶ 正在进行

接下来工作的重点已经不再是“代码是否能运行”，而应该转向模型训练本身。

---

22. 后续重点记录内容

Formal Run 1 后续需要重点保存和分析：

22.1 Validation loss curve

每个 epoch 记录：

epoch
val_loss

最终绘制完整 50 epoch 曲线。

重点观察：

- validation loss 是否持续下降；
- 哪个 epoch 达到最低值；
- 是否进入平台期；
- 后期是否回升；
- 是否有明显过拟合；
- EMA warmup 前后曲线是否发生变化。
22.2 Train loss curve

同步记录：

train/loss
train_loss/sim

观察 train 与 validation 是否一致。

22.3 EMA warmup

当前：

ema_warmup_steps = 2000

而正式训练：

2718 optimizer updates / epoch

因此第一个 epoch 内就会跨过 Teacher EMA warmup。

需要特别关注：

前 2000 step
和
warmup 后

模型行为是否存在明显变化。

22.4 Checkpoint

需要保存：

best checkpoint
last checkpoint

并记录：

best checkpoint 对应 epoch
best val_loss

22.5 异常监控

正式训练过程中真正需要警惕的是：

NaN
Inf
CUDA OOM
NCCL error
DataLoader worker exited
parquet read error
loss 突然异常爆炸

普通 Lightning 性能 warning 暂时不作为中断训练的理由。

---

23. 截至 2026-08-13 的阶段性一句话总结（历史记录）

本次工作已经从最初的：

“先让 GeneJEPA 能跑起来”

逐步完成了：

Tahoe-100M 本地化
→ 数据完整性验证
→ 单卡训练
→ 双卡 NCCL 修复
→ DDP 数据分片修复
→ 固定 25% 数据集
→ 可复现 shuffle
→ DataLoader 性能优化
→ 多 epoch 验证

目前已经进入：

> 2×RTX A6000、D=12、heads=6、固定 Tahoe training pool 25%、随机初始化、50 epochs 的 GeneJEPA Formal Run 1。

从现在开始，实验的核心评价对象是 validation loss curve 和最终 checkpoint，而不再是训练环境是否能够正常启动。
---

24. 8.14重新评估训练时长：没必要把 25% dataset 直接改成 10%

最初 Formal Run 1 的配置为：

固定 25% training subset
train_samples = 1,000,000 / epoch
max_epochs = 50

启动后根据实际速度估算，完整跑完约需要两周。领导认为耗时过长，最初提出：

是否可以把训练数据从 25% 直接缩减到 10%。

在进一步梳理训练代码后，发现当前 GeneJEPA 工程里的 epoch 并不是“完整遍历一次训练数据集”的传统定义，而是通过 train_samples 人为限制每个 epoch 的训练样本预算。

代码中的核心关系为：

steps_per_epoch = ceil(
    train_samples / (batch_size * num_devices)
)

然后 Trainer 使用：

limit_train_batches = steps_per_epoch

因此：

dataset/subset 比例
    ↓
决定模型允许从多大的数据池中取样

train_samples
    ↓
决定每个 epoch 实际处理多少 cell

max_epochs
    ↓
决定总共重复多少个训练周期

这意味着，只要缩减后的 subset 仍然足够提供每个 epoch 所要求的样本数量，那么：

25% dataset
→ 每 epoch 仍处理 1,000,000 cells

10% dataset
→ 如果 train_samples 仍然是 1,000,000
→ 每 epoch 仍然处理约 1,000,000 cells

GPU 需要执行的 raw batch 数不会按 25% → 10% 的比例下降。

因此最终没有为了“提速”直接牺牲训练数据多样性，而是保留固定 25% training subset，改为直接减少训练预算。

这个阶段形成了一个重要认识：

> **dataset percentage 主要控制数据多样性；`train_samples` 和 `max_epochs` 才是当前工程中更直接控制训练计算量和总时长的参数。**

另外，当前每个 epoch 的样本并不是先对整个训练池做一次严格的 random.sample(1,000,000)。实际数据管线是：

固定 training subset
→ HF streaming dataset
→ shuffle buffer = 50,000
→ seed = random_seed + epoch
→ 从打乱后的 streaming 数据流中持续读取
→ 达到该 epoch 的 train_samples / raw-batch 上限后停止

因此可以近似理解为“每个 epoch 从固定数据池中经过随机打乱后取一段训练样本”，但不是对全池一次性做严格均匀无放回抽样。

---

25. 训练单位重新梳理：sample、batch、raw batch、optimizer update、epoch

为了后续能够正确判断训练规模，重新明确了当前工程中各单位的关系。

25.1 Sample

当前任务中可以近似理解为：

1 sample = 1 cell

---

25.2 Per-GPU batch size

当前：

batch_size = 92

这里的 92 是 每张 GPU 一次处理的 cell 数量。

因此：

GPU0 → 92 cells
GPU1 → 92 cells

---

25.3 Raw batch

双卡同时完成一次训练迭代时：

92 × 2 GPUs
= 184 cells

在当前记录中，把这一次两卡并行训练迭代称为：

1 raw batch = 184 cells

---

25.4 Gradient accumulation 与 optimizer update

当前：

accumulate_grad_batches = 2

因此不是每个 raw batch 都立即更新模型参数，而是：

raw batch #1
184 cells
→ forward/backward
→ 梯度先保留

raw batch #2
184 cells
→ forward/backward
→ 梯度继续累积

两次梯度合并
→ optimizer.step()
→ 真正更新一次参数

所以：

1 optimizer update
≈ 2 raw batches
≈ 368 cells

当前 effective batch size 因此为：

92 per GPU
× 2 GPUs
× 2 gradient accumulation
= 368
---

26. 技术顾问关于训练加速的建议，以及当前工程实际使用的技术

训练时间过长后，技术顾问主要提出了以下方向：

1. gradient accumulation；
2. DeepSpeed 或 Hugging Face Accelerate；
3. 确认是否使用 BF16；
4. 确认 batch size；
5. 后续建议接 TensorBoard，边训练边看 validation/test loss curve；
6. 认为 50 epochs 可能也没有必要全部跑满。
在讨论这些建议时，也重新明确了当前项目的训练技术栈。

26.1 PyTorch Lightning

当前 GeneJEPA 工程并不是直接用裸 PyTorch 手工管理整个训练循环，而是：

GeneJEPA model
    ↓
PyTorch
    ↓
PyTorch Lightning
    ↓
DDP
    ↓
GPU0 + GPU1

Lightning 目前负责管理：

- 多 GPU；
- DDP；
- BF16 mixed precision；
- gradient accumulation；
- optimizer / scheduler；
- epoch；
- validation；
- checkpoint；
- logger。
因此顾问提到的“gradient accumulation 几行代码”在当前工程中不需要再手工实现，Lightning 已经提供对应参数。

---

26.2 Gradient accumulation 已经启用

当前：

accumulate_grad_batches = 2

因此 gradient accumulation 已经实际运行并经过多轮 smoke / formal training 验证。

需要注意：

当前只能说“已经启用并验证可以正常工作”，并没有专门 benchmark accumulation=1/2/4 谁更快。

因此后续对外描述应为：

Gradient accumulation 已启用，当前设置为 2。
尚未单独做不同 accumulation 值的速度对比。

---

26.3 BF16

当前硬件与日志已经明确：

Hardware support for bfloat16: True
Using precision: 'bf16-mixed'

因此当前正式训练使用：

BF16 mixed precision

它不是把所有计算强制变成 BF16，而是由混合精度机制让适合的计算使用 BF16，同时部分需要数值稳定性的计算仍可保留更高精度。

---

26.4 Batch size 的完整表述

如果只问代码配置：

batch_size = 92

但在双卡 + gradient accumulation 场景下，更完整的描述应该是：

Per-GPU batch size = 92

Global raw batch
= 92 × 2 GPUs
= 184

Gradient accumulation = 2

Effective batch size
= 92 × 2 × 2
= 368

因此后续与技术顾问沟通时，推荐直接描述为：

batch size=92 per GPU，2 GPUs，gradient accumulation=2，因此 effective batch size=368。

---

26.5 DeepSpeed / Accelerate

当前正式训练使用的是：

PyTorch Lightning + DDP

尚未切换到：

DeepSpeed

也没有把整个训练框架迁移到：

Hugging Face Accelerate

考虑到 Lightning 本身支持 DeepSpeed strategy，因此后续如果专门做性能优化，更合理的实验是：

Lightning + 当前 DDP
vs
Lightning + DeepSpeed

做短 benchmark，而不是为了使用 Accelerate 重写整个训练工程。

当前阶段由于首要目标变成“先完成本次复现训练”，因此 DeepSpeed / Accelerate 暂未实际 benchmark 或用于正式训练。

这里的 benchmark 指“保持其他条件尽量一致，只改变一个待比较因素，然后比较速度/吞吐/稳定性”。前面已经实际做过的 num_workers=0/2/4/8 对比，本质上就是一次 DataLoader benchmark。

---

27. TensorBoard 接入与 validation loss 可视化

技术顾问提出：

如果仍需要跑较多 epoch，希望增加 TensorBoard 支持，这样可以边跑边看实际 validation 或 test loss curve，而不是等全部训练结束以后再判断模型是否已经收敛。

当前 GeneJEPA 代码本身已经存在 validation loss 计算，因此主要工作是增加 TensorBoard logger，而不是重新设计 validation。

---

27.1 TensorBoard 环境确认

当前环境：

TensorBoard 2.20.0

运行时有：

TensorFlow installation not found - running with reduced feature set.

但当前项目是 PyTorch / Lightning，目标只是查看 scalar 曲线，因此没有为此额外安装 TensorFlow。

启动方式：

uv run tensorboard \
  --logdir logs/tensorboard \
  --port 6006

浏览器访问：

http://localhost:6006

---

27.2 Lightning 同时使用 W&B 与 TensorBoard

原来只使用：

WandbLogger

后续增加：

TensorBoardLogger

并将 Trainer logger 改为 logger list，使两个 logger 可以同时接收 Lightning 的 self.log(...) 指标。

结构变为：

Lightning metrics
      ↓
  ┌───┴────┐
  ↓        ↓
W&B    TensorBoard

W&B 继续保持：

WANDB_MODE=offline

TensorBoard 则用于本地实时看训练曲线。

---

27.3 清理重复的 validation loss 指标

最初 validation_step() 中，同一个 val_sim 同时记录为：

val/loss
val_loss
val_loss/sim

因此 TensorBoard 中出现多个数值完全相同的 card。

后续将核心 validation 指标统一为：

val_loss

同时显式提供真实 batch size：

self.log(
    "val_loss",
    val_sim,
    on_step=False,
    on_epoch=True,
    prog_bar=True,
    sync_dist=True,
    batch_size=batch_size,
)

这样：

- TensorBoard 中只需要关注一个核心 val_loss；
- checkpoint 继续监控同一个 val_loss；
- 避免 Lightning 从 ragged batch 结构中错误猜测 batch size。
---

27.4 TensorBoard smoke test

为了验证 TensorBoard 链路，使用过短规模 smoke：

max_epochs = 3
train_samples = 1840
10 raw batches / epoch
5 optimizer updates / epoch
log_every_n_steps = 1

结果确认：

- TensorBoard 页面可以正常打开；
- val_loss 可以正常显示；
- 多个 epoch 后 val_loss 确实呈下降趋势；
- TensorBoard 与 Lightning validation 链路已经打通。
因此随后恢复正式训练时，将：

log_every_n_steps = 1

恢复为：

log_every_n_steps = 10

避免正式训练阶段过于频繁地写日志。

---

28. 正式训练预算从 1M×50 调整为 700k×30

在明确“dataset 比例”和“训练预算”不是同一个概念以后，最终没有把固定训练数据池从 25% 降到 10%。

正式训练保留：

Fixed training subset:
843 / 3372 shards
= 25%

但训练预算调整为：

train_samples = 700,000
max_epochs = 30

其他核心模型参数保持：

d = 768
latents_L = 512
blocks_D = 12
heads_h = 6

batch_size = 92
accumulate_grad_batches = 2

Validation 恢复正式规模：

validation_num_batches = 55

正式 run 名称：

GeneJEPA-quarter-d12-h6-700k-e30-seed42-run1

checkpoint 目录：

checkpoints/
genejepa_quarter_d12_h6_700k_e30_seed42_run1

正式启动前曾通过配置打印检查发现 checkpoint_dir 仍误指向 TensorBoard smoke 目录：

checkpoints/genejepa_tensorboard_smoke_v2

随后改为全新的正式目录，避免正式 run 意外从 smoke checkpoint 恢复。这个检查再次确认了长训练前“必须使用独立 checkpoint 目录”的原则。

---

28.1 新训练计划换算

每个 epoch：

700,000 / (92 × 2 GPUs)
≈ 3805 raw batches / epoch

梯度累积：

3805 / 2
≈ 1903 optimizer updates / epoch

30 个 epoch：

1903 × 30
= 57090 total optimizer updates

正式日志实际打印：

3805 raw batches/epoch
1903 optimizer updates/epoch
57090 total updates over 30 epochs

---

28.2 相比原 50×1M 的工作量变化

原计划：

1,000,000 × 50
= 50,000,000 cell-processing budget

新计划：

700,000 × 30
= 21,000,000 cell-processing budget

比例：

21M / 50M
= 42%

因此新方案的主要训练计算预算约为原方案的：

42%

即粗略减少：

约 58%

这比单纯把 dataset 从 25% 改成 10% 更直接地缩短了训练时长，同时保留了原固定 25% 数据池的数据多样性。

---

29. 正式训练中的 DDP 数据分流和 DataLoader worker 状态

正式 700k×30 训练中再次确认了双卡数据分流仍然存在。

训练数据：

843 fixed training shards

DDP 分配：

rank 0 / GPU0
→ 422 / 843 parquet files

rank 1 / GPU1
→ 421 / 843 parquet files

Validation：

16 validation shards

rank 0
→ 8 / 16

rank 1
→ 8 / 16

因此两张 GPU 并不是重复读取完全相同的一整套 parquet，而是在文件列表层先做 disjoint DDP rank sharding，再分别建立 streaming dataset。

---

29.1 Training workers

启动时继续使用：

GENEJEPA_TRAIN_WORKERS=2

它的实际含义是：

每个 DDP rank / 每张 GPU
使用 2 个 DataLoader workers

因此整个双卡作业实际共有：

rank0 → 2 workers
rank1 → 2 workers

总计 4 个 worker 子进程

而：

DataConfig.num_workers = 8

继续保留，主要用于维持当前 validation split 逻辑中的 16 个 validation shards，并不等于正式训练实际启用了 8 个 training workers。

Validation workers 仍保持：

0

以优先保证 streaming validation 稳定。

---

29.2 GPU utilization 波动

正式训练时观察到某张 GPU 会在：

100%
→ 0%
→ 再回到 100%

之间波动。

由于：

- 每个 cell 有效基因数量不同；
- ragged batch 的实际计算量不同；
- DDP 两个 rank 在梯度同步点需要互相等待；
- 当前机器双卡无 P2P / NVLink；
因此瞬时 GPU utilization 波动不能单独证明“数据没有分流”。

日志已经确认 DDP shard 分流仍然正常。

---

30. 正式训练第一次长期 CUDA 崩溃与 checkpoint resume

正式训练在已经运行多个 epoch 后第一次出现长期运行阶段 CUDA 崩溃。

典型位置：

Epoch 9
1819 / 3805
≈ 48%

错误首先出现在：

rank1
CUDA warning: unknown error
CUDA error: unknown error

随后 rank1 退出，Lightning/DDP 强制结束另一个进程。

这不是正常训练结束。

---

30.1 Checkpoint 保留情况

崩溃后 checkpoint 目录中确认存在：

last.ckpt

scjepa-epoch=04-val_loss=0.433.ckpt
scjepa-epoch=08-val_loss=0.345.ckpt

说明至少 Epoch 0～8 的完整训练结果已经保住。

重新启动原 run 后，日志明确出现：

Resuming from checkpoint: .../last.ckpt
Loaded EMA teacher state from checkpoint.
Restored all states from the checkpoint ...

随后重新进入：

Epoch 9

因此 checkpoint resume 已经实际验证有效，恢复的不只是模型权重，还包括 Lightning checkpoint 中的训练状态。

不过由于原 last.ckpt 主要在完整 epoch 结束后保存，因此 Epoch 9 已经跑过但尚未完成的部分需要重新训练。

---

30.2 TensorBoard version 分段

每次重新启动 run 时，TensorBoardLogger 会建立新的：

version_0
version_1
version_2
...

所以 TensorBoard 页面中同一次模型训练的不同 resume 阶段可能显示为不同 run/version。

这只是日志目录分段，不代表模型从随机初始化重新开始。

---

31. CUDA 错误重复出现：Epoch 15 多次重跑

后续训练再次发生低层 CUDA failure。

典型错误包括：

rank1
CUDA warning: unspecified launch failure
CUDA error: unspecified launch failure

而且多次都是 rank1 先失败。

不同重试中崩溃位置并不固定，例如曾出现：

Epoch 15
约 1469 / 3805

以及：

Epoch 15
约 590 / 3805

因此它不像“某一个固定 batch 每次必然触发错误”，更像长期双卡运行环境中的 CUDA / GPU / WSL / DDP 稳定性问题。

本阶段没有继续优先追查根因，因为工作目标临时调整为：

> 先尽可能把本次复现训练完成。

---

31.1 模型指标在崩溃前并没有明显发散

崩溃期间训练 loss 并没有出现 NaN / Inf 爆炸。

已保存 checkpoint 的 validation loss 例如：

epoch 4  → val_loss = 0.433
epoch 8  → val_loss = 0.345

后续运行中又观察到：

val_loss ≈ 0.299

说明在 CUDA 崩溃发生的同时，模型 validation loss 仍在继续下降。

因此当前更倾向于把“模型优化状态”和“底层 CUDA 稳定性”分开看待。

---

32. 增加自动崩溃重启脚本 auto_resume_genejepa.sh

由于 CUDA 错误发生时间不可预测，而且人工重新启动会浪费大量等待时间，因此在训练命令外层增加：

auto_resume_genejepa.sh

核心逻辑：

启动 GeneJEPA
    ↓
正常结束
→ 脚本退出

异常退出
→ 获取真实训练进程 exit code
→ 记录 nvidia-smi
→ 等待 60 秒
→ 自动重新执行 GeneJEPA
→ train.py 自动发现 last.ckpt
→ resume

---

32.1 关键：使用 PIPESTATUS 获取真实退出码

由于训练命令通过：

uv run -m genejepa.train \
2>&1 | tee -a ...

写日志，如果简单读取 $?，得到的可能是 tee 的退出状态，而不是 Python 训练进程的状态。

因此脚本使用：

exit_code=${PIPESTATUS[0]}

确保能够检测到真正的 CUDA crash。

---

32.2 自动重试策略

脚本配置：

RETRY_DELAY = 60 s
MAX_RETRIES = 20

每次异常退出后：

1. 记录 crash 时间；
2. 记录 nvidia-smi；
3. 等待 60 秒；
4. 重新启动训练；
5. 同一个 checkpoint 目录继续使用。
日志已经实际出现：

Starting training attempt 1

发生 CUDA crash 后又自动进入：

Starting training attempt 2

并成功再次从 last.ckpt 恢复训练。

因此：

> “训练异常退出 → 自动重启 → checkpoint resume”这条链路已经实际验证可工作。

---

33. 为什么 Epoch 15 会反复从头重跑：增加 mid-epoch recovery checkpoint

虽然自动重启已经有效，但出现了新的实际问题：

Epoch 15 跑了一部分
→ CUDA crash
→ last.ckpt 仍是 Epoch 14 结束时
→ 自动 resume
→ Epoch 15 从 0/3805 重新开始

如果 CUDA crash 的平均间隔短于一个完整 epoch，可能出现：

Epoch 15
→ 跑到 590
→ 崩溃

Epoch 15
→ 从 0 重来
→ 跑到 1469
→ 崩溃

Epoch 15
→ 又从 0 重来

这样即使自动重启有效，也可能长期无法跨过当前 epoch。

因此进一步设计：

> 在一个 epoch 内定期保存 recovery checkpoint。

---

33.1 Recovery checkpoint 设计

在原有正式 checkpoint callback 之外，新增独立：

recovery/

目录。

目标是每隔一定训练 step 更新：

recovery/last.ckpt

初始设计频率：

every_n_train_steps = 100

这样即使在一个 epoch 中途崩溃，也可以优先从较新的 recovery checkpoint 恢复，而不是永远退回前一个完整 epoch。

---

33.2 Resume 逻辑改为比较两个 last.ckpt

恢复逻辑从只检查：

main last.ckpt

改成同时检查：

checkpoints/.../last.ckpt

checkpoints/.../recovery/last.ckpt

然后：

ckpt_path = max(
    resume_candidates,
    key=os.path.getmtime,
)

即：

> 谁的修改时间更新，就从谁恢复。

---

33.3 当前 recovery checkpoint 代码状态

截至本记录更新时，train.py 已经加入：

- recovery checkpoint callback；
- recovery/last.ckpt 路径；
- main/recovery checkpoint 时间比较；
- trainer.fit(..., ckpt_path=ckpt_path) 恢复逻辑。
但最新一次代码审查发现，上传版本中 recovery callback 仍写为：

monitor=None
save_top_k=1
save_last=True
every_n_train_steps=100

这里建议继续修正为：

monitor=None
save_top_k=0
save_last=True
every_n_train_steps=100

并可删除 recovery filename=...，只维护最新的：

recovery/last.ckpt

原因是 recovery 的目标不是保存一系列 best checkpoint，而只是保留一个最近的“救命存档”。

**因此截至当前，mid-epoch recovery 机制属于“代码已加入、最后配置修正与实际崩溃恢复效果仍需验证”的状态，不能像 epoch-level `last.ckpt` 那样视为已经完全验收。**

另外，由于训练数据使用 streaming IterableDataset，即使中途恢复了模型、optimizer、scheduler、global step 等状态，也不应承诺能够逐样本精确回到完全相同的数据流位置；当前设计的核心目标是减少整轮 epoch 被重复计算的损失。

---

34. 截至 2026-08-17 的实际实验状态

当前整体状态可以概括为：

Tahoe-100M 本地数据                ✅
数据完整性检查                     ✅
损坏 parquet 修复                   ✅
固定 25% training subset           ✅
固定 validation shards              ✅

D=12 / heads=6                      ✅
随机初始化                          ✅

2 × RTX A6000                       ✅
Lightning DDP                       ✅
NCCL CUMEM workaround               ✅
双卡无 P2P / NVLink                 已确认

Per-GPU batch size = 92             ✅
Gradient accumulation = 2           ✅
Effective batch size = 368          ✅
BF16 mixed precision                ✅

Training DataLoader workers=2/rank  ✅
DDP train shard split 422/421       ✅
Validation shard split 8/8          ✅

TensorBoard                         ✅
val_loss 实时曲线                    ✅
W&B offline                         ✅

原训练预算 1M × 50                  已取消
当前训练预算 700k × 30              ✅
总 optimizer updates = 57090        ✅

Epoch-level checkpoint resume       ✅ 已实际验证
自动 crash/restart/resume           ✅ 已实际验证

Recurring rank1 CUDA failure         ⚠️ 仍存在
Epoch 15 反复重跑                   ⚠️ 已发生
Mid-epoch recovery checkpoint        🟡 已加入代码，待最终修正/验证

正式训练最终完成                     ⏳ 尚未完成

当前最主要的工程风险已经不是数据、DataLoader 或模型能否启动，而是：

> 长期双卡训练中的 rank1 CUDA 稳定性，以及如何在这种不稳定条件下尽可能减少失败重算并完成剩余训练。

---

35. 当前的标准配置描述

当前 GeneJEPA 使用 PyTorch Lightning + 双卡 DDP，
2×RTX A6000，BF16 mixed precision。

模型：
D=12，heads=6，d=768，L=512。

数据：
固定 Tahoe training pool 的 25%，843/3372 shards；
DDP 下 rank0/rank1 分别使用 422/421 个 training shards。

Batch：
92 per GPU；
双卡 global raw batch=184；
gradient accumulation=2；
effective batch size=368。

训练预算：
train_samples=700,000 / epoch；
3805 raw batches / epoch；
1903 optimizer updates / epoch；
max_epochs=30；
57090 total optimizer updates。

DataLoader：
training num_workers=2 per rank；
validation workers=0。

监控：
W&B offline + TensorBoard；
每个 epoch 记录 val_loss；
best checkpoint 监控 val_loss。

当前主要问题：
长期双卡训练 rank1 偶发 CUDA failure；
epoch-level auto resume 已可工作；
正在增加 mid-epoch recovery checkpoint，
以避免同一个 epoch 反复从头重跑。

---

36. 更新后的一句话总结

本次 GeneJEPA 复现已经从最初的“能否把 25% Tahoe、D=12、heads=6 的模型跑起来”，进一步进入了真正的长训练工程阶段：

固定 25% Tahoe 数据池
→ 训练预算从 1M×50 压缩到 700k×30
→ 明确 batch / raw batch / optimizer / epoch 的关系
→ 确认 gradient accumulation=2
→ 确认 BF16 mixed precision
→ 接入 TensorBoard 实时观察 val_loss
→ 正式双卡 DDP 长训练
→ checkpoint resume
→ CUDA crash 自动重启
→ 为 Epoch 15 反复重跑增加 mid-epoch recovery 机制

当前模型的 validation loss 已显示出持续下降趋势，但正式 30-epoch 任务尚未最终完成。

下一阶段的直接目标不是再改变模型或数据定义，而是：

> 在保持当前实验配置一致的前提下，利用 checkpoint / auto-resume / recovery checkpoint 尽可能稳定完成剩余训练，并最终整理完整 validation loss curve、best checkpoint 和最终实验结果。

---

37. 技术顾问提出事项与当前实际情况对照
技术顾问关注点
顾问意图
当前实际情况
状态
BF / BF16
确认是否使用低精度训练加速

当前日志明确为 bf16-mixed，RTX A6000 支持 BF16
✅ 已使用
Batch size
确认训练批次规模

92 per GPU；双卡 raw batch=184；grad accumulation=2；effective batch=368
✅ 已明确
Gradient accumulation
减少频繁更新/通信，并模拟更大的 effective batch

Lightning 已设置 accumulate_grad_batches=2，不需要额外手写几行实现
✅ 已使用
Gradient accumulation benchmark
判断不同累积值是否还能进一步提速
没有专门比较 1/2/4 的速度；只验证当前 2 可正常训练
⏸ 未单独 benchmark
DeepSpeed

尝试优化多 GPU / 通信 / 显存
当前仍是 Lightning DDP；尚未实际切换或 benchmark DeepSpeed
⏸ 未实施

Hugging Face Accelerate
尝试其他分布式训练管理方案
当前 Lightning 已承担训练管理；没有为了 Accelerate 重写工程
⏸ 未实施
减少训练时间
原 50 epochs 预计约两周，需压缩
没有直接把 dataset 改成 10%；而是将 1M×50 改为 700k×30，主要计算预算约剩原来的 42%
✅ 已调整
Dataset 25% → 10%
最初作为缩短训练时间的方案
分析后确认 subset 比例主要控制数据池多样性，不直接控制当前 limit_train_batches；最终继续固定 25%
✅ 决定保持 25%
Epoch 数量
顾问认为 50 可能没有必要
已将 max_epochs 从 50 降到 30
✅ 已调整
每 epoch sample budget
直接减少每轮计算量
train_samples 从 1,000,000 降到 700,000
✅ 已调整
TensorBoard
边训练边观察 validation/test loss curve
已接入 TensorBoard 2.20.0，并通过 smoke test 验证 val_loss 可实时显示且呈下降趋势
✅ 已完成
Validation loss curve
用实际收敛情况判断是否还需要继续训练
当前每 epoch 做 validation，核心指标统一为 val_loss，checkpoint 也监控 val_loss
✅ 已有
Test loss curve
顾问口头提及 validate 或 test
当前工程确认的是 validation pipeline；没有新增独立 test pipeline/test loss
⏸ 当前以 validation 为主
训练框架
确认当前工程如何管理训练
当前为 PyTorch + PyTorch Lightning + DDP
✅ 已明确
DataLoader 性能
避免 GPU 等数据
已 benchmark 0/2/4/8 workers，最终使用 GENEJEPA_TRAIN_WORKERS=2（每 rank 2 workers）
✅ 已完成
长任务断点恢复
避免 CUDA crash 后全部重跑
epoch-level last.ckpt resume 已多次验证；额外增加 auto_resume_genejepa.sh
✅ 已完成基础恢复
中途 checkpoint
避免 Epoch 15 每次从头重跑
已设计每 100 train steps 的 recovery checkpoint，并加入“main/recovery 取较新者”逻辑；最后配置仍待修正和实测
🟡 进行中
CUDA 稳定性
非顾问最初要求，但已成为当前主要风险

rank1 多次出现 CUDA unknown error / unspecified launch failure；自动恢复可绕过部分影响，但根因尚未解决
⚠️ 未解决


38. 训练结果
38.1 vocabulary
暂时无法在飞书文档外展示此内容
38.2 Loss curve
[图片]

[图片]

39. Benchmark
39.1 开始设计外挂 Benchmark
根据技术顾问建议，训练结束后不能只依赖 GeneJEPA 自己的 val_loss 判断模型效果。
val_loss 只能反映：
Student / Predictor 对 Teacher representation 的预测目标完成得怎么样。
但它不能直接回答：
GeneJEPA 最后生成的 cell embedding 是否真的包含有意义的生物学信息。
因此决定增加一个训练之外的 external benchmark。
Benchmark 不参与 GeneJEPA 参数更新，而是在训练结束后异步执行：
已经训练好的 GeneJEPA checkpoint
        ↓
冻结模型参数
        ↓
外部单细胞表达数据
        ↓
GeneJEPA EMA Teacher
        ↓
生成每个 cell 的 768-d embedding
        ↓
简单下游分类器
        ↓
预测 cell type
        ↓
评价 embedding quality
Benchmark 的核心目的不是训练一个很强的 cell-type classifier，而是检测：
GeneJEPA 自己生成的 embedding 中，是否已经存在可以被简单模型直接读取的 cell-type information。

---
39.2 为什么没有直接沿用作者的 supervised probe
GeneJEPA 原训练代码中其实已经配置：
SupervisedValidatorCallback(
    probe_dataset_path=
        "scvi-tools/human-lung-cell-atlas-scanvi",
    probe_cell_type_col="scanvi_label",
    max_probe_cells=10_000,
)
说明作者本身也使用 HLCA 做 supervised representation evaluation。
但作者的 probe 并不是我们希望使用的最严格线性 readout。
我们的顾虑是：
GeneJEPA embedding
        ↓
能力较强的 MLP classifier
        ↓
很高的分类结果
此时很难完全区分：
A. GeneJEPA embedding 本身已经组织得很好

还是

B. embedding 中的信息仍然比较纠缠，
   但后面的 MLP 自己又学到了复杂的非线性分类规则
因此本次 benchmark 决定采用能力更弱的：
Multiclass Logistic Regression
即严格的 linear probe：
768-d embedding
        ↓
线性分类边界
        ↓
18 cell types
没有：
hidden layer
GELU
Transformer
attention
额外非线性网络
所以如果 Logistic Regression 仍然可以区分 cell type，更能说明：
cell-type 信息已经被 GeneJEPA embedding 本身编码，并具有一定线性可读性。

---
39.3 Benchmark 的主要研究问题
本次 benchmark 主要回答两个问题。
问题一：GeneJEPA embedding 有没有 cell-type information？
即：
HLCA expression
↓
GeneJEPA embedding
↓
简单 Logistic Regression
↓
能否预测真实 cell type
如果分类能力明显高于无信息状态，说明 embedding 中确实存在生物学结构。
问题二：JEPA val_loss 是否对应 downstream representation quality？
因此不能只评估一个 checkpoint，而是选择多个 checkpoint：
Epoch24
Epoch25
Epoch29
全部使用：
完全相同的 HLCA cells
完全相同的 gene mapping
完全相同的 train/test split
完全相同的 embedding extraction
完全相同的 Logistic Regression
唯一变化：
GeneJEPA checkpoint
最终比较：
JEPA val_loss
vs
HLCA downstream Accuracy / Macro-F1 / Balanced Accuracy

---
39.4 检查可用于 Benchmark 的 checkpoint
实际 checkpoint 检查结果：
Epoch19 recovery
global_step = 37500

Epoch22 recovery
global_step = 43300

Epoch24
global_step = 47575
val_loss ≈ 0.18629

Epoch25
global_step = 49478
val_loss ≈ 0.17945

Epoch29 / last.ckpt
global_step = 57090
val_loss ≈ 0.20441
其中：
Epoch25
是完整训练过程中目前 val_loss 最好的 checkpoint。
第一阶段正式 benchmark 最终选择：
Epoch24
Epoch25
Epoch29
作为 late-stage checkpoint 对比。
Epoch19 / Epoch22 暂时保留，后续如果需要增加更多时间点，可以再补 benchmark。

---
39.5 第一次尝试 HLCA：发现 minified AnnData 问题
最初按照 GeneJEPA 作者代码中的：
scvi-tools/human-lung-cell-atlas-scanvi
下载了约：
760 MB
的：
adata.h5ad
检查结果：
584,944 cells
2,000 genes
scanvi_label 存在
但是：
adata.X
虽然 shape 正常，却发现：
nnz = 0
也就是整个表达矩阵为空。
进一步加载对应 SCANVI model 后确认：
Model's adata is minified?: True
SCANVI 本身能够根据保存的模型状态重新生成 normalized expression。
但是这里存在一个 benchmark 独立性问题：
SCANVI 本身使用 cell-type supervision
        ↓
用 SCANVI 恢复 expression
        ↓
再使用这些数据测试 cell-type classification
可能把 SCANVI 本身学到的信息带入 benchmark。
因此最终决定：
放弃使用 minified SCANVI AnnData 生成表达矩阵。
重新寻找真正包含原始表达数据的 HLCA Core。

---
39.6 下载并检查完整 HLCA Core
最终下载：
benchmark_data/hlca/
688185ad-11c2-4172-a53a-f4f1f4076860.h5ad
文件大小约：
5.47 GB
数据规模：
584,944 cells
27,402 genes
HLCA 提供多个 annotation hierarchy：
ann_level_1       4 classes
ann_level_2      11 classes
ann_level_3      25 classes
cell_type        50 classes
ann_finest_level 61 classes
其中：
ann_level_1:
Epithelial
Immune
Endothelial
Stroma
本次第一版正式 benchmark 最终选择：
ann_level_3
作为分类标签。

---
39.7 确认 HLCA 中真正使用哪一个 expression matrix
完整 HLCA 同时包含：
adata.X
adata.raw.X
检查发现：
adata.X
前100个 cell 的非零值：
min ≈ 0.0687
max ≈ 9.98
属于已经 normalized / transformed 的表达。
adata.raw.X
前100个 cell：
min nonzero = 1
max = 5963
属于 count-like expression。
由于 GeneJEPA 自己的输入 pipeline 会执行：
raw expression
↓
log1p
↓
Tahoe global mean/std normalization
因此正式 benchmark 决定使用：
adata.raw.X
避免把已经 normalized 的 adata.X 再次执行 GeneJEPA preprocessing。

---
39.8 HLCA → GeneJEPA gene vocabulary 检查
GeneJEPA 当前 Tahoe vocabulary：
62,710 genes
HLCA：
27,402 genes
第一轮仅使用：
gene symbol
进行 mapping 时得到：
Mapped   = 27,098
Unmapped = 304
Overlap  = 98.89%
之后进一步检查：
Ensembl ID
+
gene symbol
发现：
Both match, same index : 27080
Ensembl only           :   304
Symbol fallback        :     1
Mapping conflicts      :    17
Completely unmapped    :     0
最终正式 mapping rule 固定为：
1. 优先使用 Ensembl ID
2. Ensembl 无法映射时才使用 gene symbol fallback
3. Ensembl 与 symbol 指向不同 GeneJEPA index 时，以 Ensembl 为准
最终：
Final mapped genes = 27,402 / 27,402
Final overlap      = 100.00%
即：
HLCA 全部27,402个基因都可以映射到 GeneJEPA/Tahoe vocabulary。

---
39.9 HLCA → GeneJEPA embedding 20-cell smoke test
在正式跑几十万 cells 前，先随机选择20个 HLCA cells 做端到端 smoke test：
HLCA raw counts
↓
gene mapping
↓
log1p
↓
Tahoe global normalization
↓
Epoch25 EMA Teacher
↓
768-d embedding
结果：
Embedding shape = (20, 768)
Finite          = True
embedding norm：
mean ≈ 39.54
std  ≈ 0.19
20个 cell 的 pairwise cosine similarity：
mean ≈ 0.952
min  ≈ 0.852
max  ≈ 0.9999
证明：
- HLCA raw counts 可以成功输入 GeneJEPA；
- gene mapping 正常；
- normalization 正常；
- checkpoint 能加载；
- EMA Teacher inference 正常；
- embedding 全部 finite。

---
39.10 第一版 10k benchmark cohort 尝试失败
最初参考作者：
max_probe_cells = 10,000
准备从 HLCA 中抽取：
10,000 cells
并使用：
80% train
20% test
stratified split
seed = 42
但 ann_level_3 中存在大量极小类别，例如：
Lymphatic EC proliferating      28
Smooth muscle FAM83D+          335
SM activated stress response   556
Lymphatic EC differentiating   566
Myofibroblasts                 716
Rare                           885
先抽10k后，某些类别甚至只剩：
1 cell
导致第二次 stratified train/test split 报错：
ValueError:
The least populated class in y has only 1 member
说明：
直接保留所有稀有类别再抽10k，不适合做稳定的 multiclass benchmark。

---
39.11 正式 Benchmark cohort 改为 18 类 + 全部细胞
最终决定：
使用 ann_level_3
↓
删除无有效标签的 None
↓
删除全数据中 <1000 cells 的类别
↓
剩余类别全部纳入
去除的6类：
Lymphatic EC proliferating       28
Smooth muscle FAM83D+           335
SM activated stress response    556
Lymphatic EC differentiating    566
Myofibroblasts                  716
Rare                            885
最终：
18 classes
578,572 cells
18类：
Macrophages                111844
Basal                       84713
Secretory                   80327
AT2                         62405
T cell lineage              50859
Multiciliated lineage       41098
Monocytes                   26529
EC capillary                23205
Fibroblasts                 20384
Innate lymphoid cell NK     16978
EC venous                   12975
Dendritic cells             10319
AT1                          7937
EC arterial                  7391
Mast cells                   6623
B cell lineage               6284
Submucosal Secretory         4700
Lymphatic EC mature          4001

---
39.12 固定 Train/Test split
全部578,572个 benchmark cells 按：
80% train
20% test
stratified
seed = 42
固定划分。
最终：
Total = 578,572

Train = 462,857
Test  = 115,715

Classes = 18
最小 test class：
Lymphatic EC mature
800 cells
最大 test class：
Macrophages
22,369 cells
固定 cohort 保存为：
benchmark_data/hlca/
benchmark_ann_level3_all18_seed42.csv
该文件从此冻结。
后续：
Epoch24
Epoch25
Epoch29
全部使用完全相同的：
cell_index
label
train/test split
不得重新随机划分。

---
39.13 为什么最终没有限制在作者的 10k cells
作者的 10,000 cells 主要存在于训练过程中的 supervised callback。
我们的 benchmark 是：
训练已经结束
↓
离线独立运行
因此计算成本并不是主要限制。
使用全部符合 benchmark 定义的细胞有几个优点：
1. 减少随机抽样波动
2. 小类别 test 样本更多
3. Macro-F1 更稳定
4. checkpoint 之间比较更加 solid
因此最终没有继续人为下采样到10k，而是使用：
578,572 cells
需要注意：
当前 split 是 cell-level stratified split，而不是 donor-disjoint split。
所以当前 benchmark 最适合回答：
不同 GeneJEPA checkpoint 的 representation quality 谁更好。
它暂时不能被解释成严格的：
对完全未见过 donor 的泛化能力。
如果后续需要更严格 external generalization benchmark，可以再增加 donor-disjoint split。

---
39.14 建立批量 embedding extraction 脚本
正式建立：
extract_hlca_embeddings.py
核心流程：
Frozen benchmark CSV
↓
按 HLCA cell_index 排序
↓
backed H5AD 分 batch 读取 raw.X
↓
HLCA gene index → GeneJEPA index
↓
log1p
↓
Tahoe global mean/std
↓
EMA Teacher get_embedding()
↓
768-d embedding
↓
直接写 disk-backed .npy
Tahoe normalization 使用训练阶段同一组固定统计：
mean = 0.8255348793661643
std  = 0.31346807453602465
Inference 固定：
use_teacher=True
即正式 benchmark 使用：
EMA Teacher encoder 的 embedding。
embedding 输出使用：
float32

---
39.15 1000-cell batch extraction smoke test
正式大规模提取前先跑：
1000 cells
batch_size = 64
Epoch25
GPU0
结果：
Shape = (1000, 768)
Finite = True
Mapped genes = 27402 / 27402

Mean throughput ≈ 66.31 cells/s
说明批量 extraction pipeline 可以正常工作。

---
39.16 embedding extraction 改为双 GPU 独立分片
考虑到之前训练阶段 DDP / NCCL 存在不稳定问题，本次 inference 没有使用 DDP。
而是采用：
578,572 cells
        ↓
排序
        ↓
切成两个连续 shard
      ↙           ↘
 GPU0             GPU1
独立进程         独立进程
      ↘           ↙
      最后磁盘合并
两张 GPU 完全独立：
不使用 DDP
不使用 NCCL
不进行 GPU 间通信
这样既可以双卡加速，又不会重新引入训练阶段 NCCL instability。

---
39.17 双 GPU extraction smoke test
先测试：
总共 2000 cells
两张 GPU：
GPU0 → 1000 cells
GPU1 → 1000 cells
结果：
GPU0 ≈ 51.19 cells/s
GPU1 ≈ 82.92 cells/s
两个进程都成功输出：
(1000, 768)
虽然两边速度存在差异，但整体 wall-clock throughput 明显优于单卡，因此正式采用双进程双 GPU 方案。

---
39.18 Epoch25 全量 embedding extraction
Epoch25 checkpoint：
scjepa-epoch=25-val_loss=0.179.ckpt
使用：
batch_size = 64
num_shards = 2
两个 shard：
part0 = 289,286 cells
part1 = 289,286 cells
最终两边都完成：
Shape = (289286, 768)
然后使用：
merge_hlca_embeddings.py
合并。
完整性检查：
Combined rows: 578572

Frozen benchmark match: OK
Finite check: OK

Classes: 18

Train = 462857
Test  = 115715
最终：
epoch25_all18_embeddings.npy
shape = (578572, 768)

epoch25_all18_rows.csv

Embedding size ≈ 1.66 GB

---
39.19 Epoch24 / Epoch29 embedding extraction
随后对：
Epoch24
Epoch29
执行完全相同的 embedding extraction。
Epoch24
part0 = 289286 × 768
part1 = 289286 × 768
第一次 Epoch24 part1 在 GPU1 中途出现：
RuntimeError:
CUDA error: unspecified launch failure
由于 .npy 使用 memmap 创建，即使进程中途崩溃，文件表面上仍可能具有完整文件大小。
因此不能通过：
文件存在
或
文件大小正常
判断 extraction 是否完成。
处理方式：
删除失败的 Epoch24 part1
↓
保留成功 part0
↓
改用 GPU0 单独重新跑 part1
补跑成功：
[4521/4521]
cells = 289286 / 289286

Shape = (289286, 768)

Mean throughput ≈ 60.96 cells/s
Epoch29
Epoch29 两个 shard 均正常完成。

---
39.20 Epoch24 / Epoch29 merge 和完整性检查
分别建立：
merge_hlca_embeddings_epoch24.py
merge_hlca_embeddings_epoch29.py
两个 checkpoint 均得到：
Embedding shape = (578572, 768)

Rows = 578572
Classes = 18

Train = 462857
Test  = 115715

Embedding size ≈ 1.66 GB
并全部通过：
Frozen benchmark match: OK
Finite check: OK
最终正式 embedding：
benchmark_results/hlca18/
├── epoch24_all18_embeddings.npy
├── epoch24_all18_rows.csv
│
├── epoch25_all18_embeddings.npy
├── epoch25_all18_rows.csv
│
├── epoch29_all18_embeddings.npy
└── epoch29_all18_rows.csv
所以三个 checkpoint 在进入分类器以前已经确认：
完全相同的578,572个细胞
完全相同的行顺序
完全相同的 label
完全相同的 train/test split
全部 finite

---
39.21 正式 Linear Probe 设计
下游 probe 固定为：
StandardScaler
↓
Multiclass Logistic Regression
StandardScaler：
只在 training set 上 fit
再 transform train/test
避免使用 test set 信息进行预处理。
Logistic Regression 最终固定：
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    tol=1e-4,
    random_state=42,
)
并固定：
class_weight = None
没有使用：
class_weight="balanced"
也没有针对 test result 调节：
C
penalty
classifier architecture
原因是：
Benchmark 的目标不是把 Logistic Regression 调到最高分类性能，而是用固定且简单的 probe 比较 GeneJEPA embedding。

---
39.22 Benchmark 指标的含义
正式关注：
Accuracy
Macro-F1
Balanced Accuracy
另外 classification report 中还有：
Precision
Recall
F1
Support
Accuracy
表示：
所有 test cells 混在一起，有多少比例预测正确。
例如：
Accuracy = 0.59
表示约：
59%
的所有 test cells 分类正确。
Recall
针对某一个 cell type：
真正属于这一类的细胞中，有多少被成功识别出来。
例如：
Macrophages Recall = 0.8167
表示：
真实 Macrophages 中约81.67%
被成功识别为 Macrophages
Precision
表示：
被模型预测成某一类的细胞中，有多少真的属于这一类。
F1
F1 同时考虑：
Precision
+
Recall
只有：
预测得准
并且
找得比较全
F1 才会高。
Macro-F1
先分别计算18个类别各自的 F1，然后：
18个 F1 直接平均
每个类别权重完全相同。
所以：
800个细胞的小类
和：
22000多个细胞的大类
在 Macro-F1 里都只算“一票”。
因此 Macro-F1 对类别不平衡更敏感。
Balanced Accuracy
相当于：
18个类别 Recall 的平均
每个类别同等重要。

---
39.23 Logistic Regression convergence smoke test
正式跑46万 training cells 前，先使用：
20,000 train
5,000 test
做 solver / convergence smoke test。
max_iter = 300
结果：
ConvergenceWarning
Iterations = 300

Accuracy          = 0.452800
Macro-F1          = 0.249647
Balanced Accuracy = 0.247595
说明：
300 iterations 不足以完成 LBFGS optimization。
max_iter = 1000
结果：
ConvergenceWarning
Iterations = 1000

Accuracy          = 0.508600
Macro-F1          = 0.313506
Balanced Accuracy = 0.304565
仍然撞到 iteration 上限。
max_iter = 5000
最终实际：
Iterations = 1796
LBFGS 在第1796次自行满足 convergence condition，没有跑满5000。
结果：
Accuracy          = 0.512200
Macro-F1          = 0.315321
Balanced Accuracy = 0.306654
这说明：
max_iter = 5000
只是一个允许优化器充分求解的上限，并不意味着一定执行5000次。
因此从此正式冻结：
solver = lbfgs
max_iter = 5000
tol = 1e-4

---
39.24 Epoch25 正式全量 Linear Probe
正式使用：
Train X = (462857, 768)
Test X  = (115715, 768)

18 classes
结果：
Accuracy          = 0.592914
Macro-F1          = 0.432404
Balanced Accuracy = 0.406109

Iterations        = 4951
LBFGS 在5000上限之前正常收敛。
训练时间约：
76.85 min

---
39.25 Epoch25 各 cell type 的分类结果
正式 test set：
暂时无法在飞书文档外展示此内容
表现相对较好的类别包括：
Macrophages
Multiciliated lineage
EC capillary
T cell lineage
AT2
Basal
Secretory
比较弱的类别包括：
Dendritic cells
EC arterial
B cell lineage
Lymphatic EC mature
EC venous
Mast cells

---
39.26 Epoch24 / Epoch25 / Epoch29 正式 checkpoint 对比
三者全部使用：
同一个578,572-cell benchmark
同一个 train/test split
同一个 StandardScaler
同一个 Logistic Regression
正式结果：
暂时无法在飞书文档外展示此内容
三个指标的排序完全一致：
Epoch25 > Epoch24 > Epoch29
而 JEPA validation loss 的排序为：
Epoch25 < Epoch24 < Epoch29
即至少在这三个 late-stage checkpoint 上：
JEPA val_loss 最好的 Epoch25，同时也是 downstream linear probe 表现最好的 checkpoint。
但目前只比较了3个 checkpoint，因此更严谨的表述应该是：
在 Epoch24、25、29 三个时间点观察到一致的 ranking。
暂时不能据此声称所有训练阶段中：
val_loss
与 downstream representation quality 一定高度相关。

---
39.27 Epoch25 confusion matrix 分析
为了进一步理解：
Accuracy ≈ 59.3%
Macro-F1 ≈ 43.2%
到底错在哪里，重新运行一次 Epoch25 Logistic Regression，并永久保存逐细胞 prediction。
新脚本：
run_hlca_linear_probe_epoch25_confusion.py
输出：
benchmark_results/hlca18/
├── epoch25_predictions_full.csv
├── epoch25_confusion_matrix_raw_full.csv
├── epoch25_confusion_matrix_row_normalized_full.csv
├── epoch25_top3_misclassifications_full.csv
└── epoch25_confusion_matrix_row_normalized_full.png
重新运行完全复现原结果：
Accuracy          = 0.592914
Macro-F1          = 0.432404
Balanced Accuracy = 0.406109

Iterations = 4951
说明 confusion-matrix 版本没有改变 benchmark protocol。

---
39.28 confusion matrix 应该怎样理解
当前保存的是：
row-normalized confusion matrix
纵轴：
True cell type
横轴：
Predicted cell type
所以：
每一行代表一种真实 cell type。
每一行总和：
100%
对角线
表示：
真实类别
=
预测类别
即预测正确。
每个类别对角线的比例实际上就是该类别的：
Recall
所以：
对角线越亮
→ 这一类 Recall 越高
→ 越容易被正确识别
对角线以外的格子则表示：
这个真实 cell type 被错分到了什么类型。

---
39.29 Epoch25 主要 misclassification 现象
Macrophages
Recall = 81.67%
F1     = 71.99%
是真实识别效果最好的主要类别之一。

---
Dendritic cells
Recall = 5.09%
F1     = 9.04%
主要错误：
Dendritic
→ Macrophages     50.29%
→ Monocytes        9.79%
→ T cell lineage   9.74%
说明：
Dendritic cells 大量被吸收到其他 immune / myeloid 类别中。

---
B cell lineage
Recall = 9.55%
F1     = 15.64%
主要错误：
B cell lineage
→ T cell lineage 45.58%
→ Macrophages    12.89%
→ Basal           9.39%
说明 B / T 等 lymphoid subtype 的线性区分仍然较弱。

---
Monocytes
Recall = 26.91%
F1     = 34.49%
主要：
Monocytes
→ Macrophages 47.70%
Monocyte / Macrophage 是当前非常明显的一组混淆。

---
Innate lymphoid cell NK
Recall = 36.93%
F1     = 43.76%
最大的错误：
NK
→ T cell lineage ≈ 36.96%
说明 NK 与 T lineage 在 embedding 空间中距离较近。

---
EC arterial
Recall = 7.85%
F1     = 12.87%
主要：
EC arterial
→ EC capillary 28.42%
→ Macrophages  10.83%
→ EC venous    10.49%
一部分错误属于 endothelial subtype 内部混淆，但同时存在明显跨谱系预测。

---
EC venous
Recall = 21.04%
F1     = 27.53%
主要：
EC venous
→ Macrophages  13.41%
→ T cell       12.49%
→ EC capillary 12.41%

---
Lymphatic EC mature
Recall = 9.00%
F1     = 15.30%
错误分布比较分散：
→ Macrophages
→ EC venous
→ T cell lineage
→ Basal
→ EC capillary
→ AT2
因此这类比单纯的相近 subtype confusion 更值得后续关注。

---
39.30 当前对 Benchmark 结果的理解
不能把：
Accuracy = 59.29%
简单理解成：
GeneJEPA 只能识别59%的细胞，所以模型很差。
本次 probe 是：
18-class
严格线性 Logistic Regression
而不是强大的 nonlinear classifier。
因此当前更准确的结论是：
Epoch25 GeneJEPA embedding 中确实存在明显的 cell-type information，而且相当一部分信息已经能够被简单线性模型读取。
但是：
这种线性可分性在18个 cell type 之间非常不均匀。
部分大类和主要 cell type：
Macrophages
T cell lineage
Multiciliated lineage
EC capillary
AT2
Basal
Secretory
已经表现出较明显的线性结构。
而：
Dendritic cells
B cell lineage
EC arterial
EC venous
Lymphatic EC mature
仍然比较困难。

---
39.31 类别数量不平衡可能是一个重要影响因素
当前18类数量差异很大。
test support：
Lymphatic EC mature       800
Submucosal Secretory      940
B cell lineage           1257
...
Macrophages             22369
总体趋势上：
大类别通常比小类别更容易获得较高 Recall / F1。
但不能把所有低 F1 都解释为“细胞太少”。
例如：
Submucosal Secretory
support = 940
F1      = 0.474
而：
Dendritic cells
support = 2064
F1      = 0.090
Dendritic 的 test cells 数量更多，但结果反而明显更差。
说明最终表现同时受到：
类别数量
+
类别本身与其他 cell type 的相似度
+
GeneJEPA representation quality
+
Tahoe → normal lung 的 domain shift
+
当前 Logistic Regression class imbalance
等多个因素影响。
本次主 benchmark 明确固定：
class_weight = None
因此不会为了提升小类别结果而在 Epoch25 后单独改变 classifier。
否则会破坏 Epoch24 / 25 / 29 的公平比较。

---
39.32 对 Tahoe 25% pretraining subset 的新疑问
当前出现一个值得后续检查的问题：
GeneJEPA 正式训练仅使用：
Tahoe training pool 的固定25% shard subset
843 / 3372 shards
因此需要进一步确认：
这个25% subset 是否虽然数据量很大，但某些 biological context 覆盖不足。
这里不能直接问：
Tahoe 中有没有 HLCA 的 Dendritic / AT2 / EC arterial
因为 Tahoe 与 HLCA 的细胞体系并不完全对应。
更合理的 audit 是比较：
完整 Tahoe training pool
vs
当前固定25% subset
检查：
cell line 数量
每个 cell line 的 cell 数
drug / perturbation 数量
每个 perturbation 的 cell 数
cell line × perturbation coverage
从而判断：
固定25% subset 是否存在明显 sampling imbalance。
这项分析目前还没有执行。

---
39.33 当前 Benchmark 已完成状态
截至目前，以下工作已经完成：
✅ 完整 HLCA Core 下载
✅ raw expression 确认
✅ HLCA → GeneJEPA 27,402 genes 100% mapping
✅ 固定18-class benchmark cohort
✅ 578,572 cells 全量纳入
✅ 固定 80/20 stratified split
✅ Epoch24 embedding
✅ Epoch25 embedding
✅ Epoch29 embedding
✅ 三个 embedding 完整性检查
✅ strict Logistic Regression protocol 固定
✅ LR convergence 测试
✅ Epoch24 full linear probe
✅ Epoch25 full linear probe
✅ Epoch29 full linear probe
✅ 三 checkpoint downstream 对比
✅ Epoch25 confusion matrix
✅ Epoch25 per-cell predictions 保存
✅ Epoch25 Top-3 misclassification 输出
正式 checkpoint benchmark 的核心结论：
Epoch25
同时具有：
最低的 JEPA val_loss
+
最高的 Accuracy
+
最高的 Macro-F1
+
最高的 Balanced Accuracy
在目前比较的三个 late-stage checkpoint 中，Epoch25 是当前最佳模型。

---
39.34 当前主要 Benchmark 文件
Benchmark cohort
benchmark_data/hlca/
benchmark_ann_level3_all18_seed42.csv
Embeddings
benchmark_results/hlca18/
epoch24_all18_embeddings.npy
epoch24_all18_rows.csv

epoch25_all18_embeddings.npy
epoch25_all18_rows.csv

epoch29_all18_embeddings.npy
epoch29_all18_rows.csv
Linear probe
benchmark_results/hlca18/
epoch24_linear_probe_full.csv
epoch25_linear_probe_full.csv
epoch29_linear_probe_full.csv
Epoch25 confusion analysis
benchmark_results/hlca18/
epoch25_predictions_full.csv

epoch25_confusion_matrix_raw_full.csv

epoch25_confusion_matrix_row_normalized_full.csv

epoch25_top3_misclassifications_full.csv

epoch25_confusion_matrix_row_normalized_full.png
主要脚本
make_hlca_benchmark_split.py

extract_hlca_embeddings.py

merge_hlca_embeddings.py
merge_hlca_embeddings_epoch24.py
merge_hlca_embeddings_epoch29.py

run_hlca_linear_probe.py
run_hlca_linear_probe_epoch24.py
run_hlca_linear_probe_epoch29.py
run_hlca_linear_probe_epoch25_confusion.py

---
39.35 当前 Benchmark 方法的限制
目前需要明确保留以下限制。
39.35.1 当前是 cell-level split
当前：
train / test
来自同一个 HLCA pool，并不是 donor-disjoint。
所以目前主要适合：
checkpoint representation comparison
而不是严格证明：
GeneJEPA 可以泛化到完全未见过的新 donor。

---
39.35.2 类别严重不平衡
例如：
Macrophages
111,844 cells

Lymphatic EC mature
4,001 cells
差距非常大。
主 benchmark 故意没有加：
class_weight="balanced"
因此小类别 Recall 较低可能有一部分来自 classifier 的 class imbalance。
但为了保持 checkpoint 比较公平，目前不修改主 benchmark protocol。

---
39.35.3 Tahoe 与 HLCA 存在 domain shift
GeneJEPA 的 representation 是在 Tahoe-100M 上学习的。
HLCA 则是正常人肺组织 atlas。
因此当前 benchmark 本身包含：
pretraining dataset
→ external biological domain
的 transfer。
这意味着一些 cell type 表现不佳不一定只由模型结构导致，也可能与 pretraining coverage 有关。

---
39.35.4 当前只比较了三个 checkpoint
目前结论只能写成：
Epoch24 / Epoch25 / Epoch29 三个 checkpoint 的 JEPA val_loss ranking 与 downstream linear-probe ranking 一致。
暂时不应该扩大成：
所有训练阶段中 val_loss 与 biological representation quality 都严格相关。

---
39.36 下一步 Benchmark 待办
目前优先级较高的后续工作包括：
A. Tahoe 25% subset coverage audit
检查：
full Tahoe training pool
vs
25% frozen subset
在：
cell line
drug
perturbation
cell line × drug
上的覆盖情况。
目的是回答：
当前25%训练子集是否存在明显的数据覆盖偏差。

---
B. 利用 HLCA hierarchy 做 coarse-to-fine benchmark
HLCA 本身已经提供：
ann_level_1 = 4 classes
ann_level_2 = 11 classes
ann_level_3 = 25 classes
当前正式实验使用：
ann_level_3 → 筛成18类
后续可以在同一批 embedding 上增加：
ann_level_1 linear probe
ann_level_2 linear probe
ann_level_3 linear probe
检查 representation 是否呈现：
粗粒度 cell lineage
比较容易识别

↓  

细粒度 cell subtype
逐渐更难识别
这可以帮助判断：
GeneJEPA 学到的是不是一种由粗到细的 biological hierarchy。

---
C. 更严格的 donor-disjoint benchmark
如果后续需要评估 external generalization，而不仅仅是 checkpoint quality，可以进一步：
按 donor 分割 train / test
保证 test donor 完全未参与 probe training。

---
D. 可选增加 Epoch19 / Epoch22
如果希望更加完整地研究：
JEPA val_loss
vs
downstream embedding quality
可以继续补：
Epoch19
Epoch22
形成更多 checkpoint 时间点，而不只比较 Epoch24 / 25 / 29。

---
39.37 当前 Benchmark 总结
当前这套 Benchmark 的完整逻辑可以概括为：
Tahoe-100M 25% subset
↓
训练 GeneJEPA
↓
得到多个 checkpoint
↓
冻结 checkpoint
↓
使用完全独立的 HLCA Core
↓
raw counts
↓
统一 GeneJEPA gene vocabulary
↓
统一 GeneJEPA preprocessing
↓
EMA Teacher
↓
768-d cell embedding
↓
固定 HLCA 18-class train/test split
↓
严格线性 Logistic Regression
↓
Accuracy
Macro-F1
Balanced Accuracy
↓
比较 Epoch24 / Epoch25 / Epoch29
目前结果：
Epoch25
在：
JEPA validation objective
和：
external downstream linear probe
两个角度上均为三个 checkpoint 中最佳。
因此现阶段可以认为：
本次训练得到的 GeneJEPA embedding 确实包含可被线性读取的 cell-type biological information；在测试的三个 late-stage checkpoint 中，Epoch25 的 representation quality 最好。
与此同时，confusion matrix 显示：
不同 cell type 的线性可分性并不均匀，一些主要类别已经形成明显结构，而部分稀有类别、免疫亚型和 endothelial subtype 仍然较难区分。
后续重点将从：
“这个 benchmark 能不能跑？”
转向：
“为什么某些类别表现较弱？”
以及：
“25% Tahoe pretraining subset 的 biological coverage 是否足够均衡？”

