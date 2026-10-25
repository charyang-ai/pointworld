# PointWorld on one AMD Radeon Pro W7900

上游是 [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)（Apache-2.0，Stanford 与 NVIDIA，2026），论文 [arXiv:2601.03782](https://arxiv.org/abs/2601.03782)。上游说明原样保留在 [`UPSTREAM_README.md`](UPSTREAM_README.md)。

这个仓库把上游 `main` 分支（`0548482`）移植到 ROCm，在一张 AMD Radeon Pro W7900D 上跑通了上游 README 里的两条评估主路径：DROID 带置信度过滤的评估，以及 BEHAVIOR 评估。第一个提交 `2413d61` 是上游原样的树（树哈希 `6ace420` 与上游一致），之后的提交才是移植改动，`git diff 2413d61` 就能看到全部改动。权重、数据集和容器层都不在仓库里。

## 改了什么

上游的 GPU 专用依赖集中在 `ptv3/` 里的三个库：spconv、torch-scatter 和 flash-attn，它们都只提供 CUDA 内核。这次用纯 PyTorch 替换它们，模型代码只改了导入行。

| 文件 | 替换了什么 | 怎么核对的 |
| --- | --- | --- |
| `ptv3/sparse_conv.py` | `spconv.SubMConv3d`（PTv3 每个 block 的 3×3×3 位置编码）和 `SparseConvTensor` | 与 spconv 2.3.8 CPU 版逐位一致，见下文 |
| `ptv3/segment.py` | `torch_scatter.segment_csr`（GridPooling 的 max/mean） | 与逐段循环的参考实现逐位一致 |
| `ptv3/varlen_attention.py` | `flash_attn_varlen_qkvpacked_func` | fp32 下与逐序列 softmax 注意力相差 7e-7 |

另外四处改动：

- **`eval.py`、`train.py`**：最前面先 `import numba`。在 ROCm 上，`import torch` 会以全局符号的方式载入 HIP 运行时，其中 libamd_comgr 自带一份 LLVM。之后再载入的 llvmlite 会把符号绑定到这份 LLVM 上，numba 一编译就段错误，DataLoader 的子进程也一样。先载入 llvmlite 就没事。
- **`scene_featurizer.py`**：本地没有 DINOv3 权重、但传了 `--model_path` 时，只构建 DINOv3 的网络结构。PointWorld 检查点里存着冻结的 DINOv3 参数，而且按键名严格加载，所以评估不需要另外申请 DINOv3 的门控权重。从零训练仍然要按上游说明下载 DINOv3。
- **`environments/requirements-rocm.txt`**：去掉 CUDA 版 torch、spconv、torch-scatter 和 flash-attn，补上 DINOv3 `hubconf.py` 会导入的 `termcolor`。其余版本与上游一致。
- **`scripts/`、`tests/`**：容器、依赖、下载、子集数据准备、评估脚本，以及核对、统计和计时脚本。

### 重复坐标：spconv 的未定义行为成了模型的一部分

PointWorld 没有对点云做体素去重。场景点和 11 个时刻的机器人点拼成一个点云送进 PTv3，按 1.5 cm 的网格离散后，很多点会落进同一个体素。在验证用的 110 个 DROID 样本上，PTv3 第一层收到的点里有 58% 和别的点共用体素，其中机器人点是 87%，场景点是 50%，单个体素最多挤了 31 个点（`results/duplicate_stats_droid.json`）。

spconv 默认坐标唯一，重复坐标下它的结果取决于用哪种算法：

- **CUDA 构建默认的 `MaskImplicitGemm`**（也就是发布的检查点训练时的行为）：每个体素在哈希表里只留一个代表点。只遍历前 13 个核位置，后 13 个靠对称写回，所以代表点拿到全部 27 个核位置，非代表点只拿到前 13 个加中心。哪个点成为代表点取决于 GPU 上的写竞争。
- **CPU 构建的 Native 算法**：前 13 个核位置把邻居体素里所有重复点的特征加起来，只写给代表点；后 13 个核位置每个点都拿到。多线程的 scatter-add 没有用原子操作，重复坐标下会丢更新，结果并不确定。

`ptv3/sparse_conv.py` 用环境变量 `POINTWORLD_SUBM_MODE` 选择语义，默认 `cuda`：

| 模式 | 含义 |
| --- | --- |
| `cuda`（默认） | MaskImplicitGemm 的语义，下标最大的点当代表 |
| `cuda_first` | 同上，下标最小的点当代表 |
| `native` | spconv CPU 构建的语义 |
| `full` | 每个点都通过代表点拿到全部 27 个核位置 |

`tests/check_sparse_ops.py` 的核对结果在 `results/check_sparse_ops.log`。坐标唯一时，四种模式与 spconv 2.3.8 CPU 版的输出逐位相同，这同时确定了核位置顺序、偏移方向和 KRSC 权重布局。在两组重复率分别为 39% 和 35% 的随机点云上，`native` 模式与单线程的 spconv 也逐位相同（脚本会自动以 `OMP_NUM_THREADS=1` 重新启动，原因是上面说的写竞争）。`cuda` 模式的语义来自阅读 spconv 2.3.8 的 CUDA 源码（`spconv/csrc/sparse/indices.py` 中的 `calc_subm_conv_indices_mask`），这次没有 NVIDIA 硬件可以直接对照，佐证见下面的消融。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`（PyTorch 2.11.0+rocm7.14.0，HIP 7.14.60850，Triton 3.7.1）
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，48 GB）
- 容器只挂 `/dev/kfd` 和这张卡对应的一个渲染节点，进程里 `torch.cuda.device_count()` 为 1。容器内另外设置了 `HIP_VISIBLE_DEVICES=0` 和 `ROCR_VISIBLE_DEVICES=0`。
- 上游锁定 Python 3.10、torch 2.5.1 和 CUDA 12.4；这里是 Python 3.12，用镜像自带的 ROCm 版 PyTorch，numpy 仍是 1.26.4。

## 复现

宿主机上准备一个工作目录，挂到容器的 `/workspace`：

| 容器内路径 | 内容 |
| --- | --- |
| `/workspace/src` | 本仓库（含 `third_party/dinov3` 子模块） |
| `/workspace/data_src` | 上游 `data` 分支，原样使用（这次是 `3872ec6`） |
| `/workspace/venv` | PyTorch 以外的依赖 |
| `/workspace/weights/<name>/model-best.pt` | 发布的检查点 |
| `/workspace/data` | 下载的分卷、解包后的 H5、转换出的 WDS |
| `/workspace/results` | 评估日志和指标 |

```bash
# 宿主机
WORK=/path/to/workdir
git clone --recursive https://github.com/charyang-ai/pointworld "$WORK/src"
git clone -b data https://github.com/NVlabs/PointWorld "$WORK/data_src"
bash "$WORK/src/scripts/radeon_start_container.sh" "$WORK" /dev/dri/renderD128
docker exec radeon-pointworld bash /workspace/src/scripts/radeon_install_deps.sh
```

`radeon_start_container.sh` 的第二个参数是要用的那张卡的渲染节点，用 `ls -l /dev/dri/by-path` 和 `rocm-smi --showbus` 对照 PCI 地址来找。脚本带了 `--ulimit core=0`：初始化过 HIP 的进程一旦崩溃，单个 core dump 就有十几 GB，会经由 apport 写到宿主机的根分区。

### 权重和数据

验证只用到发布数据的一小部分：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `nvidia/PointWorld_models` 的 `small-droid/model-best.pt` | 1.8 GB | PTv3-50M，DROID |
| 同上 `large-droid/model-best.pt` | 13.0 GB | PTv3-1B，DROID |
| 同上 `large-droid+behavior/model-best.pt` | 13.0 GB | PTv3-1B，DROID + BEHAVIOR |
| `nvidia/PointWorld-DROID` 的 `droid/flows-fs-optimized/shard-000737` | 3.6 GB | 72 个场景，1158 个片段 |
| 同上 `droid/confidence` | 1.6 GB | 过滤指标用的专家置信度 |
| `nvidia/PointWorld-BEHAVIOR` 的 `behavior/flows/task-0000` | 1.7 GB | 199 个 episode，5780 个片段 |

DROID 选分片 737，是因为在论文切分清单里，它包含的测试片段最多（110 个）。

`scripts/radeon_download.sh` 按 `scripts/downloads.tsv` 断点续传，并用 Hugging Face API 给出的 sha256 校验。默认走 hf-mirror.com，设 `HF_ENDPOINT=https://huggingface.co` 可直连。也可以用上游 README 里的 `huggingface-cli download`。

```bash
ROOT=$WORK bash $WORK/src/scripts/radeon_download.sh $WORK/src/scripts/downloads.tsv .
PKG=$WORK/data/hf/datasets/nvidia
bash $PKG/PointWorld-DROID/recover_dataset_from_parts.sh --packages $PKG/PointWorld-DROID \
  --out $WORK/data/droid_restored --threads 0 \
  --include-prefix droid/flows-fs-optimized/shard-000737 --include-prefix droid/confidence
bash $PKG/PointWorld-BEHAVIOR/recover_dataset_from_parts.sh --packages $PKG/PointWorld-BEHAVIOR \
  --out $WORK/data/behavior_restored --threads 0 --include-prefix behavior/flows/task-0000
```

### 转成 WDS

`radeon_prepare_subset.sh` 依次运行 `data` 分支的 `data_integrity_check.py`、本仓库的 `make_paper_subset_manifest.py` 和 `data` 分支的 `convert_wds.py`。

`convert_wds.py` 要求清单里的每个测试片段都在本地，所以完整的论文清单在子集上会直接报错。`make_paper_subset_manifest.py` 只保留本地存在的片段，并沿用论文对它们的测试集划分。DROID 的文件名是 `<uuid>_flows.h5`，本地片段 ID 因此带 `_flows` 后缀，而发布的清单不带；脚本在匹配时会去掉这个后缀，主分支查置信度时也是这么处理的。

```bash
docker exec radeon-pointworld bash /workspace/src/scripts/radeon_prepare_subset.sh droid \
  /workspace/data/droid_restored/droid/flows-fs-optimized /workspace/data/droid/wds \
  /workspace/data/droid_restored/droid/confidence/expert_confidence-seed=42.h5
docker exec radeon-pointworld bash /workspace/src/scripts/radeon_prepare_subset.sh behavior \
  /workspace/data/behavior_restored/behavior/flows /workspace/data/behavior/wds
```

### 评估

`radeon_eval.sh` 调用的就是上游的 `eval.py`，参数与上游 README 一致（`--batch_size=1 --eval_num_batches=-1`）：

```bash
E="docker exec radeon-pointworld bash /workspace/src/scripts/radeon_eval.sh"
$E droid-small-cuda /workspace/weights/small-droid/model-best.pt droid /workspace/data/droid/wds --confidence_thres=0.8
$E droid-large-cuda /workspace/weights/large-droid/model-best.pt droid /workspace/data/droid/wds --confidence_thres=0.8
$E droid-joint-cuda /workspace/weights/large-droid+behavior/model-best.pt droid /workspace/data/droid/wds --confidence_thres=0.8
$E behavior-joint-cuda /workspace/weights/large-droid+behavior/model-best.pt behavior /workspace/data/behavior/wds \
  --norm_stats_path=stats/droid_behavior
```

第一个 batch 要多等约 40 秒，这是 `torch.compile` 在编译 DINOv3 那一段。

## 验证

2026-10-25，单卡，所有进程退出码为 0。

### DROID：分片 737 里的 110 个论文测试片段

`filtered_l2_moved` 是上游 README 主推的指标，表示经专家置信度过滤后、运动点上的 ℓ2 误差，单位为米。置信度阈值 0.8，三次运行都保留了 66.9% 的预测点。"整次评估用时"从启动 `eval.py` 算起，包含加载检查点和首个 batch 的编译。

| 检查点 | 参数量（含 DINOv3） | filtered_l2_moved | filtered_l2_static | l2_moved | 整次评估用时 |
| --- | --- | --- | --- | --- | --- |
| small-droid | 3.54 亿 | 0.03582 | 0.00730 | 0.05887 | 37 s |
| large-droid | 12.88 亿 | 0.03253 | 0.00624 | 0.05432 | 79 s |
| large-droid+behavior | 12.88 亿 | 0.03399 | 0.00696 | 0.05647 | 78 s |

作为参照，论文 Table 1 在完整的 DROID 测试集（约 4 万条轨迹）上的数字是：PTv3-50M 为 0.0331 / 0.0067，PTv3-1B 为 0.0312 / 0.0056。两个检查点的参数量减去 DINOv3 ViT-L 的约 3 亿，正好对应 PTv3-50M 和 PTv3-1B。子集只来自一个分片，绝对值偏高一些，但 1B 优于 50M 的方向与论文一致。

### BEHAVIOR：task-0000 里的 612 个论文测试片段

| 检查点 | l2_moved | l2_static | l2 | 整次评估用时 |
| --- | --- | --- | --- | --- |
| large-droid+behavior | 0.01191 | 0.00116 | 0.00137 | 240 s |

`l2_moved` 只统计有运动点的 383 个片段。论文 Table 2 里只在 BEHAVIOR 上训练的模型，在完整测试集上是 0.0087 / 0.0010。BEHAVIOR 是仿真数据，没有置信度过滤。

### 重复坐标语义的消融

同样是 110 个 DROID 片段，只改 `POINTWORLD_SUBM_MODE`：

| 模式 | small-droid moved | small-droid static | large-droid moved | large-droid static |
| --- | --- | --- | --- | --- |
| `cuda`（默认） | 0.035817 | 0.007298 | 0.032527 | 0.006238 |
| `cuda_first` | 0.035830（+0.04%） | 0.007293 | 0.032592（+0.20%） | 0.006225 |
| `native` | 0.036878（+3.0%） | 0.008092（+10.9%） | — | — |
| `full` | 0.036473（+1.8%） | 0.007763（+6.4%） | 0.033369（+2.6%） | 0.006603（+5.9%） |

（表中均为 filtered 指标。）按 CUDA 训练时的语义算最好；换一个代表点几乎没有影响，这和 GPU 上代表点本来就由写竞争决定是吻合的。改成"更干净"的 `full`，或者 spconv CPU 版的语义，误差都会上升，静态点上更明显。`native` 模式的 scatter 用的是 GPU 原子加，两次运行的 `filtered_l2_moved` 在第 6 位有效数字上不同；其余模式都是确定性的。

### 确定性和复现

- small-droid 连续跑两次，`metrics.json` 的 250 个指标逐一相同（`results/droid-small-cuda-repeat/`）。上游 README 提到在 GPU 上评估结果不确定；在这条路径上，spconv 选代表点时的写竞争就是一个不确定的来源，这里改成了确定性的选择。
- 用本仓库的脚本新建一个容器，依赖装进全新的 venv，再跑一次 small-droid，250 个指标与正式运行逐一相同（`results/verify/`）。这次复现跑在 `SubMConv3d` 加上下标缓存之前；加缓存后，上表所有 `cuda` 和 `cuda_first` 的运行（包括 BEHAVIOR）都重跑过，指标与加缓存前逐位相同。

### 速度

前向延迟取 batch 为 1、平均 10,238 个点的单个样本，跳过前 5 个样本，统计 60 个样本的中位数（`tests/forward_latency.py`）：

| 检查点 | 中位数 | p90 |
| --- | --- | --- |
| small-droid | 139 ms | 144 ms |
| large-droid | 300 ms | 326 ms |

评估在 fp32 下进行，注意力按上游的做法转成 bf16。RDNA3 没有 TF32，所以 W7900 上的 fp32 矩阵乘法就是真正的 fp32；而 `torch.set_float32_matmul_precision("high")` 在 Ampere 及之后的 NVIDIA 卡上会改用 TF32。

论文 Table 1 给 PTv3-1B 的延迟约 0.12 s，表中没有注明硬件和点数，只能当量级参照。

`tests/profile_forward.py` 做分段计时，在每段前后同步 GPU，所以总时间比上表多一些：

| 部分 | small-droid | large-droid |
| --- | --- | --- |
| 整个前向 | 159 ms | 323 ms |
| DINOv3 场景编码 | 36 ms（23%） | 37 ms（11%） |
| PTv3 | 118 ms（74%） | 281 ms（87%） |
| 其中 `SubMConv3d` | 68 ms（43%），每样本 22 次 | 148 ms（46%），每样本 76 次 |
| 其中变长注意力 | 8 ms（5%） | 28 ms（9%） |
| 其中 segment_csr | 2 ms（1%） | 3 ms（1%） |

最初的 `SubMConv3d` 每次调用都要为 26 个核位置各做一次 `nonzero`，每次都会让主机等 GPU。改成在每个层级第一次建邻居表时，就把每个核位置的行号和输入下标一起缓存下来，large-droid 的前向从 453 ms 降到 300 ms，计算顺序不变，指标逐位相同。`SubMConv3d` 仍然是最大的一块，下一步可以把 27 个核位置合成一次矩阵乘法，但那样求和顺序会变。PyTorch 的 SDPA 在这张卡上可以走 FLASH_ATTENTION 后端，注意力不是瓶颈。

## 结果文件

| 路径 | 内容 |
| --- | --- |
| `results/<run>/eval.log`、`metrics.json` | 每次评估的完整日志和上游 `eval.py` 写出的指标 |
| `results/check_sparse_ops.log` | 算子核对 |
| `results/duplicate_stats_droid.json` | 重复体素统计 |
| `results/latency_*.json`、`profile_*.json` | 前向延迟和分段计时 |
| `results/data_prep.log` | 完整性检查、子集清单和 WDS 转换的关键行 |
| `results/port.diff` | 相对上游的代码改动（不含本 README 和结果文件） |

## 许可

代码沿用上游的 Apache-2.0，见 `LICENSE` 和 `THIRD_PARTY_LICENSES.md`；PTv3 部分是 MIT，DINOv3 子模块使用 DINOv3 License。发布的检查点使用 NVIDIA Open Model License，数据集使用 NVIDIA License，本仓库不包含这两类文件。
