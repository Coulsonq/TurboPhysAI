# Alpamayo 2 Super 模型应用说明

本文介绍在海光 HCU 上准备 [Alpamayo 2 Super](https://github.com/NVlabs/alpamayo2) Stage 1 环境、安装 TurboPhysAI，并启动单节点八卡训练。

NVIDIA 的完整 Stage 1 配方使用 4 节点、每节点 8 卡；官方单节点调试配方使用八卡，并非单卡。本文提供单节点八卡完整配置和基于官方调试配置调整的单卡实验命令，两者在 HCU 上均未完成完整验证，可能发生显存不足。

## 1. 启动容器

在 nmz5 宿主机执行。以下基础镜像与已使用的 ACT v8 容器一致，需预先存在于宿主机。容器启动参数参考[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)：

```bash
export CONTAINER_NAME=alpamayo2-hcu
export IMAGE=dit-vla-wm-base:ubuntu22.04-dtk26.10-torch2.13-py3.12-cuiqi-80fkhsqb

docker run -dit \
  --network=host \
  --name="$CONTAINER_NAME" \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=512G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /public/home/cuiqi/alpamayo2:/workspace \
  "$IMAGE" \
  /bin/bash

docker exec -it "$CONTAINER_NAME" bash
```

上面的 `--privileged` 等设置与安装指南一致，会赋予容器较高的宿主机权限，仅应在可信环境使用。`/workspace` 映射到宿主机的 `/public/home/cuiqi/alpamayo2`，容器内创建的源码、虚拟环境、日志及输出也会保存在该目录。

## 2. 获取模型源码

以下命令在容器内执行：

```bash
cd /workspace

git clone https://github.com/NVlabs/alpamayo-recipes.git
git -C /workspace/alpamayo-recipes checkout --detach 622f9d23ff9e4961ba8c6aac71367776cdf6b016

git clone https://github.com/NVlabs/alpamayo2.git
git -C /workspace/alpamayo2 checkout --detach 9596749e33552468f3ae70da8c237bdd0401e59a
```

## 3. 安装训练环境

基础镜像提供 Python 3.12、DTK 26.10 和海光 PyTorch 2.13。创建继承镜像软件包的虚拟环境，保留海光 PyTorch；不要直接执行可能安装 CUDA 版 PyTorch 的官方 `uv sync`。

```bash
cd /workspace
/usr/local/bin/python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate

python -m pip install \
  huggingface_hub==0.36.0 transformers==4.57.1 accelerate==1.14.0 \
  av==18.0.0 hydra-core==1.3.4 hydra-colorlog==1.2.0 \
  mediapy==1.2.7 physical-ai-av==0.2.2 torchmetrics==1.8.2 \
  opencv-python==5.0.0.93 rich==15.0.0 wandb==0.28.1 pytest==8.4.2

DS_BUILD_OPS=0 python -m pip install --no-build-isolation deepspeed==0.19.1

python -m pip install --no-deps -e /workspace/alpamayo2 \
  -e /workspace/alpamayo-recipes/src \
  -e /workspace/alpamayo-recipes/recipes/alpamayo2_sft
```

`DS_BUILD_OPS=0` 表示安装时不预编译 DeepSpeed 算子，运行时可能按需编译。

## 4. 获取并安装 TurboPhysAI

在上一步创建的虚拟环境中执行：

```bash
cd /workspace
git clone --branch alpamayo2 --single-branch https://github.com/Coulsonq/TurboPhysAI.git /workspace/TurboPhysAI
python -m pip install -r /workspace/TurboPhysAI/requirements.txt
python -m pip install ninja wheel 'setuptools<81'
cd /workspace/TurboPhysAI
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps dist/turbo_physai-*.whl
```

`alpamayo2` 是开发分支，并非 TurboPhysAI 正式发布版本。请记录 `git -C /workspace/TurboPhysAI rev-parse HEAD` 的输出，以便复现。TurboPhysAI 通过 wheel 安装，源码更新后需要重新构建并安装。上述完整安装流程尚需在新容器中验证。

检查安装来源和设备：

```bash
python -c 'import torch, deepspeed, turbo_physai; print("torch:", torch.__version__, "HIP:", torch.version.hip, "DeepSpeed:", deepspeed.__version__); print("HCU count:", torch.cuda.device_count()); print("TurboPhysAI:", turbo_physai.__file__)'
python -c 'import turbo_physai._C; print(turbo_physai._C.__file__)'
turbo-physai --help
```

## 5. 数据集与权重

容器内使用以下已挂载路径：

```bash
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
```

`MODEL_DIR` 是完整的 `nvidia/Alpamayo2-Super` 权重快照。当前 `PAI_DIR` 仅包含 7 路相机的 chunk 0，适用于短测；**不包含官方完整 Stage 1 配方所需的全部数据**。正式命令使用训练集 chunk 0–98、验证集 chunk 99；需先按[官方数据准备说明](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft/README.md#prepare-pai-chunks-099)准备 chunk 0–99。训练和验证区间为左闭右开：`0-99` 不包含 99，`99-100` 只包含 99。

模型和数据集受访问许可约束，使用前请确认 [模型许可](https://huggingface.co/nvidia/Alpamayo2-Super)和[数据集许可](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)适用于实际场景。不要将访问令牌写入命令、日志或文档。

## 6. 单节点八卡 Stage 1 训练

先完成完整数据准备，然后在容器内执行。下面使用官方 `sft_stage1`：7 路相机、每路 4 帧、BF16、每卡 batch size 1、梯度累积 16、重计算开启、DeepSpeed ZeRO-3。相对官方四节点配置，此处改为单节点八卡，并关闭 checkpoint 保存；有效全局 batch size 从 512 降为 128。

TurboPhysAI 的 `flash_attention` 和 `non_blocking_h2d` 改变执行方式；为保留官方每卡 4 个数据加载 worker，命令中关闭了将 worker 改为 2 的 `spawn_dataloader` 优化组。优化组已在不同的 5-step 实验配置中做过吞吐验证，但本节官方完整配置及其精度尚未验证。

```bash
source /workspace/venv/bin/activate
# 模型权重目录与数据集目录
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
# 每个训练进程的数据加载 worker 数
NUM_WORKERS=4
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --disable-group alpamayo2.spawn_dataloader \
  -- \
  torchrun \
    --standalone --nnodes=1 --nproc_per_node=8 \
    -m alpamayo2_sft.train_hf \
    --config-path pkg://alpamayo2_sft/configs \
    --config-name sft_stage1 \
    model.pretrained_model_name_or_path="$MODEL_DIR" \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.train_dataset.chunk_ids="0-99" \
    data.val_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.chunk_ids="99-100" \
    trainer.dataloader_num_workers="$NUM_WORKERS" \
    paths.output_dir="/workspace/outputs/alpamayo2-stage1-${STAMP}" \
    +trainer.save_strategy=no \
    2>&1 | tee "/workspace/logs/alpamayo2-stage1-${STAMP}.log"
```

根据实际目录修改 `MODEL_DIR` 和 `PAI_DIR`；通过 `NUM_WORKERS` 调整每个训练进程的数据加载 worker 数，默认 4，八卡合计 32 个。`+trainer.save_strategy=no` 禁止保存 checkpoint；训练日志位于容器内 `/workspace/logs/`，对应宿主机 `/public/home/cuiqi/alpamayo2/logs/`。运行时配置设置 `HIP_BUFFER_EXTRA_SIZE=0`。模型、数据和参数的实际使用以运行日志和生成的配置为准。

## 7. 单卡 Stage 1 调试实验

官方 `sft_stage1_single_node_debug` 是**单节点八卡**的两步调试配置，不是官方单卡训练方案。以下命令仅将其调整为单卡实验：使用 1 路相机、1 帧、chunk 0、每卡 batch size 1、梯度累积 1、2 step、每进程 0 个 worker 和 ZeRO-3 优化器 CPU offload；数据加载优化组保持关闭。该单卡组合尚未验证，可能因模型或训练状态过大而 OOM。调试配置默认不保存 checkpoint。

如需使用官方的八卡调试卡数，将下面的 `HIP_VISIBLE_DEVICES` 改为 `0,1,2,3,4,5,6,7`，并将 `--nproc_per_node` 改为 `8`。

```bash
source /workspace/venv/bin/activate
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
NUM_WORKERS=0
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --disable-group alpamayo2.spawn_dataloader \
  -- \
  torchrun \
    --standalone --nnodes=1 --nproc_per_node=1 \
    -m alpamayo2_sft.train_hf \
    --config-path pkg://alpamayo2_sft/configs \
    --config-name sft_stage1_single_node_debug \
    model.pretrained_model_name_or_path="$MODEL_DIR" \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.local_dir="$PAI_DIR" \
    trainer.dataloader_num_workers="$NUM_WORKERS" \
    trainer.save_strategy=no \
    paths.output_dir="/workspace/outputs/alpamayo2-stage1-debug-1gpu-${STAMP}" \
    2>&1 | tee "/workspace/logs/alpamayo2-stage1-debug-1gpu-${STAMP}.log"
```

## 8. 与官方配置的区别

| 项目 | 官方完整 Stage 1 | 本文八卡命令 | 官方单节点调试 | 本文单卡实验 |
| --- | --- | --- | --- | --- |
| 卡数 | 4 节点 × 8 卡 | 1 节点 × 8 卡 | 1 节点 × 8 卡 | 1 节点 × 1 卡 |
| 相机 / 帧 | 7 路 / 每路 4 帧 | 同官方完整配置 | 1 路 / 1 帧 | 同官方调试配置 |
| 数据 chunk | 训练 0–98；验证 99 | 相同，须准备完整数据 | 训练和验证均为 0 | 同官方调试配置 |
| 每卡 batch / 梯度累积 | 1 / 16 | 1 / 16 | 1 / 1 | 1 / 1 |
| 每进程 worker | 4 | 默认 4，可通过 `NUM_WORKERS` 调整 | 0 | 默认 0，可通过 `NUM_WORKERS` 调整 |
| 训练步数 | 100000 | 100000 | 2 | 2 |
| DeepSpeed | ZeRO-3 | ZeRO-3 | ZeRO-3，优化器 CPU offload | 同官方调试配置 |
| checkpoint | 默认保存，每 500 步 | 不保存 | 不保存 | 不保存 |

两条本文命令均使用海光镜像及其 PyTorch/DTK 环境，和官方 CUDA/H100 环境不同；并启用 TurboPhysAI 的 `flash_attention`、`non_blocking_h2d` 及 runtime 配置。路径、卡数、checkpoint 策略和可调整的 worker 以命令为准，其他未覆盖的训练参数由对应官方配置提供。

## 9. 优化项

| 优化组 | 作用 | 第 6 节命令 |
| --- | --- | --- |
| `alpamayo2.spawn_dataloader` | 每个 rank 使用 2 个 spawn 数据加载 worker | 关闭，使用命令中的 `NUM_WORKERS` |
| `alpamayo2.flash_attention` | 使用 FlashAttention 2 | 开启 |
| `alpamayo2.non_blocking_h2d` | 开启数据异步传输 | 开启 |

上述优化是在特定镜像、数据子集、BS=1、梯度累积 1 及优化器 CPU offload 条件下做的 5-step 吞吐实验；不等同于本节完整配置的性能或精度验证。需要启用 2-worker 优化时，可移除第 6 节命令中的 `--disable-group alpamayo2.spawn_dataloader`，但此时数据加载配置不再与官方默认一致。
