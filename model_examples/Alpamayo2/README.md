# Alpamayo 2 Super 模型应用说明

本文介绍在海光 HCU 上准备 [Alpamayo 2 Super](https://github.com/NVlabs/alpamayo2) Stage 1 环境、安装 TurboPhysAI，并启动单节点八卡训练。

## 1. 启动容器

容器启动参数参考[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)：

```bash
export CONTAINER_NAME=alpamayo2-v1
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

基础镜像提供 Python 3.12、DTK 26.10 和海光 PyTorch 2.13。直接使用容器内的 `/usr/local/bin/python3` 安装依赖；不要直接执行可能安装 CUDA 版 PyTorch 的官方 `uv sync`。安装会改变当前容器中该 Python 环境的包，不会修改基础镜像。

```bash
cd /workspace
/usr/local/bin/python3 -m pip install \
  huggingface_hub==0.36.0 transformers==4.57.1 accelerate==1.14.0 \
  av==18.0.0 hydra-core==1.3.4 hydra-colorlog==1.2.0 \
  mediapy==1.2.7 physical-ai-av==0.2.2 torchmetrics==1.8.2 \
  opencv-python==5.0.0.93 rich==15.0.0 wandb==0.28.1 pytest==8.4.2 \
  ninja wheel 'setuptools<81'

DS_BUILD_OPS=0 /usr/local/bin/python3 -m pip install --no-build-isolation deepspeed==0.19.1

/usr/local/bin/python3 -m pip install --no-deps -e /workspace/alpamayo2 \
  -e /workspace/alpamayo-recipes/src \
  -e /workspace/alpamayo-recipes/recipes/alpamayo2_sft
```

`DS_BUILD_OPS=0` 表示安装时不预编译 DeepSpeed 算子，运行时可能按需编译。

## 4. 获取并安装 TurboPhysAI

继续使用容器内的 `/usr/local/bin/python3` 执行：

```bash
git clone --branch alpamayo2 --single-branch \
  https://github.com/Coulsonq/TurboPhysAI.git /workspace/TurboPhysAI

/usr/local/bin/python3 -m pip install --no-build-isolation /workspace/TurboPhysAI
```

## 5. 数据集与权重

模型权重和数据集需要 Hugging Face 访问权限。请先在[模型页面](https://huggingface.co/nvidia/Alpamayo2-Super)和[数据集页面](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)阅读并确认许可条款，并按页面要求申请访问权限。然后在容器内登录：

```bash
hf auth login
hf auth whoami
```

登录时按提示输入 Hugging Face 访问令牌，不要将令牌写入命令、日志或文档。

### 5.1 下载模型权重

按照 [NVIDIA 官方说明](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft/README.md#model-access)，下载完整模型快照：

```bash
export MODEL_DIR=/workspace/alpamayo2-super
huggingface-cli download nvidia/Alpamayo2-Super --local-dir "$MODEL_DIR"
```

下载目录应包含模型配置、tokenizer、processor 文件和全部权重分片。

### 5.2 下载数据集

按照 [NVIDIA 官方数据准备说明](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft/README.md#prepare-pai-chunks-099)，下载完整 Stage 1 使用的 7 路相机、标定和 egomotion 数据：

```bash
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
cd /workspace/alpamayo-recipes

/usr/local/bin/python3 scripts/download_pai.py \
  --chunk-ids 0-100 \
  --camera \
    camera_cross_left_120fov \
    camera_front_wide_120fov \
    camera_cross_right_120fov \
    camera_rear_left_70fov \
    camera_rear_tele_30fov \
    camera_rear_right_70fov \
    camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "$PAI_DIR"
```

## 6. 单节点八卡 Stage 1 训练

| 配置 | 节点 / 卡数 | 相机 / 帧数 | 精度 | 每卡 batch size | 梯度累积 | 每进程 worker | 重计算 | DeepSpeed | 训练步数 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 官方配置 | 4 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 16 | 4 | 开启 | ZeRO-3 | 100000 | 默认保存 |
| 当前配置 | 1 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 16 | 4 | 开启 | ZeRO-3 | 100000 | 不保存 |

```bash
# 模型权重目录与数据集目录
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
# 每个训练进程的数据加载 worker 数，官方完整训练默认为 4
NUM_WORKERS=4
# 梯度累积次数，官方完整训练默认为 16
GRADIENT_ACCUMULATION_STEPS=16
# 训练 step 数，官方完整训练默认为 100000
MAX_STEPS=100000
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

/usr/local/bin/turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --disable-group alpamayo2.spawn_dataloader \
  -- \
  /usr/local/bin/torchrun \
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
    trainer.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    trainer.max_steps="$MAX_STEPS" \
    paths.output_dir="/workspace/outputs/alpamayo2-stage1-${STAMP}" \
    +trainer.save_strategy=no \
    2>&1 | tee "/workspace/logs/alpamayo2-stage1-${STAMP}.log"
```

根据实际目录修改 `MODEL_DIR` 和 `PAI_DIR`；通过 `NUM_WORKERS` 调整每个训练进程的数据加载 worker 数，通过 `GRADIENT_ACCUMULATION_STEPS` 和 `MAX_STEPS` 分别调整梯度累积次数和优化器更新步数。调整梯度累积次数会改变有效全局 batch size。`+trainer.save_strategy=no` 禁止保存 checkpoint；训练日志位于容器内 `/workspace/logs/`，对应宿主机 `/public/home/cuiqi/alpamayo2/logs/`。运行时配置设置 `HIP_BUFFER_EXTRA_SIZE=0`。模型、数据和参数的实际使用以运行日志和生成的配置为准。

## 7. 单卡 Stage 1 调试实验

| 配置 | 节点 / 卡数 | 相机 / 帧数 | 精度 | 每卡 batch size | 梯度累积 | 每进程 worker | 重计算 | DeepSpeed | 训练步数 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 官方单节点调试配置 | 1 节点 × 8 卡 | 1 路 / 1 帧 | BF16 | 1 | 1 | 0 | 开启 | ZeRO-3，优化器 CPU offload | 2 | 不保存 |
| 当前单卡实验配置 | 1 节点 × 1 卡 | 1 路 / 1 帧 | BF16 | 1 | 1 | 0 | 开启 | ZeRO-3，优化器 CPU offload | 2 | 不保存 |

当前单卡实验尚未验证，可能因显存不足而 OOM。

如需使用官方的八卡调试卡数，将下面的 `HIP_VISIBLE_DEVICES` 改为 `0,1,2,3,4,5,6,7`，并将 `--nproc_per_node` 改为 `8`。

```bash
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
# 每个训练进程的数据加载 worker 数，官方单节点调试默认为 0
NUM_WORKERS=0
# 梯度累积次数，官方单节点调试默认为 1
GRADIENT_ACCUMULATION_STEPS=1
# 训练 step 数，官方单节点调试默认为 2
MAX_STEPS=2
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

/usr/local/bin/turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --disable-group alpamayo2.spawn_dataloader \
  -- \
  /usr/local/bin/torchrun \
    --standalone --nnodes=1 --nproc_per_node=1 \
    -m alpamayo2_sft.train_hf \
    --config-path pkg://alpamayo2_sft/configs \
    --config-name sft_stage1_single_node_debug \
    model.pretrained_model_name_or_path="$MODEL_DIR" \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.local_dir="$PAI_DIR" \
    trainer.dataloader_num_workers="$NUM_WORKERS" \
    trainer.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    trainer.max_steps="$MAX_STEPS" \
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
