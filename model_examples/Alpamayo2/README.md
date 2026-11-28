# Alpamayo 2 应用说明

本文介绍在海光 HCU 上准备 [Alpamayo 2 Super](https://github.com/NVlabs/alpamayo2) Stage 1 环境、安装 TurboPhysAI，并启动单节点八卡训练。

## 1. 启动容器

容器启动参数参考[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)：

```bash
export ALPA_WORKSPACE="$HOME/alpamayo2"
mkdir -p "$ALPA_WORKSPACE"
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
  -v "$ALPA_WORKSPACE:/workspace" \
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

source /opt/dtk/env.sh
/usr/local/bin/python3 -m pip install --no-deps hjson msgpack py-cpuinfo
/usr/local/bin/python3 -m pip install --target=/workspace/deepspeed --no-deps --only-binary=:all: \
  --index-url https://pypi.sourcefind.cn/nightly/dtk2610/+simple/ \
  'deepspeed==0.19.7+dtk2610.torch2130.2611241241.g0b8d6a'

/usr/local/bin/python3 -m pip install --no-deps -e /workspace/alpamayo2 \
  -e /workspace/alpamayo-recipes/src \
  -e /workspace/alpamayo-recipes/recipes/alpamayo2_sft
```

## 4. 获取并安装 TurboPhysAI

继续使用容器内的 `/usr/local/bin/python3` 执行：

```bash
git clone --branch alpamayo2 --single-branch \
  https://github.com/Coulsonq/TurboPhysAI.git /workspace/TurboPhysAI

/usr/local/bin/python3 -m pip install --no-build-isolation /workspace/TurboPhysAI
```

当前完整配置启用了四区域 ZeRO-3 compile。TurboPhysAI 的
`alpamayo2.zero3_module_compile` 组负责 DeepSpeed 编译兼容与 native 扩展加载：
保留普通输入的动态形状、补齐调度器 keyword tensor 依赖，并将编译图的梯度归约
接入 `zero3_comm_overlap` 的独立通信组。

首次使用时，Turbo 使用随包交付的 C++ 实现和已安装 DeepSpeed 的其余源码、头文件，在
`${XDG_CACHE_HOME:-~/.cache}/turbo_physai/alpamayo2/deepcompile/` 下构建并缓存扩展。
此步骤需要镜像的 DTK 工具链、C++ 编译器和 `ninja`；后续复用缓存。
不需要手工应用 DeepSpeed 补丁、运行额外构建脚本或设置 DeepSpeed 副本的
`PYTHONPATH`；已安装的 DeepSpeed 源文件保持原样。


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

参考 [NVIDIA 官方数据准备说明](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft/README.md#prepare-pai-chunks-099)，下载 chunk 0 的 7 路相机、标定和 egomotion 数据：

```bash
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
cd /workspace/alpamayo-recipes

/usr/local/bin/python3 scripts/download_pai.py \
  --chunk-ids 0-1 \
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

执行第 2–5 节命令后，`/workspace` 下的主要目录结构如下（省略 Git、下载缓存及部分文件）：

```text
/workspace/
├── TurboPhysAI/
│   ├── model_examples/Alpamayo2/README.md
│   └── turbo_physai/optimizations/models/alpamayo2/
│       ├── configs/                # optimization.yaml、recipe.yaml、runtime.yaml
│       ├── native/                 # Turbo 管理的 C++ 源码
│       ├── native_deepspeed.py
│       └── deepspeed_compat.py
├── deepspeed/                      # pip --target 的安装目录
│   ├── deepspeed/                  # 海光版 DeepSpeed Python 包
│   ├── deepspeed.libs/
│   └── deepspeed-*.dist-info/
├── alpamayo2/
│   └── src/
├── alpamayo-recipes/
│   ├── recipes/alpamayo2_sft/
│   ├── src/
│   └── scripts/download_pai.py
├── alpamayo2-super/
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00015.safetensors
│   ├── …                          # 共 15 个权重分片
│   └── model-00015-of-00015.safetensors
└── PhysicalAI-Autonomous-Vehicles-stage1-7cam/
    ├── clip_index.parquet
    ├── features.csv
    ├── calibration/
    │   ├── camera_intrinsics/
    │   └── sensor_extrinsics/
    ├── camera/                    # 5.2 中指定的 7 路相机目录
    └── labels/egomotion/
```

启动第 6 节训练后才会生成 `logs/`、`outputs/` 及 runtime 中指定的 `alpamayo2-compile/` 编译缓存；native 扩展缓存由 Turbo 自动管理。

## 6. 单节点八卡 Stage 1 训练

| 配置 | 节点 / 卡数 | 相机 / 帧数 | 精度 | 每卡 batch size | 梯度累积 | 每进程 worker | 重计算 | DeepSpeed | 训练步数 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 官方配置 | 4 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 16 | 4 | 开启 | ZeRO-3 | 100000 | 默认保存 |
| 当前配置 | 1 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 1 | 2 | 开启 | ZeRO-3 | 100000 | 不保存 |

训练和验证均使用 chunk 0，样本是否重叠取决于数据集划分；此配置用于调试，不能代表完整数据集训练或独立验证。

八卡训练和下方单卡调试命令均显式覆盖 `PYTHONPATH`，指向第 3 节确定的 `/workspace/deepspeed` 安装目录及模型、recipes 源码目录，避免继承旧 DeepSpeed 副本路径而导致 `zero3_module_compile` 的 `source.identity` 检查失败。

```bash
# 模型权重目录与数据集目录
export MODEL_DIR=/workspace/alpamayo2-super
export PAI_DIR=/workspace/PhysicalAI-Autonomous-Vehicles-stage1-7cam
# 关闭 worker/OMP 优化组时使用的 worker 数，官方完整训练默认为 4
NUM_WORKERS=4
# 梯度累积次数，官方完整训练默认为 16
GRADIENT_ACCUMULATION_STEPS=1
# 训练 step 数，官方完整训练默认为 100000
MAX_STEPS=100000
source /opt/dtk/env.sh
# 固定使用 workspace 中的海光版 DeepSpeed，覆盖旧 PYTHONPATH
export PYTHONPATH="/workspace/deepspeed:/workspace/alpamayo2/src:/workspace/alpamayo-recipes/recipes"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

/usr/local/bin/turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --log-report \
  -- \
  /usr/local/bin/torchrun \
    --standalone --nnodes=1 --nproc_per_node=8 \
    -m alpamayo2_sft.train_hf \
    --config-path pkg://alpamayo2_sft/configs \
    --config-name sft_stage1 \
    model.pretrained_model_name_or_path="$MODEL_DIR" \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.train_dataset.chunk_ids="0-1" \
    data.val_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.chunk_ids="0-1" \
    trainer.dataloader_num_workers="$NUM_WORKERS" \
    trainer.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    trainer.max_steps="$MAX_STEPS" \
    paths.output_dir="/workspace/outputs/alpamayo2-stage1-${STAMP}" \
    +trainer.save_strategy=no \
    2>&1 | tee "/workspace/logs/alpamayo2-stage1-${STAMP}.log"
```

根据实际目录修改 `MODEL_DIR` 和 `PAI_DIR`；默认启用 worker/OMP 优化组，使用每个 rank 2 个 worker 和 4 个 PyTorch CPU 线程；关闭该组后通过 `NUM_WORKERS` 调整 worker 数。通过 `GRADIENT_ACCUMULATION_STEPS` 和 `MAX_STEPS` 分别调整梯度累积次数和优化器更新步数。调整梯度累积次数会改变有效全局 batch size。`+trainer.save_strategy=no` 禁止保存 checkpoint；训练日志位于容器内 `/workspace/logs/`，对应宿主机 `$ALPA_WORKSPACE/logs/`。运行时配置设置 `HIP_BUFFER_EXTRA_SIZE=0`、`MIOPEN_FIND_MODE=1` 和 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`。模型、数据和参数的实际使用以运行日志和生成的配置为准。

## 7. 单卡 Stage 1 调试实验

| 配置 | 节点 / 卡数 | 相机 / 帧数 | 精度 | 每卡 batch size | 梯度累积 | 每进程 worker | 重计算 | DeepSpeed | 训练步数 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 官方单节点调试配置 | 1 节点 × 8 卡 | 1 路 / 1 帧 | BF16 | 1 | 1 | 0 | 开启 | ZeRO-3，优化器 CPU offload | 2 | 不保存 |
| 当前单卡实验配置 | 1 节点 × 1 卡 | 1 路 / 1 帧 | BF16 | 1 | 1 | 0 | 开启 | ZeRO-3，优化器 CPU offload | 2 | 不保存 |

当前单卡实验尚未验证，可能因显存不足而 OOM。该实验保留 CPU offload，因此命令中关闭仅支持无 CPU offload 的 `zero3_module_compile` 组。

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
source /opt/dtk/env.sh
# 固定使用 workspace 中的海光版 DeepSpeed，覆盖旧 PYTHONPATH
export PYTHONPATH="/workspace/deepspeed:/workspace/alpamayo2/src:/workspace/alpamayo-recipes/recipes"
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/alpamayo-recipes/recipes/alpamayo2_sft
mkdir -p /workspace/logs /workspace/outputs
STAMP="$(date +%Y%m%d_%H%M%S)"
set -o pipefail

/usr/local/bin/turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/alpamayo2/configs/runtime.yaml \
  --log-report \
  --disable-group alpamayo2.dataloader_workers_omp,alpamayo2.disable_cpu_offload,alpamayo2.zero3_module_compile \
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
| 数据 chunk | 训练 0–98；验证 99 | 训练和验证均为 0 | 训练和验证均为 0 | 同官方调试配置 |
| 每卡 batch / 梯度累积 | 1 / 16 | 1 / 1 | 1 / 1 | 1 / 1 |
| 每进程 worker | 4 | 默认 2；关闭 worker/OMP 组后可通过 `NUM_WORKERS` 调整 | 0 | 默认 0，可通过 `NUM_WORKERS` 调整 |
| 训练步数 | 100000 | 100000 | 2 | 2 |
| DeepSpeed | ZeRO-3 | ZeRO-3 | ZeRO-3，优化器 CPU offload | 同官方调试配置 |
| checkpoint | 默认保存，每 500 步 | 不保存 | 不保存 | 不保存 |

两条本文命令均使用海光镜像及其 PyTorch/DTK 环境，和官方 CUDA/H100 环境不同；并启用 TurboPhysAI 的 `flash_attention`、`non_blocking_h2d`、`channels_last_vision`、`bf16_communication`、`zero3_comm_overlap` 及 runtime 配置。路径、卡数、checkpoint 策略和可调整的 worker 以命令为准，其他未覆盖的训练参数由对应官方配置提供。
