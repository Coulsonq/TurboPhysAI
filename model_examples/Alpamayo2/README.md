# Alpamayo 2 应用说明

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

source /opt/dtk/env.sh
/usr/local/bin/python3 -m pip install --no-deps hjson msgpack py-cpuinfo
/usr/local/bin/python3 -m pip install --no-deps --only-binary=:all: \
  --index-url https://pypi.sourcefind.cn/nightly/dtk2610/+simple/ \
  --target /workspace/deepspeed-wheel \
  'deepspeed==0.19.7+dtk2610.torch2130.2611241241.g0b8d6a'

/usr/local/bin/python3 -m pip install --no-deps -e /workspace/alpamayo2 \
  -e /workspace/alpamayo-recipes/src \
  -e /workspace/alpamayo-recipes/recipes/alpamayo2_sft
```

DeepSpeed 使用[光源发布的固定版本 wheel](https://pypi.sourcefind.cn/das_nightly/dtk2610/deepspeed/0.19.7%2Bdtk2610.torch2130.2611241241.g0b8d6a)，适配 Python 3.12、DTK 26.10 和海光 Torch 2.13，对应源码提交 `0b8d6afe36f9d1e1c045449a3d16121aea4f69b3`。这是后续 nmz107 实验使用的预编译包来源。`--no-deps` 保留镜像中的海光 Torch 和现有依赖；`--target` 将 wheel 解包到独立目录，供第 4 节构建带补丁的副本。`/workspace/deepspeed-wheel` 和第 4 节的输出目录应使用新目录。

## 4. 获取并安装 TurboPhysAI

继续使用容器内的 `/usr/local/bin/python3` 执行：

```bash
git clone --branch alpamayo2 --single-branch \
  https://github.com/Coulsonq/TurboPhysAI.git /workspace/TurboPhysAI

/usr/local/bin/python3 -m pip install --no-build-isolation /workspace/TurboPhysAI
```

当前完整配置启用了四区域 ZeRO-3 compile，还需要构建实验中使用的 DeepSpeed native 副本。以下命令继续在容器内执行，需要 `patch`、C++ 编译器和镜像自带的 DTK 工具链：

```bash
source /opt/dtk/env.sh
mkdir -p /workspace/alpamayo2-compile
MAX_JOBS=2 /usr/local/bin/python3 \
  /workspace/TurboPhysAI/model_examples/Alpamayo2/deepspeed_patches/build_native.py \
  --package /workspace/deepspeed-wheel \
  --output /workspace/alpamayo2-compile/deepspeed_native

export PYTHONPATH="/workspace/alpamayo2-compile/deepspeed_native:${PYTHONPATH:-}"
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
import deepspeed
from deepspeed.ops.op_builder.dc import DeepCompileBuilder

root = Path('/workspace/alpamayo2-compile/deepspeed_native').resolve()
native = DeepCompileBuilder().load()
assert Path(deepspeed.__file__).resolve().is_relative_to(root)
assert Path(native.__file__).resolve().is_relative_to(root)
assert callable(getattr(native, 'set_reduce_process_group', None))
assert callable(getattr(native, 'get_reduce_group_name', None))
print('DeepSpeed:', deepspeed.__version__, deepspeed.__file__)
print('DeepCompile:', native.__file__)
PY
```

[构建脚本](deepspeed_patches/build_native.py)复制光源 wheel 到新的输出目录，依次应用三个补丁，并用该包自己的 builder 重编 DeepCompile 扩展：

- `native_reduce_group.patch`：为编译图的梯度归约接入现有独立通信组。
- `zero3_input_guards.patch`：仅对 ZeRO 参数处理释放后的 shape guard，保留普通输入的动态形状。
- `zero3_scheduler_kwargs.patch`：让反向调度器计入 keyword tensor 依赖。

后续实验实际加载的是 `/workspace/alpamayo2-compile/deepspeed_native/deepspeed/`。每次在新 shell 启动训练时，都需要先加载 DTK 环境并设置下面训练命令中的 `PYTHONPATH`。上述检查确认 Python 包和扩展的加载来源；训练是否成功仍以实际日志为准。

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

## 6. 单节点八卡 Stage 1 训练

| 配置 | 节点 / 卡数 | 相机 / 帧数 | 精度 | 每卡 batch size | 梯度累积 | 每进程 worker | 重计算 | DeepSpeed | 训练步数 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 官方配置 | 4 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 16 | 4 | 开启 | ZeRO-3 | 100000 | 默认保存 |
| 当前配置 | 1 节点 × 8 卡 | 7 路 / 每路 4 帧 | BF16 | 1 | 1 | 2 | 开启 | ZeRO-3 | 100000 | 不保存 |

训练和验证均使用 chunk 0，样本是否重叠取决于数据集划分；此配置用于调试，不能代表完整数据集训练或独立验证。

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
export PYTHONPATH="/workspace/alpamayo2-compile/deepspeed_native:${PYTHONPATH:-}"
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

根据实际目录修改 `MODEL_DIR` 和 `PAI_DIR`；默认启用 worker/OMP 优化组，使用每个 rank 2 个 worker 和 4 个 PyTorch CPU 线程；关闭该组后通过 `NUM_WORKERS` 调整 worker 数。通过 `GRADIENT_ACCUMULATION_STEPS` 和 `MAX_STEPS` 分别调整梯度累积次数和优化器更新步数。调整梯度累积次数会改变有效全局 batch size。`+trainer.save_strategy=no` 禁止保存 checkpoint；训练日志位于容器内 `/workspace/logs/`，对应宿主机 `/public/home/cuiqi/alpamayo2/logs/`。运行时配置设置 `HIP_BUFFER_EXTRA_SIZE=0`、`MIOPEN_FIND_MODE=1` 和 `PYTORCH_MIOPEN_SUGGEST_NHWC=1`。模型、数据和参数的实际使用以运行日志和生成的配置为准。

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
export PYTHONPATH="/workspace/alpamayo2-compile/deepspeed_native:${PYTHONPATH:-}"
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

## 9. 优化项

| 优化组 | 作用 | 第 6 节命令 |
| --- | --- | --- |
| `alpamayo2.dataloader_workers_omp` | 每个 rank 使用 2 个 DataLoader worker，设置 PyTorch CPU 线程数及 `OMP_NUM_THREADS` 为 4；保留原有进程启动方式 | 开启 |
| `alpamayo2.flash_attention` | 使用 FlashAttention 2 | 开启 |
| `alpamayo2.non_blocking_h2d` | 开启数据异步传输 | 开启 |
| `alpamayo2.channels_last_vision` | 将 Qwen3-VL 视觉 patch 的 Conv3d 输入转换为 `channels_last_3d`，配合 runtime 配置中的两个 MIOpen 环境变量 | 开启 |
| `alpamayo2.disable_cpu_offload` | 将 DeepSpeed 配置中已有的 CPU 优化器和参数 offload 关闭 | 开启 |
| `alpamayo2.bf16_communication` | BF16 训练时，将 DeepSpeed 梯度归约通信 dtype 设置为 BF16 | 开启 |
| `alpamayo2.zero3_comm_overlap` | 为 ZeRO-3 梯度 Reduce-Scatter 创建独立通信组，使参数 All-Gather 不被梯度打包和归约阻塞 | 开启 |
| `alpamayo2.zero3_module_compile` | 使用 native DeepSpeed 分别编译视觉 Encoder、Qwen 文本主干、Embedding 和 LM head；保留非重入 checkpoint，要求 ZeRO-3 且无 CPU offload | 开启 |

关闭 worker/OMP 组时，在 `turbo-physai run` 的 `--` 之前增加 `--disable-group alpamayo2.dataloader_workers_omp`。组开启后在训练参数初始化时设置线程数：PyTorch OpenMP 的线程数会调整；已初始化的其他第三方 OpenMP 库不一定受环境变量变更影响。

关闭 `channels_last_vision` 时，在 `turbo-physai run` 的 `--` 之前增加 `--disable-group alpamayo2.channels_last_vision`。优化组和运行时配置相互独立：仅关闭该组不会取消两个 MIOpen 环境变量。若也要取消环境变量，可将以下配置另存为自己的运行时配置，并用其路径替换训练命令中的 `--runtime-config`：

```yaml
schema_version: turbophysai/runtime-config/v1
kind: RuntimeConfig
environment:
  set:
    HIP_BUFFER_EXTRA_SIZE: "0"
  unset:
    - MIOPEN_FIND_MODE
    - PYTORCH_MIOPEN_SUGGEST_NHWC
process:
  numa: false
```

默认启用 `bf16_communication`：在 Accelerate 处理 DeepSpeed 配置前设置 `communication_data_type=bf16`，覆盖已有值；该组仅支持 BF16 混合精度训练。关闭该组时，在 `turbo-physai run` 的 `--` 之前增加 `--disable-group alpamayo2.bf16_communication`，恢复 Accelerate 的原有通信 dtype 选择。BF16 归约会改变浮点舍入，并覆盖当前 ROCm 环境中 Accelerate 的 FP32 通信保护；通信 dtype 与 H20 对齐不代表 loss 曲线或最终精度一致，需结合任务精度验证。

默认启用 `zero3_comm_overlap`，关闭方式为 `--disable-group alpamayo2.zero3_comm_overlap`。该组仅支持单节点、WORLD 成员与 rank 顺序一致的 NCCL 数据并行、`reduce_scatter=true`，不支持 sequence parallel 或 all-to-all 梯度归约。不改变通信 dtype、梯度缩放和优化器更新前的完成同步；不支持的配置会明确报错。实现保留在 Alpamayo2 优化目录，本次未迁移到公共 DeepSpeed 优化目录。

合入前的独立梯度通信组实现已在八卡 Stage 1、每卡 batch 1、GA 1、BF16 通信、无 CPU offload 下完成七步短训。去除首步后平均耗时从 18.9714 秒降至 15.4463 秒，全局吞吐从 0.4217 提高至 0.5179 sample/s（约 22.82%）；两个代表 rank 的第 3 步 trace 均确认梯度通信与主计算重叠。七步日志 loss 和 grad_norm 逐项相同，另有三步重计算小模型参数对照；这是合入前的短训数据，尚未验证完整 loss 曲线、跨机或其他并行布局。

CPU offload 组在配置中 `enabled: true` 表示关闭已有的 CPU offload；第 6 节启用该组，以满足 native compile 的无 CPU offload 要求。官方 `sft_stage1` 默认的 `configs/deepspeed/zero3.json` 本身没有 CPU offload，因此开启此组不会改变默认配置。若需要比较 CPU offload，可先关闭 `zero3_module_compile`，在训练参数中加 `trainer.deepspeed=configs/deepspeed/zero3_optimizer_offload.json`，再通过 `disable_cpu_offload` 组控制是否保留 CPU offload；关闭 offload 会增加显存占用。

`channels_last_vision` 与上述 MIOpen 设置已完成八卡 Stage 1 的 7 步短训；每卡 batch 1、使用 FP32 梯度通信且无优化器 CPU offload 时，第 2–7 步平均耗时由 23.7344 秒降至 23.3056 秒，全局吞吐提高约 1.84%。这是单次短训结果，完整 loss 曲线和最终精度尚未验证。
