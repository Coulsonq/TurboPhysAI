# Alpamayo 2 Super 模型应用说明

本文介绍在海光 HCU 环境中准备 Alpamayo 2 Super 源码、依赖、权重和数据，并运行单机八卡 Stage 1 训练短测。

## 1. 模型简介

[Alpamayo 2 Super](https://github.com/NVlabs/alpamayo2) 是用于自动驾驶的视觉语言动作模型。[官方 SFT 配方](https://github.com/NVlabs/alpamayo-recipes/tree/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft)分为两个阶段：Stage 1 微调视觉语言模型，Stage 2 冻结视觉语言模型并训练动作 Expert。

本文记录的是基于公开权重和数据的 Stage 1 短测，不是完整训练结果。目前尚未接入 Alpamayo 2 的 TurboPhysAI 优化配置，训练命令不使用 `turbo-physai run`。

## 2. 代码及环境

本次测试使用以下版本：

| 项目 | 版本 |
| --- | --- |
| 训练代码 | [NVlabs/alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes/tree/622f9d23ff9e4961ba8c6aac71367776cdf6b016)，commit `622f9d23ff9e4961ba8c6aac71367776cdf6b016` |
| 模型代码 | [NVlabs/alpamayo2](https://github.com/NVlabs/alpamayo2/tree/9596749e33552468f3ae70da8c237bdd0401e59a)，commit `9596749e33552468f3ae70da8c237bdd0401e59a` |
| 运行环境 | Python 3.12、DTK 26.10、海光 PyTorch 2.13 |
| 训练组件 | Transformers 4.57.1、DeepSpeed 0.19.1 |
| 测试设备 | 单机 8 张 BW1100 HCU |

请先根据[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)准备包含对应 DTK、海光 PyTorch 和 HCU 运行库的环境，并确认容器可访问 8 张卡。以下操作均在容器内执行。

官方依赖配置面向 CUDA 环境并固定了其他版本的 PyTorch。HCU 环境应保留镜像中的海光 PyTorch，不要直接执行可能覆盖该版本的 `uv sync`。

## 3. 安装源码及依赖

### 3.1 获取源码

```bash
export ALPA_WORKSPACE=/workspace/alpamayo2_task
mkdir -p "$ALPA_WORKSPACE"

git clone https://github.com/NVlabs/alpamayo-recipes.git "$ALPA_WORKSPACE/alpamayo-recipes"
git -C "$ALPA_WORKSPACE/alpamayo-recipes" checkout --detach 622f9d23ff9e4961ba8c6aac71367776cdf6b016

git clone https://github.com/NVlabs/alpamayo2.git "$ALPA_WORKSPACE/alpamayo2"
git -C "$ALPA_WORKSPACE/alpamayo2" checkout --detach 9596749e33552468f3ae70da8c237bdd0401e59a
```

### 3.2 安装依赖

使用系统环境中的海光 PyTorch 创建虚拟环境：

```bash
/usr/local/bin/python3 -m venv --system-site-packages "$ALPA_WORKSPACE/venv"
source "$ALPA_WORKSPACE/venv/bin/activate"

python -m pip install \
  huggingface_hub==0.36.0 transformers==4.57.1 accelerate==1.14.0 \
  av==18.0.0 hydra-core==1.3.4 hydra-colorlog==1.2.0 \
  mediapy==1.2.7 physical-ai-av==0.2.2 torchmetrics==1.8.2 \
  opencv-python==5.0.0.93 rich==15.0.0 wandb==0.28.1

DS_BUILD_OPS=0 python -m pip install --no-build-isolation deepspeed==0.19.1

python -m pip install --no-deps -e "$ALPA_WORKSPACE/alpamayo2" \
  -e "$ALPA_WORKSPACE/alpamayo-recipes/src" \
  -e "$ALPA_WORKSPACE/alpamayo-recipes/recipes/alpamayo2_sft"
```

`DS_BUILD_OPS=0` 表示安装时不预编译 DeepSpeed 算子；实际运行时所需扩展可能按需编译，需要可用的 DTK 编译工具链。安装后确认导入的 PyTorch 仍为海光版本且能够访问 HCU：

```bash
python -c 'import torch, deepspeed; print("torch:", torch.__version__, "HIP:", torch.version.hip, "DeepSpeed:", deepspeed.__version__); print("device count:", torch.cuda.device_count()); print(torch.ones(1, device="cuda"))'
```

若出现 HCU 运行库缺失，应先按镜像及设备环境要求配置运行库，再进行训练。不同镜像的依赖可能存在差异；上述安装步骤需在目标环境验证。

## 4. 下载权重与数据

模型和数据集受访问条件约束。请先阅读并确认 [Alpamayo 2 Super 模型许可](https://huggingface.co/nvidia/Alpamayo2-Super)和 [Physical AI 数据集许可](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)适用于实际使用场景，然后使用具备访问权限的 Hugging Face 账号登录。不要在命令、日志或文档中填写访问令牌。

```bash
hf auth login
hf auth whoami
```

### 4.1 下载预训练权重

```bash
export MODEL_DIR="$ALPA_WORKSPACE/checkpoints/alpamayo2-super"
hf download nvidia/Alpamayo2-Super \
  --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe \
  --local-dir "$MODEL_DIR"
```

请预留足够的磁盘空间，模型快照应包含权重分片、模型配置、tokenizer 和 processor 文件。

### 4.2 下载短测数据

以下命令下载数据集的 chunk 0，包含 7 路相机、车辆运动信息和所需标定数据。`0-1` 为左闭右开范围，不包含 chunk 1。

```bash
export PAI_DIR="$ALPA_WORKSPACE/PhysicalAI-Autonomous-Vehicles-stage1-7cam"
python "$ALPA_WORKSPACE/alpamayo-recipes/scripts/download_pai.py" \
  --revision 14ed7a77897edafb113dc86fde8b8173ae7fb028 \
  --chunk-ids 0-1 \
  --camera camera_cross_left_120fov camera_front_wide_120fov \
    camera_cross_right_120fov camera_rear_left_70fov \
    camera_rear_tele_30fov camera_rear_right_70fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "$PAI_DIR"
```

该数据仅用于短测；下面的命令将训练和验证路径都指向 chunk 0，不构成独立验证集。完整训练的数据划分请参考[官方说明](https://github.com/NVlabs/alpamayo-recipes/blob/622f9d23ff9e4961ba8c6aac71367776cdf6b016/recipes/alpamayo2_sft/README.md#prepare-pai-chunks-099)。

## 5. 运行单机八卡训练短测

以下配置使用 7 路相机、每路 4 帧，开启梯度重计算，单卡 batch size 为 1，梯度累积为 1，全局 batch size 为 8；DeepSpeed 使用 ZeRO-3 和优化器 CPU offload。首次运行可能需要编译扩展。

- `HIP_BUFFER_EXTRA_SIZE=0`：按本次测试配置设置 HCU 运行时环境变量。
- `trainer.max_steps`：训练步数，此处设为 6，用于短测。
- `trainer.dataloader_num_workers`：每个训练进程的数据加载 worker 数，此处设为 0。
- `+trainer.save_strategy=no`：短测不保存 checkpoint；需要保存时应调整保存策略。
- 日志写入 `$ALPA_WORKSPACE/logs`，训练输出写入 `$ALPA_WORKSPACE/outputs`。

```bash
cd "$ALPA_WORKSPACE/alpamayo-recipes/recipes/alpamayo2_sft"
mkdir -p "$ALPA_WORKSPACE/logs" "$ALPA_WORKSPACE/torch_extensions"
STAMP="$(date +%Y%m%d_%H%M%S)"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_BUFFER_EXTRA_SIZE=0
export OMP_NUM_THREADS=1
export MAX_JOBS=8
export WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR="$ALPA_WORKSPACE/torch_extensions"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
set -o pipefail

python -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  -m alpamayo2_sft.train_hf \
  --config-path pkg://alpamayo2_sft/configs \
  --config-name sft_stage1 \
  model.pretrained_model_name_or_path="$MODEL_DIR" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.chunk_ids=0-1 \
  data.val_dataset.chunk_ids=0-1 \
  paths.output_dir="$ALPA_WORKSPACE/outputs/stage1_8hcu_${STAMP}" \
  trainer.max_steps=6 \
  trainer.gradient_checkpointing=true \
  trainer.gradient_accumulation_steps=1 \
  trainer.per_device_train_batch_size=1 \
  trainer.deepspeed=configs/deepspeed/zero3_optimizer_offload.json \
  trainer.dataloader_num_workers=0 \
  trainer.logging_steps=1 \
  +trainer.save_strategy=no \
  trainer.report_to=none \
  2>&1 | tee "$ALPA_WORKSPACE/logs/stage1_8hcu_${STAMP}.log"
```

以上命令与已完成步骤的短测使用相同的主要训练配置；本次记录中运行完成 6 step 后手动停止。短测结果不能代表完整训练的收敛精度或稳定吞吐，显存占用也会受卡上其他进程影响。Stage 2、完整数据集及长期训练请分别验证。
