# LeRobot ACT 模型应用说明

在海光 HCU 上运行 LeRobot ACT 原生训练和 TurboPhysAI 配置训练。

## 1. 环境准备

基础镜像包含 Python 3.12、海光 Torch 2.13 和 torchvision 0.28。

以下命令在宿主机执行，容器名可自行修改。

### 1.1 拉取基础镜像

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.13.0-ubuntu22.04-dtk26.10-py3.12-minimal
```

### 1.2 创建并进入容器

```bash
docker run -d \
  --name=lerobot-act-turbophysai-v2 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-drop=ALL \
  --cap-add=SYS_NICE \
  --security-opt no-new-privileges \
  --shm-size=64G \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -w /workspace \
  --entrypoint /bin/bash \
  harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.13.0-ubuntu22.04-dtk26.10-py3.12-minimal \
  -c 'sleep infinity' &&
docker exec -it lerobot-act-turbophysai-v2 bash
```

后续命令均在容器内执行。

## 2. 安装环境及源码

### 2.1 拉取官方源码

拉取 LeRobot 官方源码。

```bash
cd /workspace &&
test ! -e lerobot && test ! -L lerobot &&
git init lerobot &&
git -C lerobot remote add origin https://github.com/huggingface/lerobot.git &&
git -C lerobot fetch --depth 1 origin 3f2179f3b69708b6ad009b2e7685dd9d05269ee1 &&
git -C lerobot checkout --detach FETCH_HEAD
```

### 2.2 拉取并安装 TurboPhysAI

```bash
cd /workspace &&
test ! -e TurboPhysAI && test ! -L TurboPhysAI &&
git clone --branch ACT-turbo-gitlab-base \
  https://github.com/Coulsonq/TurboPhysAI.git TurboPhysAI &&
git -C TurboPhysAI rev-parse HEAD &&
env -u PYTHONPATH python -m pip install --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'setuptools>=71,<81' &&
env -u PYTHONPATH python -m pip install \
  --no-deps --no-build-isolation -e /workspace/TurboPhysAI
```

### 2.3 安装环境

安装 [requirements.txt](requirements.txt) 和 LeRobot 源码；安装 LeRobot 时使用 `--no-deps`。

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /workspace/TurboPhysAI/model_examples/LeRobot-ACT/requirements.txt

python -m pip install \
  --no-deps --no-build-isolation -e /workspace/lerobot

mkdir /workspace/act_ffmpeg4 &&
cd /workspace/act_ffmpeg4 &&
curl -fsSLO https://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavdevice58_4.4.2-0ubuntu0.22.04.1_amd64.deb &&
curl -fsSLO https://archive.ubuntu.com/ubuntu/pool/main/libc/libcdio-paranoia/libcdio-paranoia2_10.2+2.0.0-1build3_amd64.deb &&
curl -fsSLO https://archive.ubuntu.com/ubuntu/pool/main/libc/libcdio-paranoia/libcdio-cdda2_10.2+2.0.0-1build3_amd64.deb &&
printf '%s\n' \
  'b44a8a7c9e6e96bbb4f3a617c6a421bb42aefe1df6041c4a35f9f0b6b75f2399  libavdevice58_4.4.2-0ubuntu0.22.04.1_amd64.deb' \
  '6ebe35ba93f5fe2145133b9d56a54f2666ea5e15ce4fe475b6555d20412e6515  libcdio-paranoia2_10.2+2.0.0-1build3_amd64.deb' \
  'fe997e00ff8bce19be8b9ce40a9c0dd7edcbaaec15863d42026f25a9772881ff  libcdio-cdda2_10.2+2.0.0-1build3_amd64.deb' |
  sha256sum -c - &&
dpkg-deb -x libavdevice58_4.4.2-0ubuntu0.22.04.1_amd64.deb rootfs &&
dpkg-deb -x libcdio-paranoia2_10.2+2.0.0-1build3_amd64.deb rootfs &&
dpkg-deb -x libcdio-cdda2_10.2+2.0.0-1build3_amd64.deb rootfs
python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cpu 'torchcodec==0.12.0+cpu'
```

## 3. 下载数据和权重

### 3.1 下载数据集

设置 HF 镜像地址并下载 [LeRobot 官方数据集](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
python -c 'from huggingface_hub import snapshot_download; snapshot_download(
    repo_id="lerobot/aloha_mobile_cabinet",
    repo_type="dataset",
    revision="7a752b39f7e69de7e38aee485a6bab07528a061a",
    local_dir="/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet"
)'
```

### 3.2 下载权重

下载 ACT 使用的 ResNet18 预训练骨干权重：

```bash
TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch \
  python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)'
```

## 4. 运行训练

以下示例每卡 `batch_size=8`、`num_workers=4`；原生训练使用单卡运行 10 步，Turbo 训练默认使用 8 卡运行 100 步。Turbo 的进程数根据 `GPU_IDS` 自动计算。

### 4.1 原生训练

```bash
cd /workspace/lerobot
mkdir -p /workspace/act_runs
BATCH_SIZE=8
STAMP="$(date +%Y%m%d_%H%M%S)"
env -u PYTHONPATH HIP_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH="/workspace/act_ffmpeg4/rootfs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  python -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_mobile_cabinet \
    --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet \
    --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a \
    --dataset.video_backend=torchcodec \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=4 \
    --steps=10 \
    --log_freq=1 \
    --seed=42 \
    --wandb.enable=false \
    --output_dir=/workspace/act_runs/native_10step \
  2>&1 | tee -a "/workspace/act_runs/native_1gpu_bs${BATCH_SIZE}_${STAMP}.log"
```

### 4.2 TurboPhysAI 训练

使用 ACT 优化配置训练。`lerobot_act.apex_fused_adamw` 默认开启，需安装与当前 Torch 匹配的 Apex；该优化只对单设备 FP32 参数生效，参数缺失梯度时会报错停止。原 AdamW 与 Apex 的优化器断点不能互相恢复。

如需关闭 Apex，将 `configs/recipe.yaml` 中该组的 `enabled` 改为 `false`，再重新生成配置：

```bash
env -u PYTHONPATH turbo-physai optimization generate \
  --recipe /workspace/TurboPhysAI/turbo_physai/optimizations/models/lerobot_act/configs/recipe.yaml \
  --repo /workspace/lerobot \
  --commit 3f2179f3b69708b6ad009b2e7685dd9d05269ee1 \
  --output /workspace/TurboPhysAI/turbo_physai/optimizations/models/lerobot_act/configs/optimization.yaml \
  --force
```

```bash
cd /workspace/lerobot
mkdir -p /workspace/act_runs
GPU_IDS=0,1,2,3,4,5,6,7
GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPU_IDS}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=86
env -u PYTHONPATH HIP_VISIBLE_DEVICES="${GPU_IDS}" \
  LD_LIBRARY_PATH="/workspace/act_ffmpeg4/rootfs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  turbo-physai run \
    --optimization-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/lerobot_act/configs/optimization.yaml \
    --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/lerobot_act/configs/runtime.yaml \
    --log-report \
    torchrun --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
      -m lerobot.scripts.lerobot_train \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --dataset.repo_id=lerobot/aloha_mobile_cabinet \
      --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet \
      --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a \
      --dataset.video_backend=torchcodec \
      --batch_size="${BATCH_SIZE}" \
      --num_workers=8 \
      --prefetch_factor=2 \
      --steps=100 \
      --log_freq=1 \
      --wandb.enable=false \
      --output_dir="/workspace/act_runs/turbo_${GPU_COUNT}gpu_bs${BATCH_SIZE}_${STAMP}" \
  2>&1 | tee -a "/workspace/act_runs/ACT_BW1100_${GPU_COUNT}gpu_bs${BATCH_SIZE}_${STAMP}.log"
```
