# LeRobot ACT 模型应用说明

在海光 HCU 上运行 LeRobot ACT 原生训练和 TurboPhysAI 空配置训练。

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
git clone --branch ACT-TurboPhysAI \
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
```

## 3. 下载数据和权重

### 3.1 下载数据集

设置 HF 镜像地址并下载 [LeRobot 官方数据集](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
  hf download lerobot/aloha_mobile_cabinet \
  --repo-type dataset \
  --revision 7a752b39f7e69de7e38aee485a6bab07528a061a \
  --local-dir /workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet
```

### 3.2 下载权重

下载 ACT 使用的 ResNet18 预训练骨干权重：

```bash
TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch \
  python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)'
```

## 4. 运行训练

以下示例均为单卡：`HIP_VISIBLE_DEVICES=0`、每卡 `batch_size=8`、`num_workers=4`，训练 `steps=10`。可按需修改对应参数。

### 4.1 原生训练

```bash
cd /workspace/lerobot
mkdir -p /workspace/act_runs
set -o pipefail
TRAIN_ARGS=(
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
  --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet
  --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a
  --dataset.video_backend=pyav --log_freq=1
  --seed=42 --wandb.enable=false
)
test ! -e /workspace/act_runs/native_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
env -u PYTHONPATH HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  python -m lerobot.scripts.lerobot_train \
  "${TRAIN_ARGS[@]}" \
  --batch_size=8 \
  --num_workers=4 \
  --steps=10 \
  --output_dir=/workspace/act_runs/native_10step \
  2>&1 | tee /workspace/act_runs/native_10step.log
```

### 4.2 TurboPhysAI 空配置训练

先执行 4.1，复用其中的 `TRAIN_ARGS`；以下命令不启用优化组或 RuntimeConfig。

```bash
test ! -e /workspace/optimization.empty.yaml || { echo '配置已存在，请更换路径' >&2; exit 1; }
printf '%s\n' \
  'schema_version: turbophysai/optimization-config/v1' \
  'kind: OptimizationConfig' \
  'metadata:' \
  '  id: model.lerobot_act.baseline' \
  '  version: "1.0.0"' \
  'optimization_modules: []' \
  'optimization_groups: []' > /workspace/optimization.empty.yaml

env -u PYTHONPATH python \
  -m turbo_physai.cli optimization check /workspace/optimization.empty.yaml --repo /workspace/lerobot
test ! -e /workspace/act_runs/turbo_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
env -u PYTHONPATH HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  python -m turbo_physai.cli run \
  --optimization-config /workspace/optimization.empty.yaml --disable-numa --log-report -- \
  python -m lerobot.scripts.lerobot_train \
  "${TRAIN_ARGS[@]}" \
  --batch_size=8 \
  --num_workers=4 \
  --steps=10 \
  --output_dir=/workspace/act_runs/turbo_10step \
  2>&1 | tee /workspace/act_runs/turbo_10step.log
```

