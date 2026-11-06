# LeRobot ACT 模型应用说明

本文介绍在海光 HCU 上准备 LeRobot ACT 环境，并分别运行原生训练和 TurboPhysAI 空配置训练。当前不启用额外优化，也不修改官方模型源码。

已在 BW1100 上以单卡、batch size 8、seed 42 分别完成 10 step 短训；卡与其他任务共享，结果不能作为性能或精度结论。以下步骤根据实际操作整理，尚未在全新容器中逐条复跑。

## 1. 环境准备

本次使用 Python 3.12、镜像自带海光 Torch 2.13 和 torchvision 0.28。LeRobot 固定版本要求 Torch < 2.12、torchvision < 0.27，因此本次属于兼容性短训，不完全符合上游依赖要求。

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

单机八卡使用独立的容器网络和 IPC；多机训练需根据实际通信配置调整。放宽 `memlock` 的容器启动方式尚未单独验证。

后续命令均在容器内执行。

## 2. 下载源码并安装依赖

固定源码版本：LeRobot `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`、TurboPhysAI `b2bb3eb9a7c342a88a350e863f0dd7299784ad26`。

```bash
(
set -Eeuo pipefail
cd /workspace
test ! -e lerobot && test ! -L lerobot
test ! -e TurboPhysAI && test ! -L TurboPhysAI
git init lerobot
git -C lerobot remote add origin https://github.com/huggingface/lerobot.git
git -C lerobot fetch --depth 1 origin 3f2179f3b69708b6ad009b2e7685dd9d05269ee1
git -C lerobot checkout --detach FETCH_HEAD
git clone https://github.com/HYGON-AI/TurboPhysAI.git TurboPhysAI
git -C TurboPhysAI checkout --detach b2bb3eb9a7c342a88a350e863f0dd7299784ad26
)
```

创建独立环境并保留镜像自带的 Torch。安装前生成依赖计划，若包含 `torch`、`torchvision` 或 `torchcodec` 则停止，避免替换厂商版本；清除继承的 `PYTHONPATH` 以保证虚拟环境生效。

```bash
(
set -Eeuo pipefail
test ! -e /workspace/act_env && test ! -L /workspace/act_env
test ! -e /workspace/act-deps-plan.json && test ! -L /workspace/act-deps-plan.json
env -u PYTHONPATH python -m venv --system-site-packages /workspace/act_env
env -u PYTHONPATH /workspace/act_env/bin/python -m pip install \
  --dry-run --report /workspace/act-deps-plan.json \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'numpy>=2,<2.3' 'opencv-python-headless>=4.9,<4.14' 'Pillow>=10,<13' \
  'einops>=0.8,<0.9' 'draccus==0.10.0' 'huggingface-hub>=1,<2' \
  'requests>=2.32,<3' 'gymnasium>=1.1.1,<2' 'safetensors>=0.4.3,<1' \
  'packaging>=24.2,<26' 'termcolor>=2.4,<4' 'tqdm>=4.66,<5' \
  'datasets>=4.7,<5' 'pandas>=2,<3' 'pyarrow>=21,<30' 'av>=15,<16' \
  'jsonlines>=4,<5' 'wandb>=0.24,<0.28' 'accelerate>=1.14,<2' \
  'PyYAML>=6.0' 'setuptools>=71,<81' scipy

env -u PYTHONPATH /workspace/act_env/bin/python - <<'PY'
import json, subprocess, sys
with open('/workspace/act-deps-plan.json', encoding='utf-8') as f:
    items = [(x['metadata']['name'], x['metadata']['version']) for x in json.load(f)['install']]
if any(name.lower().replace('_', '-') in {'torch', 'torchvision', 'torchcodec'} for name, _ in items):
    raise SystemExit('依赖计划包含 Torch 栈，请检查后重新配置。')
if items:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--index-url',
                    'https://pypi.tuna.tsinghua.edu.cn/simple',
                    *(f'{name}=={version}' for name, version in items)], check=True)
PY
env -u PYTHONPATH /workspace/act_env/bin/python -m pip install \
  --no-deps --no-build-isolation -e /workspace/lerobot
env -u PYTHONPATH /workspace/act_env/bin/python -c \
  'import torch, torchvision, av; print(torch.__version__, torch.__file__); print(torchvision.__version__, torchvision.__file__); print("av", av.__version__)'
)
```

`pip check` 会报告上述 Torch 和 torchvision 版本不符合上游约束；若有其他依赖错误，需要另行处理。

## 3. 下载数据和权重

使用提供的 COS 数据包，数据集为 `lerobot/aloha_mobile_cabinet`，数据包声明 revision `7a752b39f7e69de7e38aee485a6bab07528a061a`。权重是 ResNet18 骨干初始化权重，不是完整 ACT checkpoint。

```bash
(
set -Eeuo pipefail
for path in /workspace/act_assets /workspace/act_assets/downloads /workspace/act_assets/data /workspace/act_assets/ckpt; do
  test ! -L "$path"
done
test ! -e /workspace/act_assets/data/LeRobot-ACT && test ! -L /workspace/act_assets/data/LeRobot-ACT
test ! -e /workspace/act_assets/ckpt/LeRobot-ACT && test ! -L /workspace/act_assets/ckpt/LeRobot-ACT
mkdir -p /workspace/act_assets/{downloads,data,ckpt}
BASE=https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/embodiedai
for kind in data ckpt; do
  curl --fail --location --retry 3 --continue-at - \
    -o "/workspace/act_assets/downloads/${kind}-LeRobot-ACT.tar.gz" "$BASE/$kind/LeRobot-ACT.tar.gz"
done
# 解压前拒绝绝对路径、上级路径、链接和特殊文件。
env -u PYTHONPATH /workspace/act_env/bin/python - <<'PY'
import pathlib, tarfile
for kind in ('data', 'ckpt'):
    with tarfile.open(f'/workspace/act_assets/downloads/{kind}-LeRobot-ACT.tar.gz') as archive:
        for member in archive:
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise SystemExit(f'不安全的归档成员: {member.name}')
PY
for kind in data ckpt; do
  tar -xzf "/workspace/act_assets/downloads/${kind}-LeRobot-ACT.tar.gz" \
    --keep-old-files --no-same-owner -C "/workspace/act_assets/$kind"
done
sha256sum /workspace/act_assets/ckpt/LeRobot-ACT/torch/hub/checkpoints/resnet18-f37072fd.pth
)
```

本次 ResNet18 权重 SHA-256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 4. 运行训练

本次使用 `pyav` 解码，之前的 H20 测试使用 `torchcodec`，两者不能直接作为严格性能或精度对比。运行前请将 `HIP_VISIBLE_DEVICES` 改为实际空闲卡。

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
  --dataset.video_backend=pyav --batch_size=8 --steps=10 --log_freq=1
  --seed=42 --wandb.enable=false
)
test ! -e /workspace/act_runs/native_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
env -u PYTHONPATH HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  /workspace/act_env/bin/python -m lerobot.scripts.lerobot_train \
  "${TRAIN_ARGS[@]}" --output_dir=/workspace/act_runs/native_10step \
  2>&1 | tee /workspace/act_runs/native_10step.log
```

### 4.2 TurboPhysAI 空配置训练

以下命令复用上面的 `TRAIN_ARGS`，通过源码运行 Turbo CLI，不安装 wheel、不启用优化组或 RuntimeConfig。

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

env PYTHONPATH=/workspace/TurboPhysAI /workspace/act_env/bin/python \
  -m turbo_physai.cli optimization check /workspace/optimization.empty.yaml --repo /workspace/lerobot
test ! -e /workspace/act_runs/turbo_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
env PYTHONPATH=/workspace/TurboPhysAI HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  /workspace/act_env/bin/python -m turbo_physai.cli run \
  --optimization-config /workspace/optimization.empty.yaml --disable-numa --log-report -- \
  /workspace/act_env/bin/python -m lerobot.scripts.lerobot_train \
  "${TRAIN_ARGS[@]}" --output_dir=/workspace/act_runs/turbo_10step \
  2>&1 | tee /workspace/act_runs/turbo_10step.log
```

空配置报告 `applied=0 skipped=0 blocked=0 failed=0`，表示未应用优化；训练结果以日志为准。完整训练、多卡、精度及性能尚未验证。
