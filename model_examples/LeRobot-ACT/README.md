# LeRobot ACT 模型应用说明

本文介绍在海光 HCU 上准备 LeRobot ACT 环境，并分别运行原生训练和 TurboPhysAI 空配置训练。当前不启用额外优化，也不修改官方模型源码。

此前使用虚拟环境在 BW1100 上以单卡、batch size 8、seed 42 分别完成 10 step 短训；卡与其他任务共享，结果不能作为性能或精度结论。以下改为容器全局安装的步骤尚未从零实测。

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

## 2. 安装环境及源码

### 2.1 拉取官方源码

拉取固定版本的 LeRobot 官方源码；安装依赖后再以可编辑方式安装。

```bash
(
cd /workspace &&
test ! -e lerobot && test ! -L lerobot &&
git init lerobot &&
git -C lerobot remote add origin https://github.com/huggingface/lerobot.git &&
git -C lerobot fetch --depth 1 origin 3f2179f3b69708b6ad009b2e7685dd9d05269ee1 &&
git -C lerobot checkout --detach FETCH_HEAD
)
```

### 2.2 拉取并安装 TurboPhysAI

TurboPhysAI 构建需要 `setuptools` 提供的 `pkg_resources`，因此先安装构建所需的版本；其余训练依赖在下一步安装。

```bash
(
cd /workspace &&
test ! -e TurboPhysAI && test ! -L TurboPhysAI &&
git clone --branch ACT-TurboPhysAI \
  https://github.com/Coulsonq/TurboPhysAI.git TurboPhysAI &&
git -C TurboPhysAI rev-parse HEAD &&
env -u PYTHONPATH python -m pip install --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'setuptools>=71,<81' &&
env -u PYTHONPATH python -m pip install \
  --no-deps --no-build-isolation -e /workspace/TurboPhysAI
)
```

TurboPhysAI 安装会构建原生扩展，需要镜像内具备相应编译工具。此前短训通过源码运行 Turbo，此安装方式尚未在本镜像中实测。

### 2.3 安装环境

从 [requirements.txt](requirements.txt) 安装 ACT 训练依赖到容器的全局 Python 环境。安装前生成依赖计划，若包含 `torch`、`torchvision` 或 `torchcodec` 则停止，避免替换镜像自带的厂商版本；其余已有依赖可能因版本要求被替换。清除继承的 `PYTHONPATH`，保证使用容器内的默认 Python 包路径。

```bash
(
test ! -e /workspace/act-deps-plan.json && test ! -L /workspace/act-deps-plan.json &&
env -u PYTHONPATH python -m pip install \
  --dry-run --report /workspace/act-deps-plan.json \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /workspace/TurboPhysAI/model_examples/LeRobot-ACT/requirements.txt &&

env -u PYTHONPATH python - <<'PY' &&
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
env -u PYTHONPATH python -m pip install \
  --no-deps --no-build-isolation -e /workspace/lerobot &&
env -u PYTHONPATH python -c \
  'import torch, torchvision, av; print(torch.__version__, torch.__file__); print(torchvision.__version__, torchvision.__file__); print("av", av.__version__)'
)
```

`pip check` 会报告上述 Torch 和 torchvision 版本不符合上游约束；若有其他依赖错误，需要另行处理。

## 3. 下载数据和权重

数据集使用 LeRobot 官方的 `lerobot/aloha_mobile_cabinet`，固定 revision 为 `7a752b39f7e69de7e38aee485a6bab07528a061a`；权重使用 ACT 默认的 torchvision ResNet18 预训练骨干。

### 3.1 下载数据集

使用 `huggingface_hub` 提供的 `hf` 命令从 [LeRobot 官方数据集](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)下载：

```bash
(
test ! -L /workspace/act_assets &&
test ! -L /workspace/act_assets/data &&
test ! -e /workspace/act_assets/data/LeRobot-ACT &&
test ! -L /workspace/act_assets/data/LeRobot-ACT &&
env -u PYTHONPATH -u HF_ENDPOINT -u HF_HUB_OFFLINE \
  hf download lerobot/aloha_mobile_cabinet \
  --repo-type dataset \
  --revision 7a752b39f7e69de7e38aee485a6bab07528a061a \
  --local-dir /workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet
)
```

### 3.2 下载权重

通过 torchvision 官方接口下载 `ResNet18_Weights.IMAGENET1K_V1` 到训练使用的 Torch 缓存目录。这是 ResNet18 骨干初始化权重，不是完整 ACT checkpoint。

```bash
(
test ! -L /workspace/act_assets &&
test ! -L /workspace/act_assets/ckpt &&
test ! -e /workspace/act_assets/ckpt/LeRobot-ACT &&
test ! -L /workspace/act_assets/ckpt/LeRobot-ACT &&
env -u PYTHONPATH TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch \
  python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)' &&
printf '%s  %s\n' \
  'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec' \
  '/workspace/act_assets/ckpt/LeRobot-ACT/torch/hub/checkpoints/resnet18-f37072fd.pth' | sha256sum -c -
)
```

以上命令要求对应目标目录尚不存在，以免覆盖已有数据。

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
  python -m lerobot.scripts.lerobot_train \
  "${TRAIN_ARGS[@]}" --output_dir=/workspace/act_runs/native_10step \
  2>&1 | tee /workspace/act_runs/native_10step.log
```

### 4.2 TurboPhysAI 空配置训练

以下命令复用上面的 `TRAIN_ARGS`，使用已安装的 Turbo CLI，不启用优化组或 RuntimeConfig。

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
  "${TRAIN_ARGS[@]}" --output_dir=/workspace/act_runs/turbo_10step \
  2>&1 | tee /workspace/act_runs/turbo_10step.log
```

空配置报告 `applied=0 skipped=0 blocked=0 failed=0`，表示未应用优化；训练结果以日志为准。完整训练、多卡、精度及性能尚未验证。


