# LeRobot ACT：HCU 环境与 TurboPhysAI 接入（初步验证）

## 1. 验证范围

本文记录在海光 HCU 上准备官方 LeRobot ACT、数据和权重，并分别运行原生训练及 TurboPhysAI 空配置训练的方法。本次不启用训练优化，不修改官方模型源码。

已在 `nmz104` 的 `lerobot-act-turbophysai-v2` 容器中完成两次单卡、batch size 8、seed 42、各 10 step 的短训；这是启动和数据链路验证，不是完整训练、精度或性能验证。测试时 0 号卡与其他任务共享，不应据此比较吞吐。

本文是初步构建步骤：关键命令、下载与短训已分别验证，但未在一个全新容器中逐字从头执行本文所有代码块；v2 实际复用了 v1 下载的数据和权重。安装计划中将 `setuptools` 合并进 dry-run，是对本次分开安装步骤的整理，也尚未整体复测。

| 项目 | 本次实际使用 |
|---|---|
| HCU | BW1100，单卡 |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.13.0-ubuntu22.04-dtk26.10-py3.12-minimal` |
| 镜像 digest | `sha256:6765b988bc75fce4b2f4f3ac92f337e12ab395597db449862f4022276bac22c6` |
| Python / Torch / torchvision | 3.12.12 / 2.13.0 / 0.28.0，后两者使用镜像自带海光版本 |
| LeRobot | `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`，源码包版本 0.6.1 |
| TurboPhysAI | `b2bb3eb9a7c342a88a350e863f0dd7299784ad26` |
| 数据集 | `lerobot/aloha_mobile_cabinet`，用户提供的 COS 数据包声明 revision `7a752b39f7e69de7e38aee485a6bab07528a061a` |
| 视频解码 | 官方支持的 `pyav`；此前 H20 实验使用 `torchcodec`，口径不同 |

**依赖边界：**该 LeRobot 提交声明 `torch<2.12`、`torchvision<0.27`；本镜像中的 2.13 / 0.28 超出该范围。本次是在指定镜像中的兼容性短训，不是完全满足官方依赖声明的复现。此前 H20 使用另一套 PyTorch 环境，不能直接据此声称精度或性能可比。

## 2. 拉取镜像并创建容器

以下命令在服务器上执行。按实际机器修改容器名及设备配置；不要占用他人的 HCU。

```bash
(
set -Eeuo pipefail
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.13.0-ubuntu22.04-dtk26.10-py3.12-minimal

docker run -d \
  --name lerobot-act-turbophysai-v2 \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --cap-drop=ALL --security-opt no-new-privileges \
  --shm-size 64g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -w /workspace --entrypoint /bin/bash \
  harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.13.0-ubuntu22.04-dtk26.10-py3.12-minimal \
  -c 'sleep infinity'

docker exec -it lerobot-act-turbophysai-v2 bash
)
```

后续命令均在容器内执行。这里采用设备映射和只读驱动挂载，没有使用 `--privileged` 或宿主 IPC；多卡训练需要按实际数据加载压力重新评估共享内存。

## 3. 获取源码

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

git -C lerobot rev-parse HEAD
git -C TurboPhysAI rev-parse HEAD
)
```

## 4. 安装训练依赖

镜像的 `PYTHONPATH` 会把系统包提前到虚拟环境之前。下列步骤仅为当前命令清除它，创建能够读取镜像厂商 Torch 的独立环境。不要对 LeRobot 执行会自动解析并降级 Torch 的普通依赖安装。

```bash
(
set -Eeuo pipefail
test ! -e /workspace/act_env && test ! -L /workspace/act_env
env -u PYTHONPATH python -m venv --system-site-packages /workspace/act_env

# 先生成依赖安装计划，不执行安装。可按网络情况更换可信 pip 镜像。
test ! -e /workspace/act-deps-plan.json || { echo '计划文件已存在，请更换路径' >&2; exit 1; }
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

# 审核计划：禁止出现厂商 Torch 栈，再安装该计划里的精确版本。
env -u PYTHONPATH /workspace/act_env/bin/python - <<'PY'
import json
import subprocess
import sys

with open('/workspace/act-deps-plan.json', encoding='utf-8') as f:
    plan = json.load(f)
items = [(entry['metadata']['name'], entry['metadata']['version']) for entry in plan['install']]
forbidden = {'torch', 'torchvision', 'torchcodec'}
if any(name.lower().replace('_', '-') in forbidden for name, _ in items):
    raise SystemExit('安装计划包含 Torch 栈，请检查后停止。')
for name, version in items:
    print(f'{name}=={version}')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-deps',
    '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
    *(f'{name}=={version}' for name, version in items),
], check=True)
PY

env -u PYTHONPATH /workspace/act_env/bin/python -m pip install \
  --no-deps --no-build-isolation -e /workspace/lerobot

env -u PYTHONPATH /workspace/act_env/bin/python -c \
  'import torch, torchvision, av; print(torch.__version__, torch.__file__); print(torchvision.__version__, torchvision.__file__); print("av", av.__version__)'
)

# 此检查预计因 Torch 和 torchvision 超出官方范围而返回非零。
env -u PYTHONPATH /workspace/act_env/bin/python -m pip check
```

最后的 `pip check` 在本镜像下会对 Torch 和 torchvision 版本上限报错，退出码非零；这是已知的官方依赖偏差，**不能忽略其他新增报错**。本次额外安装的关键版本为 `av==15.1.0`、`datasets==4.8.5`、`accelerate==1.15.0`；上面的范围解析可能随时间变化，实际安装以生成的计划为准。

## 5. 下载数据与预训练权重

本次数据和权重来自用户提供的 COS 镜像。数据包内的 `DATASET_SOURCE.txt` 声明上面的数据集和 revision；此声明不等同于已对 Hugging Face 全量文件逐一核验。权重是 ResNet18 骨干初始化权重，**不是完整的 ACT 模型 checkpoint**。

```bash
(
set -Eeuo pipefail
for path in /workspace/act_assets /workspace/act_assets/downloads /workspace/act_assets/data /workspace/act_assets/ckpt; do
  test ! -L "$path"
done
test ! -e /workspace/act_assets/data/LeRobot-ACT && test ! -L /workspace/act_assets/data/LeRobot-ACT
test ! -e /workspace/act_assets/ckpt/LeRobot-ACT && test ! -L /workspace/act_assets/ckpt/LeRobot-ACT
mkdir -p /workspace/act_assets/downloads /workspace/act_assets/data /workspace/act_assets/ckpt

curl --fail --location --retry 3 --continue-at - \
  --output /workspace/act_assets/downloads/data-LeRobot-ACT.tar.gz \
  https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/embodiedai/data/LeRobot-ACT.tar.gz
curl --fail --location --retry 3 --continue-at - \
  --output /workspace/act_assets/downloads/ckpt-LeRobot-ACT.tar.gz \
  https://hygon-thpcbench-1251001002.cos.ap-shanghai.myqcloud.com/hygon/embodiedai/ckpt/LeRobot-ACT.tar.gz

# 先拒绝绝对路径、..、链接和特殊文件，再解压；保留下载归档。
env -u PYTHONPATH /workspace/act_env/bin/python - <<'PY'
import pathlib
import tarfile

for path in (
    '/workspace/act_assets/downloads/data-LeRobot-ACT.tar.gz',
    '/workspace/act_assets/downloads/ckpt-LeRobot-ACT.tar.gz',
):
    with tarfile.open(path) as archive:
        for member in archive:
            parts = pathlib.PurePosixPath(member.name)
            if parts.is_absolute() or '..' in parts.parts or not (member.isfile() or member.isdir()):
                raise SystemExit(f'不安全的归档成员: {member.name}')
    print('checked:', path)
PY

tar -xzf /workspace/act_assets/downloads/data-LeRobot-ACT.tar.gz \
  --keep-old-files --no-same-owner -C /workspace/act_assets/data
tar -xzf /workspace/act_assets/downloads/ckpt-LeRobot-ACT.tar.gz \
  --keep-old-files --no-same-owner -C /workspace/act_assets/ckpt

cat /workspace/act_assets/data/LeRobot-ACT/DATASET_SOURCE.txt
sha256sum /workspace/act_assets/ckpt/LeRobot-ACT/torch/hub/checkpoints/resnet18-f37072fd.pth
)
```

本次 ResNet18 权重 SHA-256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。本次实际通过容器间复制复用了已下载的归档解压结果；上述 COS 下载和安全解压方法此前已在 v1 容器执行。

## 6. 原生 ACT 短训

下面仅作为 10 step 启动验证。`HIP_VISIBLE_DEVICES=0` 需改为本机可用卡；请先检查其他任务，不要擅自共享显卡。

```bash
cd /workspace/lerobot
mkdir -p /workspace/act_runs
test ! -e /workspace/act_runs/native_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
set -o pipefail

env -u PYTHONPATH HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  /workspace/act_env/bin/python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet \
  --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a \
  --dataset.video_backend=pyav --output_dir=/workspace/act_runs/native_10step \
  --batch_size=8 --steps=10 --log_freq=1 --seed=42 --wandb.enable=false \
  2>&1 | tee /workspace/act_runs/native_10step.log
```

## 7. TurboPhysAI 空配置短训

本次直接从 TurboPhysAI **源码目录**运行官方 CLI 的 Python 模块，没有构建或安装 Turbo wheel。`PYTHONPATH` 只包含该源码目录；LeRobot 通过前面的可编辑安装加载。以下空配置不启用任何优化组，也不加载 RuntimeConfig。

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

cd /workspace/lerobot
env PYTHONPATH=/workspace/TurboPhysAI /workspace/act_env/bin/python \
  -m turbo_physai.cli optimization check /workspace/optimization.empty.yaml --repo /workspace/lerobot

test ! -e /workspace/act_runs/turbo_10step.log || { echo '日志已存在，请更换路径' >&2; exit 1; }
set -o pipefail
env PYTHONPATH=/workspace/TurboPhysAI HIP_VISIBLE_DEVICES=0 \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  /workspace/act_env/bin/python -m turbo_physai.cli run \
  --optimization-config /workspace/optimization.empty.yaml --disable-numa --log-report -- \
  /workspace/act_env/bin/python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet \
  --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a \
  --dataset.video_backend=pyav --output_dir=/workspace/act_runs/turbo_10step \
  --batch_size=8 --steps=10 --log_freq=1 --seed=42 --wandb.enable=false \
  2>&1 | tee /workspace/act_runs/turbo_10step.log
```

不要省略 `--optimization-config`：不指定配置时，Turbo 可能加载公共优化组。本次报告为 `applied=0 skipped=0 blocked=0 failed=0`；这只说明空配置没有应用优化，不单独证明训练成功。

## 8. 本次结果与目录

- 原生训练：10/10 step，正常退出，保存末步 checkpoint；日志 `/workspace/act_runs/native_10step_v1.log`。
- Turbo 空配置：10/10 step，正常退出，保存末步 checkpoint；日志 `/workspace/act_runs/turbo_10step_v1.log`。
- 两次均使用同一源码、数据、权重、batch 及 seed；终端打印的 loss 接近，但不足以证明完整精度等价。
- 实际环境 `pip check` 仍因上述 Torch / torchvision 版本超范围退出非零；Torch 未被替换。
- `accelerate` 对宿主 Linux 4.18 内核发出低于建议版本的警告。本次短训未挂起，长期稳定性未验证。
- 本次没有构建 Turbo wheel，没有运行多卡、完整训练、任务精度评测或性能对比。

目录：`/workspace/lerobot`（官方源码）、`/workspace/TurboPhysAI`（Turbo 源码）、`/workspace/act_env`（Python 环境）、`/workspace/act_assets`（数据与权重）、`/workspace/act_runs`（日志及 checkpoint）。

官方参考：[ACT 文档](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/docs/source/act.mdx)、[固定版依赖声明](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/pyproject.toml)、[TurboPhysAI](https://github.com/HYGON-AI/TurboPhysAI)。
