# LeRobot ACT 优化来源

标准配置为 `configs/recipe.yaml` 和生成的 `configs/optimization.yaml`，启用七个可兼容的 Group，包括 LayerNorm Apex。空优化组配置保留在 `configs/optimization.baseline.yaml`；运行时使用 `configs/runtime.yaml`。修改默认启用状态时应调整 Recipe 并重新生成 OptimizationConfig。

## 参考文件夹：LeRobot-ACT

- `lerobot_act.ddp_find_unused`：ACT 的 DDP 设置 `find_unused_parameters=False`；对应 `patches/0100-act-dcu-training-optimization.patch`。
- `lerobot_act.channels_last_backbone_compile` 中的 `channels_last` 部分：视觉 backbone、输入和图像投影使用 `channels_last`；同一个补丁还在保存优化器状态时处理非连续张量。
- `configs/runtime.yaml` 的 `MIOPEN_FIND_MODE`、`MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32` 和 `PYTORCH_MIOPEN_SUGGEST_NHWC`：对应 `start_act.sh`。

## ACT-H20 实验

- `lerobot_act.channels_last_backbone_compile` 中的 `torch.compile` 部分：编译视觉 backbone。它和参考文件夹的布局优化共用同一个 Turbo 替换目标，因此合在一个 Group 中。
- `lerobot_act.sampler_vector`：批量计算采样索引。
- `lerobot_act.ffn_backward_fusion`：融合 ReLU 和 dropout 的反向；当前实现仅命中 FP32、每卡 batch 8 的三种 ACT 形状，其余情况回退。
- `lerobot_act.attention_compile`：受 ACT-H20 attention 反向融合启发，在 HCU 上编译 ACT encoder 的 attention；这是 PyTorch/Inductor 的 HCU 实现，不是 H20 CUDA kernel 的直接移植。
- `lerobot_act.pool_backward_gather`：受 ACT-H20 Pool 反向 gather 启发，使用 HCU Triton 实现 PyTorch 原生 max-pool 的反向；这是新实现，只涵盖 pool 反向，不是 H20 的 pool＋ReLU＋FrozenBN 完整融合。当前仅命中 FP32、每卡 batch 8 的 ACT 形状。

H20 原版 Pool、Attention 反向 kernel 及 cuDNN `benchmark_limit` 依赖 NVIDIA 或固定 batch 86，尚未直接作为 HCU 优化启用。Attention 和 Pool 候选的收益尚需进一步复测。

本配置用于性能实验；TF32 和 compile 等的训练精度尚未验收。H20 本地实验代码的来源与许可仍需在正式发布前确认。

## HCU 实验：LayerNorm 反向

- `lerobot_act.layernorm_apex_backward`：在 ACTPolicy 前向范围内，对实测的 `(902, 8, 512)` FP32 LayerNorm 形状复用已安装的 Apex 反向实现；其他形状、设备、编译或不支持的条件回退原实现。
- Group 在标准 Recipe 中默认启用。此前测速使用的是独立实验代码；当前作用域收紧后的版本尚未重新测量或完成训练精度验证。
- `runtime.yaml` 中的 rank CPU 绑核沿用 nmz104 v3 八卡实验配置，不启用 NUMA 内存绑定；其他机器或设备顺序需按实际拓扑调整。
