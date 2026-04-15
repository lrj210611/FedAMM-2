# Module 2 Change Log

Date: 2026-04-14
Scope: FedMASS Module 2 only (`Mask-aware Distilled Pseudo-label Filtering`)

## 1. 修改文件列表

- `FedAMM/options.py`
- `FedAMM/train.py`
- `FedAMM/utils/fedmass_pseudo.py`
- `FedAMM/dataset/datasets_nii.py`

说明：

- 本次没有继续修改 `models/rfnet.py`。
- Module 2 复用了 Module 1 已经补好的 `forward_fused_features(...)` 接口。

## 2. 新增参数列表

在 `options.py` 中新增了以下 Module 2 参数：

- `--enable_pseudo_filtering`
- `--ema_decay`
- `--pseudo_loss_weight`
- `--pseudo_warmup_rounds`
- `--pseudo_conf_base`
- `--pseudo_conf_gamma`
- `--pseudo_temperature`
- `--pseudo_consistency_eps`
- `--pseudo_lambda_ce`
- `--pseudo_lambda_dice`
- `--pseudo_lambda_anchor`
- `--pseudo_log_interval`

额外新增了一个数据入口参数：

- `--unlabeled_train_file`

用途：

- 用于为每个客户端提供无标注缺失模态数据源。
- 若该项为 `None`，当前实现会退化为复用该客户端已有 `train_file` 作为 Module 2 数据源。

## 3. 新增函数列表

### `utils/fedmass_pseudo.py`

- `create_ema_teacher`
- `update_ema_teacher`
- `get_available_modality_count`
- `compute_mask_aware_weight`
- `compute_mask_aware_threshold`
- `compute_anchor_similarity_probs`
- `refine_pseudo_probs`
- `build_pseudo_label_filters`
- `compute_unsupervised_pseudo_loss`
- `extract_unlabeled_prototypes`
- `compute_unlabeled_anchor_alignment_loss`
- `compute_pseudo_filtering`

### `dataset/datasets_nii.py`

- `Brats_loadall_unlabeled_missing_nii`

### `train.py`

- 扩展了 `local_training(...)` 的可选参数：
  - `unlabeled_dataloader=None`
  - `teacher_model=None`
  - `global_anchor_bank=None`

## 4. Module 2 数据流说明

当前 Module 2 的数据流为：

```text
unlabeled missing-modal batch
        ↓
dataset 返回 weak_x / strong_x / mask / mask_id / name
        ↓
EMA teacher 用 weak_x + mask 做 fused prediction / fused feature
        ↓
student 用 strong_x + mask 做 fused prediction / fused feature
        ↓
若 global_anchor_bank 存在：
        计算 teacher feature 与 anchor bank 的 cosine similarity
        得到 anchor_probs
否则：
        anchor_probs = None
        自动退化为 teacher-only pseudo-label
        ↓
根据 mask 计算 omega_s = |s| / 4
        ↓
融合 teacher_probs 与 anchor_probs
        ↓
得到 refined pseudo probs
        ↓
三重过滤：
1. mask-aware confidence threshold
2. teacher-student consistency filter
3. anchor agreement filter
        ↓
得到 selected_mask 与 pseudo_labels
        ↓
仅在 selected_mask 区域计算 pseudo CE + Dice loss
        ↓
从 selected_mask 区域提取 unlabeled prototypes
        ↓
若 global_anchor_bank 存在，计算 prototype-anchor alignment loss
        ↓
total pseudo loss = pseudo CE/Dice + lambda_anchor * anchor alignment
        ↓
乘 pseudo_loss_weight 加入本地总 loss
        ↓
iteration 结束后更新 EMA teacher
```

## 5. 兼容性保证说明

当前实现保证以下兼容性：

### A. `enable_anchor_supervision=False` 且 `enable_pseudo_filtering=False`

- Module 1 与 Module 2 均不启用。
- 本地训练仍走原来的 FedAMM 主路径。
- 不会额外取 unlabeled batch，不会创建 teacher，不会引入伪标签损失。

### B. `enable_anchor_supervision=True` 且 `enable_pseudo_filtering=False`

- 行为保持为当前 Module 1。
- 不会启用 EMA teacher，不会引入伪标签损失。

### C. `enable_pseudo_filtering=True` 且 `global_anchor_bank=None`

- Module 2 自动退化为 `teacher-only pseudo-label`。
- `anchor_probs = None`
- `anchor agreement filter` 自动全 True。
- `anchor alignment loss` 自动为 0。

### D. `unlabeled_dataloader=None`

- Module 2 自动跳过。
- 不报错。
- 不影响原有监督/Module 1 流程。

### E. `selected_mask` 全 False

- 无监督 CE/Dice loss 自动返回 0。
- 不会产生 NaN。

### F. anchor feature 维度不匹配

- `compute_anchor_similarity_probs(...)` 自动返回 `None`。
- Module 2 自动退化为 teacher-only。

## 6. 当前没有实现的内容

本次明确没有实现以下内容：

- server-side full-modal anchor bank update
- server-side missing-pattern prototype bank update
- reliability-aware aggregation
- client reliability score
- Module 3 的全模型/encoder 可靠聚合

这些内容都属于 Module 3，不在本次范围内。

当前 `global_anchor_bank` 在 `train.py` 中仍为只读输入，默认值是 `None`。
也就是说：

- Module 2 已经支持“使用 global anchor 修正伪标签”的逻辑；
- 但当前工程里还没有实现服务端去更新并维护这个 bank；
- 因此当前默认运行时会自然退化为 teacher-only。

## 7. 潜在风险点

### 1. 当前 unlabeled 数据源的临时假设

若 `unlabeled_train_file[client_id]` 没有提供，当前实现会回退为：

- 复用该客户端原有 `train_file`

这保证了 Module 2 可以先跑通，但它仍然只是临时方案。
严格的 FedMASS 设定下，后续应提供独立的 `D_k^u` 数据划分。

### 2. EMA teacher 的多进程成本

当前实现支持把 `teacher_model` 作为 `local_training(...)` 的参数传入。
当 `use_multiprocessing=True` 时，teacher 的序列化和回传会带来额外开销。
功能上可工作，但后续若继续扩展，建议把 teacher state 改成更轻量的 client state 管理方式。

### 3. 目前 global anchor bank 默认不存在

因为 Module 3 尚未实现，当前训练时大概率不会实际使用 anchor-guided refinement，只会执行 teacher-only 路径。
这不是 bug，而是当前阶段按范围裁剪后的预期行为。

### 4. 当前 Module 2 只接入 fused path

本次伪标签分支使用的是 fused logits / fused feature。
没有继续扩展到 PRM 辅助头或更复杂的一致性分支。
这是刻意保持低耦合的结果。

## 8. 下一步建议

建议下一步优先做以下两件事中的一个：

### 路线 A：先补数据闭环

- 为每个客户端显式生成：
  - labeled full-modal split
  - unlabeled missing-modal split
- 避免 Module 1 / Module 2 都回退复用 `train_file`

### 路线 B：进入 Module 3

- 引入全局 `full-modal anchor bank`
- 聚合 `local_full_anchors`
- 聚合 `local_unlabeled_prototypes`
- 计算 `client reliability`
- 将 `global_anchor_bank` 真正注入 Module 2

## 9. 编译检查

已执行语法编译检查并通过：

- `python -m py_compile options.py`
- `python -m py_compile train.py`
- `python -m py_compile utils/fedmass_pseudo.py`
- `python -m py_compile dataset/datasets_nii.py`

说明：

- 本次只验证语法编译；
- 不使用缺失的外部医学依赖去做运行时导入验证；
- 因此不会把 `nibabel` 一类环境缺失误判为代码错误。

## 10. 本次范围声明

请特别注意：

- 当前 Module 2 **不实现 server-side anchor bank update**
- 当前 Module 2 **不实现 reliability-aware aggregation**
- 这两部分都属于 **Module 3**

本次实现目标是：

- 低耦合
- 可关闭
- 可降级
- 不破坏原 FedAMM / 当前 Module 1 行为
