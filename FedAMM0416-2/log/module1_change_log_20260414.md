# Module 1 Change Log

Date: 2026-04-14
Scope: FedMASS Module 1 only (`Full-modal Anchor Supervision`)

## Files Changed

- `FedAMM/options.py`
- `FedAMM/dataset/datasets_nii.py`
- `FedAMM/models/rfnet.py`
- `FedAMM/train.py`
- `FedAMM/utils/fedmass_anchor.py`

## What Changed

### 1. `options.py`

- Added Module 1 switches and weights:
  - `--enable_anchor_supervision`
  - `--anchor_warmup_rounds`
  - `--anchor_batch_size`
  - `--anchor_loss_weight`
  - `--anchor_lambda_seg`
  - `--anchor_lambda_sep`
  - `--anchor_lambda_kd`
  - `--anchor_lambda_proto`
  - `--anchor_lambda_prm`
  - `--anchor_log_interval`
- Added `--anchor_train_file` as an optional per-client labeled full-modal source.

### 2. `dataset/datasets_nii.py`

- Added `full_mask_array = [True, True, True, True]`.
- Added `_load_case_names(...)` so Module 1 can read either `.txt` or `.csv` case lists.
- Added `Brats_loadall_labeled_full_nii`, which:
  - always returns full four-modal input,
  - returns one-hot target,
  - returns a full-modal mask.

### 3. `models/rfnet.py`

- Refactored encoder path into `_encode_modalities(...)`.
- Added `forward_fused_features(...)` for future clean access to fused outputs/features.
- Extended `forward(...)` with `return_features=False`.
- When `return_features=True`, training code can get the fused feature map needed for anchor extraction.
- During inference (`self.is_training == False` or `target is None`), `forward(...)` now returns only fused prediction probability, which matches the current evaluation path.

### 4. `utils/fedmass_anchor.py`

- New Module 1 utility file.
- Added:
  - `build_full_modal_mask(...)`
  - `extract_class_anchors(...)`
  - `init_local_anchor_state(...)`
  - `update_local_anchor_state(...)`
  - `finalize_local_anchor_state(...)`
  - `compute_anchor_supervision(...)`
- `compute_anchor_supervision(...)` currently computes:
  - full-modal segmentation loss,
  - optional unimodal supervised loss,
  - KD loss,
  - prototype loss,
  - PRM loss,
  - batch-level class anchors from fused features.

### 5. `train.py`

- Imported the new full-modal labeled dataset and Module 1 utility file.
- Extended `local_training(...)` with optional `anchor_dataloader`.
- Added per-iteration Module 1 branch:
  - samples labeled full-modal batch,
  - computes Module 1 loss,
  - adds it to the original FedAMM loss,
  - accumulates local full-modal anchors.
- Added Module 1 loss scalars into TensorBoard logging.
- Built one extra labeled loader per client when `--enable_anchor_supervision` is enabled.
- Added fallback behavior:
  - if `anchor_train_file[client_id]` is `None`,
  - Module 1 reuses that client's existing `train_file` list as the labeled full-modal case source.
- Local training now returns `local_full_anchors` in addition to the original outputs.

## Current Assumption

Because the codebase does not yet contain a dedicated labeled/unlabeled split for FedMASS, Module 1 uses this rule for now:

- if a dedicated `anchor_train_file` is provided, use it;
- otherwise, reuse the current client `train_file` case list and force full-modal supervision on those cases.

This keeps Module 1 runnable without blocking on data split generation, but it is still a temporary assumption.

## Checks Performed

### Syntax Compile Check

Passed for:

- `FedAMM/options.py`
- `FedAMM/dataset/datasets_nii.py`
- `FedAMM/models/rfnet.py`
- `FedAMM/utils/fedmass_anchor.py`
- `FedAMM/train.py`

### Runtime Import Check

Blocked by local environment dependency:

- `ModuleNotFoundError: No module named 'nibabel'`

This is an environment issue, not a syntax issue introduced by the patch.

## Not Changed Yet

- No EMA teacher
- No pseudo-label generation/filtering
- No reliability-aware aggregation
- No server-side anchor bank update
- No missing-pattern prototype bank

Those remain for Module 2 and Module 3.
