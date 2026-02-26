# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ctrl-World is a controllable generative world model for robot manipulation. It extends Stable Video Diffusion (SVD) with action conditioning to enable policy-in-the-loop rollouts entirely in imagination. Built on PyTorch and diffusers. Paper: arXiv 2510.10125.

## Commands

### Environment Setup
```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt

# Optional: for policy-in-the-loop with pi0.5 (requires openpi)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Data Preparation
```bash
# Step 1: Extract VAE latents from raw DROID video (parallelized)
accelerate launch dataset_example/extract_latent.py \
  --droid_hf_path ${path_to_droid} --droid_output_path dataset_example/droid \
  --svd_path ${svd_path}

# Step 2: Generate dataset meta info (normalization stats + train/val splits)
python dataset_meta_info/create_meta_info.py \
  --droid_output_path ${path_to_processed_droid} --dataset_name droid
```

### Training
```bash
# Test run with small subset (offline W&B)
WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset

# Full training on DROID
accelerate launch --main_process_port 29501 scripts/train_wm.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid
```

### Inference
```bash
# Replay recorded trajectory
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ${svd_path} \
  --clip_model_path ${clip_path} --ckpt_path ${ctrl_world_ckpt}

# Interactive keyboard control (commands: l/r/f/b/u/d/o/c)
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py \
  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset --svd_model_path ${svd_path} \
  --clip_model_path ${clip_path} --ckpt_path ${ctrl_world_ckpt} \
  --task_type keyboard --keyboard lllrrr

# Policy-in-the-loop with pi0.5 (requires openpi + JAX)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
  --task_type pickplace --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info --svd_model_path ${svd_path} \
  --clip_model_path ${clip_path} --ckpt_path ${ctrl_world_ckpt} --pi_ckpt ${pi_ckpt}
```

Rollout outputs are saved to `synthetic_traj/`.

### No Tests or Linting

This project has no test suite, CI/CD, or linting configuration. Validation happens during training (every 2500 steps). Manual testing is done via the rollout scripts with the provided `dataset_example/droid_subset` data.

## Architecture

### Data Flow
Raw video → `extract_latent.py` (VAE encoding) → stored latents → `Dataset_mix` loads latents at training time. Training operates entirely in latent space (shape: `B, T, 4, 32, 32`) to avoid repeated VAE encoding. Multi-camera views (3 cameras) are concatenated at the channel dimension.

### Core Model (`models/`)
- **ctrl_world.py**: `CrtlWorld` class (note the typo in the class name) — main model combining pretrained SVD with action-conditioned UNet. Forward pass: encode action → noise latents → concat history at channel dim → UNet → diffusion loss on future frames only.
- **unet_spatio_temporal_condition.py**: `UNetSpatioTemporalConditionModel` — extended SVD UNet with frame-level cross-attention for action conditioning. 4 down blocks, mid block, 4 up blocks; channels (320, 640, 1280, 1280).
- **pipeline_ctrl_world.py**: `CtrlWorldDiffusionPipeline` — inference pipeline extending `StableVideoDiffusionPipeline`. Handles history frame concatenation, classifier-free guidance with linear scaling, and frame-level action conditioning.
- **action_adapter/train2.py**: Dynamics model converting joint velocities to cartesian poses for pi05 policy integration.
- **utils.py**: Forward kinematics for Franka Panda (7-DOF) and keyboard-to-action-chunk conversion.

### Frozen vs Trainable Components
- **Frozen**: VAE (`AutoencoderKLTemporalDecoder`), Image Encoder (CLIP ViT), Text Encoder (CLIP)
- **Trainable**: UNet (spatio-temporal with action cross-attention), Action Encoder (`Action_encoder2`)

### Training (`scripts/train_wm.py`)
Uses HuggingFace Accelerate for distributed training. fp16 mixed precision, gradient accumulation, W&B/SwanLab logging. Validation every 2500 steps, checkpoints every 20000 steps. Classifier-free guidance uses 5% action dropout during training.

### Rollout Scripts (`scripts/`)
All rollout scripts share a common `agent` class that loads the pretrained `CrtlWorld` model and normalizes actions using dataset percentile statistics from `stat.json`.
- `rollout_replay_traj.py`: Replays recorded DROID actions through world model
- `rollout_key_board.py`: Interactive keyboard control (l/r/f/b/u/d/o/c commands)
- `rollout_interact_pi.py`: Policy-in-the-loop with pi05 VLA model (JAX-based, needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`)

### Data (`dataset/`)
- `dataset_droid_exp33.py`: `Dataset_mix` class — loads pre-computed VAE latents, handles multi-dataset weighted sampling, normalizes state to [-1, 1] using dataset percentile statistics (1st/99th percentiles from `stat.json`).

## Key Configuration (`config.py`)
All parameters in the `wm_args` dataclass. Critical defaults:
- Action dim: 7, Num future frames: 5, Num history frames: 6
- Latent shape: (B, T, 4, 32, 32), Resolution: 320×192
- Task types: pickplace, towel_fold, wipe_table, tissue, close_laptop, stack, drawer, keyboard
- Per-task settings (gripper limits, interaction count, eval trajectories) are configured in `__post_init__()`. Add new tasks there.

## Required Checkpoints
| Name | Purpose | Size |
|------|---------|------|
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | CLIP text/image encoder | ~600MB |
| [stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | Pretrained SVD base model | ~8GB |
| [Ctrl-World](https://huggingface.co/yjguo/Ctrl-World) | Trained world model checkpoint | ~8GB |
