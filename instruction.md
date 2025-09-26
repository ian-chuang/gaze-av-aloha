# Real Robot Data Collection and Training Pipeline

This repository provides scripts and instructions for operating the real robot (2-arm and 3-arm configurations), collecting datasets, converting them to the AV-Aloha format, and training/evaluating gaze and foveated vision models.

---

## 1. Start the Robot and Collect Data

### 1.1 Three-Arm Robot

1. **Start the robot (Terminal 1):**
   ```bash
   cd real_robot_sh/set_real_robot_with3arms
   ./robot_real.sh
   ```

2. **Start data collection (Terminal 2):**
   ```bash
   cd real_robot_sh/set_real_robot_with3arms
   ./record_real.sh
   ```

   > In `record_real_episodes.py`, you can configure:
   - `repo_id`
   - `task`
   - `num_episodes`
   - `root`

---

### 1.2 Two-Arm Robot

1. **Start the robot (Terminal 1):**
   ```bash
   cd real_robot_sh/set_real_robot_with2arms
   ./robot_real_2arms.sh
   ```

2. **Start data collection (Terminal 2):**
   ```bash
   cd real_robot_sh/set_real_robot_with2arms
   ./record_real_2arms.sh
   ```

   > Similar to the 3-arm setup, edit `record_real_episodes.py` to set `repo_id`, `task`, `num_episodes`, and `root`.

---

## 2. Convert Dataset to AV-Aloha Format

Navigate to the script directory:
```bash
cd gaze-av-aloha/gym_av_aloha/scripts/
```

Convert dataset:
```bash
python convert_lerobot_to_avaloha.py --repo_id Jinyu220/vedio --start_episode 0 --end_episode 6
```

To skip failed episodes:
```bash
python convert_lerobot_to_avaloha_skip.py --repo_id Jinyu220/vedio --start_episode 0 --end_episode 6
```

---

## 3. Train a Gaze Model

```bash
cd gaze-av-aloha/gaze_av_aloha/scripts/
python train_gaze_model.py
```

Or specify arguments:
```bash
python train_gaze_model_argparse.py --task shoot --dataset Jinyu220/shoot
```

---

## 4. Train a Foveated Policy

### 4.1 Configure Task

1. Create a custom `.yaml` file for your task:
   ```bash
   cd gaze-av-aloha/gaze_av_aloha/configs/task
   ```
2. Modify `default.yaml` under:
   ```bash
   cd gaze-av-aloha/gaze_av_aloha/configs/
   ```

### 4.2 Train with Foveated ViT Policy
```bash
python gaze_av_aloha/scripts/train.py   policy=foveated_vit_policy   policy.use_gaze_as_action=false   policy.gaze_model_repo_id=Jinyu220/gaze_model_av_aloha_real_NEW1_hook_circlev2   policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_foveated_vit   policy.optimizer_lr_backbone=1e-5   wandb.enable=true   wandb.project=hang_augementation_hook_Ian_unet   wandb.entity=jinyuzou220-uc-davis   wandb.job_name=fov-unet-augementation_hook_Ian   device=cuda
```

Replace `policy.gaze_model_repo_id` with your own trained gaze model.

### 4.3 Train with Fine-Tuned ViT Policy
```bash
python gaze_av_aloha/scripts/train.py   policy=vit_policy   task=<task_name e.g. av_aloha_sim_thread_needle>   policy.vision_encoder_kwargs.repo_id=iantc104/mae_vitb_vit   policy.optimizer_lr_backbone=1e-5   wandb.enable=true   wandb.project=<project_name>   wandb.entity=<your_wandb_entity>   wandb.job_name=fine   device=cuda
```

---

## 5. Evaluation

### 5.1 Three-Arm Robot

1. **Start the robot (Terminal 1):**
   ```bash
   cd real_robot_sh/set_real_robot_with3arms
   ./robot_real.sh
   ```

2. **Run evaluation (Terminal 2):**
   ```bash
   cd real_robot_sh/set_real_robot_with3arms
   ./eval.sh
   ```

> Configure `record_real_episodes.py` to adjust `repo_id`, `task`, `num_episodes`, and `root`.

---

### 5.2 Two-Arm Robot

1. **Start the robot (Terminal 1):**
   ```bash
   cd real_robot_sh/set_real_robot_with2arms
   ./robot_real_2arms.sh
   ```

2. **Run evaluation (Terminal 2):**
   ```bash
   cd real_robot_sh/set_real_robot_with2arms
   ./eval_2arms.sh
   # or
   ./eval_2arms_zjy.sh
   ```


