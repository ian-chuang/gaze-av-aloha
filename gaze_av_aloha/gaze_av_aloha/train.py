import logging
import time
from pprint import pformat
import torch
from torch.amp import GradScaler
from contextlib import nullcontext
from gaze_av_aloha.utils.dataset_utils import cycle, EpisodeAwareSampler
from gaze_av_aloha.utils.logging_utils import AverageMeter, MetricsTracker, format_big_number
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import random
import numpy as np
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset, AVAlohaDatasetMeta
import importlib
import gymnasium as gym
import wandb
import shutil
from gaze_av_aloha.eval import eval_policy
from gaze_av_aloha.visualize import visualize_policy, NoTerminationWrapper
from gaze_av_aloha.utils.dataclass_utils import save_dataclass, load_dataclass
from gaze_av_aloha.configs import Config
import gaze_av_aloha # import to ensure all configs are registered

def train(cfg: Config):
    # get hydra run directory
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Run directory: {run_dir}")

    # check if resuming from a checkpoint
    checkpoint_path = None
    if cfg.resume:
        if cfg.checkpoint_path:
            checkpoint_path = Path(cfg.checkpoint_path)
        else:
            logging.info("checkpoint_path not provided, using latest checkpoint in run directory")
            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists(): 
                raise ValueError(f"directory {checkpoint_dir} does not exist, please provide a valid checkpoint_path")
            checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not checkpoint_dirs: raise ValueError(f"No checkpoints found in {checkpoint_dir}, please provide a valid checkpoint_path")
            checkpoint_path = max(checkpoint_dirs, key=lambda d: int(d.name))
        logging.info(f"Using checkpoint path: {checkpoint_path}")
        cfg_path = checkpoint_path / "config.json"
        if not cfg_path.exists():
            raise ValueError(f"Checkpoint path {checkpoint_path} does not contain a config.json file")
        cfg = load_dataclass(cfg_path)
        cfg = OmegaConf.structured(cfg)  # Convert to DictConfig
        overrides = OmegaConf.from_dotlist(list(hydra.core.hydra_config.HydraConfig.get().overrides.task))
        cfg = OmegaConf.merge(cfg, overrides)
        logging.info(f"Loaded config from {cfg_path} with overrides: {pformat(overrides, indent=4)}")
    
    # Convert DictConfig to Config object
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, Config), "Config should be an instance of Config class"
    logging.info(pformat(cfg, indent=4))

    # set seed
    logging.info(f"Setting random seed to {cfg.seed}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # get device
    device = torch.device(cfg.device)
    logging.info(f"Device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # create policy
    logging.info(f"Creating policy of type {cfg.policy.type}")
    dataset_meta = AVAlohaDatasetMeta(repo_id=cfg.task.dataset_repo_id, root=cfg.task.dataset_root)
    stats = dataset_meta.stats
    stats.update(cfg.task.override_stats)
    if cfg.policy.type == "gaze_policy":
        from gaze_av_aloha.policies.gaze_policy.gaze_policy import GazePolicy
        policy = GazePolicy(policy_cfg=cfg.policy, task_cfg=cfg.task, stats=stats)
    else:
        raise ValueError(f"Unknown policy type: {cfg.policy.type}")
    if checkpoint_path:
        logging.info(f"Loading policy from {checkpoint_path / 'policy'}")
        policy = policy.from_pretrained(checkpoint_path / "policy")
    policy.to(device)

    # create dataset
    logging.info(f"Creating dataset with repo_id={cfg.task.dataset_repo_id} and root={cfg.task.dataset_root}")
    delta_timestamps = policy.get_delta_timestamps()
    logging.info(f"""
        Delta timestamps:
        {pformat(delta_timestamps, indent=4)}
    """)
    dataset = AVAlohaDataset(
        repo_id=cfg.task.dataset_repo_id, 
        root=cfg.task.dataset_root,
        episodes=cfg.task.dataset_episodes,
        delta_timestamps=delta_timestamps,
    )

    # create eval env
    eval_envs = None
    if cfg.train.eval_freq > 0 and len(cfg.task.envs) > 0:
        eval_envs = {}
        for env_name, env_cfg in cfg.task.envs.items():
            logging.info(f"""
                Creating {env_cfg.eval_n_envs} {env_name} environments
                id={env_cfg.env_package}/{env_cfg.env_name}
                kwargs={pformat(env_cfg.env_kwargs, indent=4)}
            """)
            importlib.import_module(env_cfg.env_package)
            eval_env = gym.vector.SyncVectorEnv(
                [
                    lambda: gym.make(
                        f"{env_cfg.env_package}/{env_cfg.env_name}",
                        **env_cfg.env_kwargs
                    ) 
                    for _ in range(env_cfg.eval_n_envs)
                ]
            )
            eval_envs[env_name] = eval_env

    # create vis env
    viz_envs = None
    if cfg.train.viz_freq > 0 and len(cfg.task.envs) > 0:
        viz_envs = {}
        for env_name, env_cfg in cfg.task.envs.items():
            logging.info(f"""
                Creating 1 visualization environment for {env_name}
                id={env_cfg.env_package}/{env_cfg.env_name}
                kwargs={pformat(env_cfg.env_kwargs, indent=4)}
            """)
            importlib.import_module(env_cfg.env_package)
            viz_env = gym.vector.SyncVectorEnv(
                [
                    lambda: NoTerminationWrapper(gym.make(
                        f"{env_cfg.env_package}/{env_cfg.env_name}",
                        **env_cfg.env_kwargs
                    ))
                ]
            )
            viz_envs[env_name] = viz_env

    # create optimizer and lr scheduler
    logging.info(f"Creating optimizer, scheduler,")
    optimizer = policy.get_optimizer()
    lr_scheduler = policy.get_scheduler(optimizer, cfg.train.steps)
    grad_scaler = GradScaler(device.type, enabled=cfg.train.use_amp)
    ema = policy.get_ema()
    step = 0  # number of policy updates (forward + backward + optim)

    if checkpoint_path:
        try:
            training_state = torch.load(checkpoint_path / "training_state.pt")
            optimizer.load_state_dict(training_state["optimizer"])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            if ema is not None:
                ema.load_state_dict(training_state["ema"])
            step = training_state["step"]
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e
        logging.info(f"Resuming training from step {step}")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(f"{cfg.train.steps=} ({format_big_number(cfg.train.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # start Wandb
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.job_name if cfg.wandb.job_name else None,
            id=cfg.wandb.run_id if cfg.wandb.run_id else None,
            config=cfg,
            resume="allow" if cfg.resume else None,
            dir=run_dir,
        )
        cfg.wandb.run_id = wandb.run.id

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.train.num_workers,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.train.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.train.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        start_time = time.perf_counter()
        policy.train()


        with torch.autocast(device_type=device.type) if cfg.train.use_amp else nullcontext():
            loss, output_dict = policy.forward(batch)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        grad_scaler.scale(loss).backward()
        # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            cfg.train.grad_clip_norm,
            error_if_nonfinite=False,
        )
        grad_scaler.step(optimizer)
        # Updates the scale for next iteration.
        grad_scaler.update()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if ema is not None:
            ema.step(policy.parameters())

        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = time.perf_counter() - start_time

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.train.log_freq > 0 and step % cfg.train.log_freq == 0
        is_saving_step = cfg.train.save_freq > 0 and (step % cfg.train.save_freq == 0 or step == cfg.train.steps)
        is_keeping_step = cfg.train.keep_freq > 0 and (step % cfg.train.keep_freq == 0 or step == cfg.train.steps)
        is_eval_step = cfg.train.eval_freq > 0 and step % cfg.train.eval_freq == 0
        is_viz_step = cfg.train.viz_freq > 0 and step % cfg.train.viz_freq == 0

        # log metrics
        if is_log_step:
            logging.info(train_tracker)
            if cfg.wandb.enable and step >= wandb.run.step:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                for k, v in wandb_log_dict.items():
                    if not isinstance(v, (int, float, str)): continue
                    wandb.log({f"train/{k}": v}, step=step)
            train_tracker.reset_averages()

        # save policy
        if cfg.train.save_checkpoint and (is_saving_step or is_keeping_step):
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = run_dir / "checkpoints" / str(step).zfill(10)
            policy.save_pretrained(checkpoint_dir / "policy")
            training_state = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                "step": step,
            }
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            save_dataclass(cfg, checkpoint_dir / "config.json")

            # remove previous checkpoints if needed
            prev_saving_step = step - cfg.train.save_freq
            if prev_saving_step % cfg.train.keep_freq != 0:
                prev_checkpoint_dir = run_dir / "checkpoints" / str(prev_saving_step).zfill(10)
                if prev_checkpoint_dir.exists():
                    logging.info(f"Removing previous checkpoint at {prev_checkpoint_dir}")
                    shutil.rmtree(prev_checkpoint_dir)

        # load ema weights to an eval policy
        if is_eval_step or is_viz_step:
            if ema is not None:
                ema.store(policy.parameters())
                ema.copy_to(policy.parameters())

            if eval_envs and is_eval_step:
                step_id = str(step).zfill(10)
                for env_name, env_cfg in cfg.task.envs.items():
                    eval_env = eval_envs[env_name]
                    for options_name, options in env_cfg.eval_options.items():
                        logging.info(f"Eval policy at step {step} on {env_name} with {options_name} options: {options_name}={pformat(options, indent=4)}")
                        with torch.no_grad():
                            eval_info = eval_policy(
                                eval_env,
                                policy,
                                env_cfg.eval_n_episodes,
                                videos_dir=run_dir / "eval" / env_name / options_name / f"videos_step_{step_id}",
                                max_episodes_rendered=4,
                                start_seed=cfg.seed,
                                options=options,
                            )

                        logging.info(eval_info["aggregated"])
                        if cfg.wandb.enable and step >= wandb.run.step:
                            for k, v in eval_info["aggregated"].items():
                                if not isinstance(v, (int, float, str)): continue
                                wandb.log({f"eval/{k}_{env_name}_{options_name}": v}, step=step)
                            wandb_video = wandb.Video(eval_info['video_paths'][0], fps=cfg.task.fps, format="mp4")
                            wandb.log({f"eval/video_{env_name}_{options_name}": wandb_video}, step=step)

            if viz_envs and is_viz_step:
                step_id = str(step).zfill(10)
                for env_name, env_cfg in cfg.task.envs.items():
                    viz_env = viz_envs[env_name]
                    for options_name, options in env_cfg.eval_options.items():
                        logging.info(
                            f"Visualize policy at step {step} on {env_name} with {options_name} options: {options_name}={pformat(options, indent=4)}"
                        )
                        with torch.no_grad():
                            video_paths = visualize_policy(
                                viz_env,
                                policy,
                                videos_dir=run_dir / "visualize" / env_name / options_name / f"videos_step_{step_id}",
                                seed=step,
                                steps=env_cfg.visualization_steps,
                                options=options,
                            )
                        if cfg.wandb.enable and step >= wandb.run.step:
                            for video_path in video_paths:
                                name = Path(video_path).stem
                                wandb_video = wandb.Video(video_path, fps=cfg.task.fps, format="mp4")
                                wandb.log({f"visualize/{name}_{env_name}_{options_name}": wandb_video}, step=step)

            if ema is not None:
                ema.restore(policy.parameters())

    if eval_env:
        eval_env.close()
    if viz_env:
        viz_env.close()

    if cfg.wandb.enable:
        wandb.finish()

    logging.info("End of training")