"""
Main script to run the complete pipeline for training the regional ocean surface emulator
as well as running inference with trained models.

Uses hydra and omegaconf to organise configurations from hierarchical yaml files containing
the model hyperparameters, metric logging setups, and cluster-related parameters.
"""

import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("Working dir:", os.getcwd())
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Resolve exp_dir ──
    # Always use Hydra's timestamped output dir (outputs/YYYY-MM-DD/HH-MM-SS/).
    # Logs and checkpoints are stored there alongside .hydra/.
    exp_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Experiment dir: {exp_dir}")

    # ── Seed ──
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # ── Check for existing checkpoint to resume ──
    ckpt_path = None
    ckpt_dir = exp_dir / "checkpoints"
    if cfg.resume and ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.iterdir(), key=os.path.getmtime)
        if hasattr(cfg, "ckpt_filename_match"):
            ckpts = [x for x in ckpts if str(cfg.ckpt_filename_match) in x.name]
        if ckpts:
            ckpt_path = str(ckpts[-1])
            print(f"Resuming from checkpoint: {ckpt_path}")

    # ── Instantiate datasets ──
    if cfg.mode == "train":
        print("Preparing train and validation datasets...")

        trainset = instantiate(cfg.dataloader.dataset)
        val_args = OmegaConf.to_container(
            getattr(cfg.dataloader, "validation_args", OmegaConf.create({})),
            resolve=True,
        )
        valset = instantiate(cfg.dataloader.dataset, **val_args)

        print(f"  Train samples: {len(trainset)}")
        print(f"  Val samples:   {len(valset)}")
        sample = trainset[0]
        print(f"  State shape:   {sample['state'].shape}")
        print(f"  Mask shape:    {sample['mask'].shape}")

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=False,
            pin_memory=True,
        )

    elif cfg.mode == "test":
        print("Preparing test dataset...")
        test_args = OmegaConf.to_container(
            getattr(cfg.dataloader, "test_args", OmegaConf.create({})),
            resolve=True,
        )
        testset = instantiate(cfg.dataloader.dataset, **test_args)
        print(f"  Test samples: {len(testset)}")

        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=False,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'train' or 'test'.")

    # ── Instantiate Lightning module ──
    print("Instantiating model...")
    pl_module = instantiate(cfg.module.module, cfg.module)

    # ── Loggers ──
    log_dir = exp_dir / "logs"
    csv_logger = CSVLogger(save_dir=str(log_dir), name="csv")
    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tensorboard")
    loggers = [csv_logger, tb_logger]
    print(f"  CSV logs:         {csv_logger.log_dir}")
    print(f"  TensorBoard logs: {tb_logger.log_dir}")

    # ── Callbacks ──
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(exp_dir / "checkpoints"),
            filename="best-epoch={epoch}-val_rmse={val_rmse:.4f}",
            monitor="val_rmse",
            mode="min",
            save_top_k=3,
            every_n_epochs=1,
            save_last=True,  # also keeps last.ckpt
        ),
    ]

    # ── Trainer ──
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        gradient_clip_val=1.0,
        max_steps=cfg.max_steps,
        enable_checkpointing=True,
        default_root_dir=str(exp_dir),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=loggers,
        callbacks=callbacks,
    )

    # ── Run ──
    if cfg.debug:
        breakpoint()

    if cfg.mode == "train":
        print("Starting training...")
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)
    elif cfg.mode == "test":
        print("Starting inference...")
        trainer.test(pl_module, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()