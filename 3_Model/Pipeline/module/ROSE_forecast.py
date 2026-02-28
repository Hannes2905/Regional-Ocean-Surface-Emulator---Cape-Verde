"""
PyTorch Lightning module for the forecast model.

Handles:
- Single-step forward pass through embedder + UNet backbone
- Masked MSE / RMSE loss computation (ocean pixels only)
- Per-variable diagnostic logging (one metric per physical variable per epoch)
- Training, validation, and test step definitions for Lightning
- AdamW optimizer with cosine-warmup learning rate schedule

Batch dict contract (produced by CMEMSDataset):
    state:          [B, n_vars, H, W]   input at time t
    next_state:     [B, n_vars, H, W]   target at time t + lead_time
    prev_state:     [B, n_vars, H, W]   (optional) state at time t - lead_time
    mask:           [B, 1, H, W]        ocean=1, land=0
    timestamp:      [B]                 seconds since epoch (int64)
    lead_time_hours:[B]                 lead time in hours
"""

import lightning as L
import torch
import torch.nn as nn
import torch.utils.checkpoint as gradient_checkpoint
import torch.nn.functional as F
from hydra.utils import instantiate
from diffusers.optimization import get_cosine_schedule_with_warmup


class ROSEForecastModule(L.LightningModule):
    """
    Lightning module wrapping the embedder + UNet backbone for ocean surface forecasting.

    The module supports both single-step.

    Args:
        cfg: Hydra/OmegaConf config node containing sub-configs for:
             - cfg.embedder: embedder config (instantiated via hydra)
             - cfg.UNET: UNet backbone config (instantiated via hydra)
             - cfg.lr: learning rate (default 1e-4)
             - cfg.weight_decay: AdamW weight decay (default 1e-5)
             - cfg.num_warmup_steps: cosine-warmup steps (default 1000)
             - cfg.num_training_steps: total training steps (default 100000)
             - cfg.use_grad_checkpoint: enable gradient checkpointing (default False)
             - cfg.variable_names: list of variable names for per-var logging
        name: Identifier string stored on the module (default "forecast")
        residual_prediction: If True, the UNet predicts the state increment
                             (delta) and the current state is added back to
                             obtain the forecast (default True)
    """

    def __init__(
        self,
        cfg,
        name: str = "forecast",
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.residual_prediction = residual_prediction

        # Instantiate embedder (encodes state + prev_state into backbone input)
        self.embedder = instantiate(cfg.embedder)

        # Instantiate UNet backbone via Hydra; channel counts are derived
        # automatically from the embedder
        self.UNET = instantiate(
            cfg.UNET,
            n_channels=self.embedder.n_input_channels,
            n_classes=self.embedder.n_output_channels,
        )

        # Gradient checkpointing
        self.use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)

        # Persist all scalar hyperparameters; cfg itself is excluded because
        # OmegaConf DictConfig objects are not directly picklable
        self.save_hyperparameters(ignore=["cfg"])


    def forward(self, batch, *args, **kwargs):
        """
        Single-step forward pass: encode → UNet → decode → optional residual add.

        Args:
            batch: Dict containing at minimum 'state' [B, n_vars, H, W].
                   Optional keys: 'prev_state' [B, n_vars, H, W], 'mask' [B, 1, H, W].

        Returns:
            out: Predicted state at t + lead_time [B, n_vars, H, W]
        """
        # Encode: concatenate current state and (optional) previous state along
        # the channel dimension → [B, C_in, H, W]
        x = self.embedder.encode(
            batch['state'],
            batch.get('prev_state', None),
        )

        # Retrieve the land/ocean mask used by the partial convolution layers
        mask = batch.get('mask', None)
        if mask is None:
            # Fall back to an all-ones mask when none is provided (no masking)
            mask = torch.ones(
                x.shape[0], 1, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )

        # Partial convolution requires mask channels == input channels:
        # broadcast [B, 1, H, W] → [B, C_in, H, W]
        if mask.shape[1] != x.shape[1]:
            mask = mask.expand(-1, x.shape[1], -1, -1)

        # UNet backbone forward pass; returns raw prediction [B, n_vars, H, W]
        raw = self.UNET(x, mask)
        out = self.embedder.decode(raw)

        # Residual formulation: the UNet predicts the state increment
        if self.residual_prediction:
            out = out + batch['state']

        return out


    def compute_loss(self, pred, gt, mask=None):
        """
        Compute masked MSE loss and RMSE diagnostic.

        When a mask is supplied, the loss is averaged only over ocean pixels
        (mask == 1), ignoring land cells that carry no physical signal.

        Args:
            pred: Predicted state [B, n_vars, H, W]
            gt:   Ground-truth state [B, n_vars, H, W]
            mask: Optional ocean mask [B, 1, H, W], ocean=1, land=0.
                  If None, all pixels contribute equally.

        Returns:
            loss:    MSE scalar used for backpropagation
            metrics: Dict with detached 'mse' and 'rmse' scalars for logging
        """
        if mask is not None:
            # Broadcast mask from [B, 1, H, W] to [B, n_vars, H, W] so every
            # variable channel is masked consistently
            mask_expanded = mask.expand_as(pred)
            diff_sq = (pred - gt) ** 2
            # Clamp denominator to avoid division by zero on all-land batches
            mse = (diff_sq * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
        else:
            mse = F.mse_loss(pred, gt, reduction='mean')

        rmse = torch.sqrt(mse)

        return mse, {"mse": mse.detach(), "rmse": rmse.detach()}


    def training_step(self, batch):
        """
        Lightning training step: forward pass, loss, and metric logging.

        Args:
            batch:     Batch dict from CMEMSDataset

        Returns:
            loss: MSE scalar passed to the optimizer
        """
        pred = self.forward(batch)
        loss, metrics = self.compute_loss(
            pred, batch['next_state'], mask=batch.get('mask'),
        )

        # Log aggregate metrics; prog_bar=True shows MSE in the tqdm progress bar
        self.log('train/mse', metrics['mse'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/rmse', metrics['rmse'], on_step=True, on_epoch=True)
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True)

        return loss


    def validation_step(self, batch):
        """
        Lightning validation step: forward pass, loss, and metric logging.

        Args:
            batch:     Batch dict from CMEMSDataset

        Returns:
            loss: MSE scalar
        """
        pred = self.forward(batch)
        loss, metrics = self.compute_loss(
            pred, batch['next_state'], mask=batch.get('mask'),
        )

        self.log('val/mse', metrics['mse'], on_step=False, on_epoch=True)
        self.log('val/rmse', metrics['rmse'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True)
        self.log('val_rmse', metrics['rmse'], on_step=False, on_epoch=True)


        return loss


    def test_step(self, batch):
        """
        Lightning test step: forward pass, loss, and metric logging.

        Args:
            batch:     Batch dict from CMEMSDataset

        Returns:
            loss: MSE scalar
        """
        pred = self.forward(batch)
        loss, metrics = self.compute_loss(
            pred, batch['next_state'], mask=batch.get('mask'),
        )

        self.log('test/mse', metrics['mse'], on_step=False, on_epoch=True)
        self.log('test/rmse', metrics['rmse'], on_step=False, on_epoch=True)
        self.log('test/loss', loss.detach(), on_step=False, on_epoch=True)


        return loss


    def configure_optimizers(self):
        """
        Set up AdamW optimizer with a cosine decay + linear warmup schedule.

        The scheduler is stepped every training step ('interval': 'step') to
        produce a smooth learning-rate curve.  All hyperparameters are read
        from the Hydra config with sensible defaults so experiments can be
        fully controlled from the YAML config files.

        Returns:
            Dict consumed by Lightning containing 'optimizer' and
            'lr_scheduler' with interval and frequency settings.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.get('lr', 1e-4),
            weight_decay=self.cfg.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),  
        )

        # Cosine schedule with a short linear warmup; num_cycles=1 means the
        # LR reaches its minimum exactly at num_training_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.get('num_warmup_steps', 1000),
            num_training_steps=self.cfg.get('num_training_steps', 100000),
            num_cycles=1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # update every batch, not every epoch
                'frequency': 1,
            },
        }
