"""
Joint MOMENT + Chronos-2 Model for Time Series Forecasting
===========================================================
Integrates MOMENT (feature extractor) with Chronos-2 (forecaster) for
steel production forecasting with fault signature preservation.

Five-Stage Architecture:
1. Dual-track preprocessing (RevIN for MOMENT, arcsinh for Chronos-2)
2. History encoding via MOMENT (freeze early layers, unfreeze last blocks)
3. Feature projection & temporal reconstruction (bridge + future tokens)
4. Forecasting via Chronos-2 (LoRA fine-tuning on attention layers)
5. Loss calculation (Quantile/Pinball loss)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

from model import Chronos2Model, Chronos2Output
from config import Chronos2CoreConfig, Chronos2ForecastingConfig
from bridge import FeatureBridge, create_bridge
from joint_config import JointConfig


def freeze_parameters(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in a model.

    Args:
        model: PyTorch module to freeze

    Returns:
        The same model with frozen parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


class JointMOMENTChronos2(nn.Module):
    """
    Joint training model combining MOMENT and Chronos-2.

    This model:
    1. Extracts features from historical data using MOMENT
    2. Projects MOMENT features to Chronos-2 dimension via bridge
    3. Adds learnable future tokens as prediction placeholders
    4. Forecasts using Chronos-2 with LoRA fine-tuning
    """

    def __init__(self, config: JointConfig):
        """
        Initialize joint model.

        Args:
            config: Joint configuration with all hyperparameters
        """
        super().__init__()

        self.config = config

        # ====================================================================
        # Stage 1: Load MOMENT from HuggingFace
        # ====================================================================
        print(f"\n{'='*80}")
        print("Loading MOMENT from HuggingFace")
        print(f"{'='*80}")

        self.moment = MOMENTPipeline.from_pretrained(
            config.moment_checkpoint,
            model_kwargs={'task_name': 'embedding'},
        )
        self.moment.init()

        print(f"MOMENT loaded: {config.moment_checkpoint}")
        print(f"  Hidden dim: {config.moment_d_model}")
        print(f"  Patch length: {config.moment_patch_len}")

        # Apply freezing strategy
        self._freeze_moment_layers(config.moment_unfreeze_last_n)

        # Enable gradient checkpointing if requested
        if config.moment_enable_gradient_checkpointing:
            if hasattr(self.moment, 'encoder') and hasattr(self.moment.encoder, 'gradient_checkpointing_enable'):
                self.moment.encoder.gradient_checkpointing_enable()
                print("  Gradient checkpointing enabled")

        # ====================================================================
        # Stage 2: Initialize Chronos-2
        # ====================================================================
        print(f"\n{'='*80}")
        print("Initializing Chronos-2")
        print(f"{'='*80}")

        # Create Chronos-2 config from JointConfig
        # Chronos2Model expects the config object to have a 'chronos_config' attribute
        chronos_core_dict = config.get_chronos_core_config_dict()
        chronos_core_dict['chronos_config'] = config.get_chronos_forecasting_config_dict()
        chronos_core_config = Chronos2CoreConfig(**chronos_core_dict)

        # Load or create Chronos-2 model
        if config.chronos_checkpoint is not None:
            print(f"Loading Chronos-2 from checkpoint: {config.chronos_checkpoint}")
            self.chronos = Chronos2Model.from_pretrained(config.chronos_checkpoint)
        else:
            print("Creating Chronos-2 from scratch")
            self.chronos = Chronos2Model(config=chronos_core_config)

        print(f"Chronos-2 initialized:")
        print(f"  Hidden dim: {config.chronos_d_model}")
        print(f"  Num layers: {config.chronos_num_layers}")
        print(f"  Num heads: {config.chronos_num_heads}")
        print(f"  Num quantiles: {len(config.chronos_quantiles)}")
        print(f"  Use arcsinh: {config.chronos_use_arcsinh}")
        print(f"  Use REG token: {config.chronos_use_reg_token}")

        # ====================================================================
        # Stage 3: Create Feature Bridge
        # ====================================================================
        print(f"\n{'='*80}")
        print("Creating Feature Bridge")
        print(f"{'='*80}")

        bridge_type = "residual" if config.bridge_use_residual else "simple"
        self.bridge = create_bridge(
            moment_dim=config.moment_d_model,
            chronos_dim=config.chronos_d_model,
            num_future_patches=config.num_future_patches,
            bridge_type=bridge_type,
            dropout=0.1,
        )

        print(f"Bridge created: {bridge_type}")
        print(f"  MOMENT dim → Chronos dim: {config.moment_d_model} → {config.chronos_d_model}")
        print(f"  Future patches: {config.num_future_patches}")

        # ====================================================================
        # Store dimensions for reference
        # ====================================================================
        self.moment_dim = config.moment_d_model
        self.chronos_dim = config.chronos_d_model
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        self.num_future_patches = config.num_future_patches

        print(f"\n{'='*80}")
        print("Joint Model Initialization Complete")
        print(f"{'='*80}\n")

    def _freeze_moment_layers(self, unfreeze_last_n: int):
        """
        Freeze MOMENT layers except the last N transformer blocks.

        Args:
            unfreeze_last_n: Number of last transformer blocks to keep trainable
        """
        print(f"\nApplying MOMENT freezing strategy:")

        # Freeze patch embedding
        if hasattr(self.moment, 'patch_embedding'):
            self.moment.patch_embedding = freeze_parameters(self.moment.patch_embedding)
            print(f"  ✓ Patch embedding frozen")

        # Freeze encoder except last N blocks
        if hasattr(self.moment, 'encoder') and hasattr(self.moment.encoder, 'block'):
            num_blocks = len(self.moment.encoder.block)

            # Freeze early blocks
            for i in range(num_blocks - unfreeze_last_n):
                self.moment.encoder.block[i] = freeze_parameters(self.moment.encoder.block[i])

            print(f"  ✓ Encoder: frozen {num_blocks - unfreeze_last_n}/{num_blocks} blocks")
            print(f"  ✓ Trainable: last {unfreeze_last_n} blocks")

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.moment.parameters())
        trainable_params = sum(p.numel() for p in self.moment.parameters() if p.requires_grad)

        print(f"  MOMENT parameters: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.2f}%)")

    def _compute_loc_scale_from_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute instance normalization statistics from context.

        This replicates Chronos-2's instance normalization logic to provide
        loc_scale when using pre-computed embeddings.

        Args:
            context: Historical data (batch_size, context_length)
            context_mask: Mask for valid values (batch_size, context_length)

        Returns:
            loc: Location parameter (batch_size, 1)
            scale: Scale parameter (batch_size, 1)
        """
        if context_mask is None:
            context_mask = ~torch.isnan(context)

        # Compute statistics over valid values
        context_masked = torch.where(context_mask, context, torch.zeros_like(context))

        # Location: median (approximated by mean for simplicity)
        loc = context_masked.sum(dim=-1, keepdim=True) / context_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # Scale: MAD (median absolute deviation), approximated by std
        context_centered = context_masked - loc
        scale = torch.sqrt(
            (context_centered ** 2).sum(dim=-1, keepdim=True) / context_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        )
        scale = scale.clamp(min=1e-8)  # Avoid division by zero

        return loc, scale

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        output_attentions: bool = False,
    ) -> Chronos2Output:
        """
        Five-stage forward pass.

        Args:
            batch: Dictionary containing:
                - target: (batch_size, n_targets, context_length) historical targets
                - past_covariates: dict of (batch_size, context_length) covariates (optional)
                - future_target: (batch_size, n_targets, prediction_length) future targets (for training)
                - future_covariates: empty dict (not used in current setup)
            output_attentions: Whether to return attention weights

        Returns:
            Chronos2Output with loss and quantile predictions
        """
        # ====================================================================
        # Stage 1: Dual-track preprocessing (already done in dataset)
        # ====================================================================
        # Extract from batch
        target = batch['target']  # (batch_size, n_targets, context_length)
        future_target = batch.get('future_target')  # (batch_size, n_targets, prediction_length)

        batch_size, n_targets, context_len = target.shape

        # For now, we'll use the first target variable for MOMENT encoding
        # TODO: Support multivariate input by averaging or concatenating
        context = target[:, 0, :]  # (batch_size, context_length)

        # ====================================================================
        # Stage 2: MOMENT feature extraction
        # ====================================================================
        # MOMENT expects (batch, n_channels, seq_len)
        moment_input = context.unsqueeze(1)  # (batch_size, 1, context_length)

        # Forward through MOMENT in eval mode for frozen layers
        with torch.set_grad_enabled(self.training):
            moment_output = self.moment.embed(
                x_enc=moment_input,
                reduction='none'  # Keep all patch embeddings
            )

        # Extract patch embeddings
        # Shape: (batch_size, num_patches, moment_dim=768)
        # MOMENT output is 4D: (batch_size, num_patches, num_channels, hidden_dim)
        # We need to squeeze the channel dimension
        moment_features = moment_output.embeddings

        # Debug: Print shape to understand MOMENT output structure
        print(f"[DEBUG] moment_output.embeddings shape: {moment_features.shape}")
        print(f"[DEBUG] moment_output.embeddings ndim: {moment_features.ndim}")

        if moment_features.ndim == 4:
            # Squeeze channel dimension (should be 1)
            moment_features = moment_features.squeeze(2)
            print(f"[DEBUG] After squeezing channel dim: {moment_features.shape}")
        elif moment_features.ndim == 3:
            # Already 3D, no need to squeeze
            pass
        else:
            raise ValueError(
                f"Unexpected MOMENT output dimensions: {moment_features.shape}. "
                f"Expected 3D or 4D tensor."
            )

        # ====================================================================
        # Stage 3: Feature bridge projection
        # ====================================================================
        # Project MOMENT features to Chronos dimension
        projected_history, future_tokens = self.bridge(moment_features)
        # projected_history: (batch_size, num_history_patches, chronos_dim=512)
        # future_tokens: (batch_size, num_future_patches, chronos_dim=512)

        # Add REG token if needed (handled by Chronos-2 internally when use_reg_token=True)
        # Chronos-2 expects: [history patches] [REG] [future patches]
        # We provide: [projected history] [future tokens]
        # Chronos-2 will insert REG token between them

        # Combine history + future for Chronos-2 input
        # Note: REG token insertion is handled inside Chronos-2's encode() method
        combined_embeds = torch.cat([
            projected_history,  # (B, H, 512)
            future_tokens       # (B, F, 512)
        ], dim=1)  # (B, H+F, 512)

        # ====================================================================
        # Stage 4 & 5: Chronos-2 forecasting + loss computation
        # ====================================================================
        # Compute loc_scale for instance normalization (needed for loss)
        loc_scale = self._compute_loc_scale_from_context(context)

        # Prepare future target for loss computation
        if future_target is not None:
            # Use first target variable (same as context)
            future_target_single = future_target[:, 0, :]  # (batch_size, prediction_length)
        else:
            future_target_single = None

        # Call modified Chronos-2 with pre-computed embeddings
        outputs = self.chronos(
            context=None,  # Not used when inputs_embeds is provided
            inputs_embeds=combined_embeds,  # Pre-computed embeddings
            loc_scale=loc_scale,  # For instance normalization in loss
            num_output_patches=self.num_future_patches,
            future_target=future_target_single,
            future_target_mask=None,  # Auto-constructed from NaN
            group_ids=None,  # Each series treated independently
            future_covariates=None,  # Not used in joint model
            future_covariates_mask=None,
            output_attentions=output_attentions,
        )

        return outputs

    def predict(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate predictions for given context.

        Args:
            context: Historical data (batch_size, context_length)
            num_samples: Number of samples to generate (not used, kept for compatibility)

        Returns:
            Quantile predictions (batch_size, num_quantiles, prediction_length)
        """
        self.eval()

        with torch.no_grad():
            # Create batch dict
            batch = {
                'target': context.unsqueeze(1),  # (B, 1, context_length)
                'past_covariates': {},
                'future_target': None,
                'future_covariates': {},
            }

            outputs = self.forward(batch, output_attentions=False)

        return outputs.quantile_preds

    def get_trainable_parameters(self):
        """
        Get all trainable parameters (for optimizer).

        Returns:
            Generator of trainable parameters
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def print_trainable_parameters(self):
        """Print summary of trainable parameters by component."""
        print(f"\n{'='*80}")
        print("Trainable Parameters Summary")
        print(f"{'='*80}")

        # MOMENT
        moment_total = sum(p.numel() for p in self.moment.parameters())
        moment_trainable = sum(p.numel() for p in self.moment.parameters() if p.requires_grad)
        print(f"MOMENT: {moment_trainable:,} / {moment_total:,} ({100*moment_trainable/moment_total:.2f}%)")

        # Bridge
        bridge_trainable = sum(p.numel() for p in self.bridge.parameters() if p.requires_grad)
        print(f"Bridge: {bridge_trainable:,} (100%)")

        # Chronos-2 (will show LoRA percentage after LoRA is applied)
        chronos_total = sum(p.numel() for p in self.chronos.parameters())
        chronos_trainable = sum(p.numel() for p in self.chronos.parameters() if p.requires_grad)
        print(f"Chronos-2: {chronos_trainable:,} / {chronos_total:,} ({100*chronos_trainable/chronos_total:.2f}%)")

        # Total
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{'='*80}")
        print(f"Total: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test joint model initialization
    print("Testing Joint MOMENT + Chronos-2 Model...")

    from joint_config import get_default_config

    # Create config
    config = get_default_config()

    # Print config summary
    print(config.summary())

    # Initialize model (will download MOMENT if not cached)
    model = JointMOMENTChronos2(config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Test forward pass
    print(f"{'='*80}")
    print("Testing forward pass...")
    print(f"{'='*80}")

    batch_size = 2
    n_targets = 1

    # Create dummy batch
    batch = {
        'target': torch.randn(batch_size, n_targets, config.context_length),
        'past_covariates': {},
        'future_target': torch.randn(batch_size, n_targets, config.prediction_length),
        'future_covariates': {},
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    print(f"Input shape: {batch['target'].shape}")
    print(f"Output quantile_preds shape: {outputs.quantile_preds.shape}")
    print(f"Expected shape: ({batch_size}, {len(config.chronos_quantiles)}, {config.prediction_length})")

    assert outputs.quantile_preds.shape == (batch_size, len(config.chronos_quantiles), config.prediction_length)

    print("\n✓ Joint model test passed!")
