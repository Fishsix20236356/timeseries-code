"""
Configuration for MOMENT + Chronos-2 Joint Training Framework
===============================================================
Defines hyperparameters and settings for the joint training pipeline.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class JointConfig:
    """
    Comprehensive configuration for joint MOMENT + Chronos-2 training.

    This configuration handles both models, the feature bridge, LoRA fine-tuning,
    and training hyperparameters.
    """

    # ============================================================================
    # Data Configuration
    # ============================================================================

    data_dir: str = "./json"
    """Directory containing JSON time series files"""

    metadata_path: str = "./info2.xlsx"
    """Path to metadata Excel file"""

    context_length: int = 512
    """Historical window size (number of timesteps)"""

    prediction_length: int = 96
    """Forecast horizon (number of timesteps)"""

    train_ratio: float = 0.7
    """Training set ratio"""

    val_ratio: float = 0.15
    """Validation set ratio (test = 1 - train - val)"""

    # ============================================================================
    # MOMENT Configuration
    # ============================================================================

    moment_checkpoint: str = "AutonLab/MOMENT-1-large"
    """HuggingFace model checkpoint for MOMENT"""

    moment_d_model: int = 768
    """MOMENT hidden dimension (768 for MOMENT-1-large)"""

    moment_patch_len: int = 8
    """MOMENT patch length"""

    moment_patch_stride: int = 8
    """MOMENT patch stride"""

    moment_unfreeze_last_n: int = 2
    """Number of last transformer blocks to unfreeze in MOMENT"""

    moment_enable_gradient_checkpointing: bool = True
    """Enable gradient checkpointing for MOMENT to save memory"""

    # ============================================================================
    # Chronos-2 Configuration
    # ============================================================================

    chronos_checkpoint: str | None = None
    """Path to local Chronos-2 checkpoint (None to create from scratch)"""

    chronos_d_model: int = 512
    """Chronos-2 hidden dimension"""

    chronos_d_kv: int = 64
    """Chronos-2 key/value projection dimension"""

    chronos_d_ff: int = 2048
    """Chronos-2 feed-forward dimension"""

    chronos_num_layers: int = 6
    """Number of encoder layers in Chronos-2"""

    chronos_num_heads: int = 8
    """Number of attention heads in Chronos-2"""

    chronos_dropout_rate: float = 0.1
    """Dropout rate for Chronos-2"""

    chronos_layer_norm_epsilon: float = 1e-6
    """Layer normalization epsilon for Chronos-2"""

    chronos_patch_size: int = 8
    """Chronos-2 input/output patch size (must match MOMENT)"""

    chronos_patch_stride: int = 8
    """Chronos-2 patch stride"""

    chronos_use_arcsinh: bool = True
    """Use arcsinh transformation for Chronos-2 targets"""

    chronos_use_reg_token: bool = True
    """Insert REG token separator between history and future"""

    chronos_quantiles: List[float] = field(default_factory=lambda: [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ])
    """Quantile levels for probabilistic forecasting (19 quantiles)"""

    # ============================================================================
    # Feature Bridge Configuration
    # ============================================================================

    num_future_patches: int = 12
    """Number of future patches (prediction_length / patch_size)"""

    bridge_use_residual: bool = False
    """Use residual connection in projector"""

    # ============================================================================
    # LoRA Configuration (for Chronos-2)
    # ============================================================================

    lora_r: int = 8
    """LoRA rank (low-rank dimension)"""

    lora_alpha: int = 32
    """LoRA alpha (scaling factor)"""

    lora_dropout: float = 0.1
    """LoRA dropout rate"""

    lora_target_modules: List[str] = field(default_factory=lambda: ['q', 'k', 'v', 'o'])
    """LoRA target modules (attention projections)"""

    lora_bias: str = "none"
    """LoRA bias handling ('none', 'all', or 'lora_only')"""

    # ============================================================================
    # Training Configuration
    # ============================================================================

    batch_size: int = 8
    """Training batch size"""

    num_epochs: int = 50
    """Number of training epochs"""

    learning_rate: float = 1e-4
    """Learning rate for AdamW optimizer"""

    weight_decay: float = 0.01
    """Weight decay (L2 regularization)"""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping"""

    warmup_steps: int = 500
    """Number of warmup steps for learning rate scheduler"""

    eval_steps: int = 500
    """Evaluate every N steps"""

    save_steps: int = 1000
    """Save checkpoint every N steps"""

    logging_steps: int = 50
    """Log metrics every N steps"""

    # ============================================================================
    # Hardware Configuration
    # ============================================================================

    device: str = "cuda"
    """Device to use ('cuda' or 'cpu')"""

    mixed_precision: bool = True
    """Use mixed precision training (FP16)"""

    num_workers: int = 4
    """Number of dataloader workers"""

    pin_memory: bool = True
    """Pin memory for faster data transfer"""

    # ============================================================================
    # Checkpoint Configuration
    # ============================================================================

    output_dir: str = "./checkpoints"
    """Directory to save checkpoints"""

    resume_from_checkpoint: str | None = None
    """Path to checkpoint to resume from"""

    save_total_limit: int = 3
    """Maximum number of checkpoints to keep"""

    # ============================================================================
    # Computed Properties
    # ============================================================================

    def __post_init__(self):
        """Validate and compute derived properties."""
        # Ensure patch sizes match
        assert self.moment_patch_len == self.chronos_patch_size, \
            f"MOMENT patch_len ({self.moment_patch_len}) must match Chronos patch_size ({self.chronos_patch_size})"

        # Compute num_future_patches
        assert self.prediction_length % self.chronos_patch_size == 0, \
            f"prediction_length ({self.prediction_length}) must be divisible by patch_size ({self.chronos_patch_size})"
        self.num_future_patches = self.prediction_length // self.chronos_patch_size

        # Validate ratios
        assert 0 < self.train_ratio < 1, "train_ratio must be between 0 and 1"
        assert 0 < self.val_ratio < 1, "val_ratio must be between 0 and 1"
        assert self.train_ratio + self.val_ratio < 1, "train_ratio + val_ratio must be < 1"

    def get_chronos_core_config_dict(self) -> dict:
        """
        Get Chronos2CoreConfig parameters as dict.

        Returns:
            Dictionary with Chronos core config parameters
        """
        return {
            'd_model': self.chronos_d_model,
            'd_kv': self.chronos_d_kv,
            'd_ff': self.chronos_d_ff,
            'num_layers': self.chronos_num_layers,
            'num_heads': self.chronos_num_heads,
            'dropout_rate': self.chronos_dropout_rate,
            'layer_norm_epsilon': self.chronos_layer_norm_epsilon,
            'vocab_size': 2 if self.chronos_use_reg_token else 1,
            'pad_token_id': 0,
        }

    def get_chronos_forecasting_config_dict(self) -> dict:
        """
        Get Chronos2ForecastingConfig parameters as dict.

        Returns:
            Dictionary with Chronos forecasting config parameters
        """
        return {
            'context_length': self.context_length,
            'output_patch_size': self.chronos_patch_size,
            'input_patch_size': self.chronos_patch_size,
            'input_patch_stride': self.chronos_patch_stride,
            'quantiles': self.chronos_quantiles,
            'use_reg_token': self.chronos_use_reg_token,
            'use_arcsinh': self.chronos_use_arcsinh,
            'max_output_patches': self.num_future_patches,
        }

    def get_lora_config_dict(self) -> dict:
        """
        Get LoRA configuration as dict.

        Returns:
            Dictionary with LoRA config parameters
        """
        return {
            'r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'target_modules': self.lora_target_modules,
            'lora_dropout': self.lora_dropout,
            'bias': self.lora_bias,
        }

    def summary(self) -> str:
        """
        Generate a configuration summary string.

        Returns:
            Formatted configuration summary
        """
        summary = []
        summary.append("=" * 80)
        summary.append("JOINT TRAINING CONFIGURATION SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        summary.append("Data:")
        summary.append(f"  Context length: {self.context_length}")
        summary.append(f"  Prediction length: {self.prediction_length}")
        summary.append(f"  Batch size: {self.batch_size}")
        summary.append("")
        summary.append("MOMENT:")
        summary.append(f"  Checkpoint: {self.moment_checkpoint}")
        summary.append(f"  Hidden dim: {self.moment_d_model}")
        summary.append(f"  Unfreeze last {self.moment_unfreeze_last_n} blocks")
        summary.append("")
        summary.append("Chronos-2:")
        summary.append(f"  Hidden dim: {self.chronos_d_model}")
        summary.append(f"  Num layers: {self.chronos_num_layers}")
        summary.append(f"  Num quantiles: {len(self.chronos_quantiles)}")
        summary.append(f"  Use arcsinh: {self.chronos_use_arcsinh}")
        summary.append(f"  Use REG token: {self.chronos_use_reg_token}")
        summary.append("")
        summary.append("Bridge:")
        summary.append(f"  MOMENT dim → Chronos dim: {self.moment_d_model} → {self.chronos_d_model}")
        summary.append(f"  Future patches: {self.num_future_patches}")
        summary.append("")
        summary.append("LoRA:")
        summary.append(f"  Rank: {self.lora_r}")
        summary.append(f"  Alpha: {self.lora_alpha}")
        summary.append(f"  Target modules: {', '.join(self.lora_target_modules)}")
        summary.append("")
        summary.append("Training:")
        summary.append(f"  Epochs: {self.num_epochs}")
        summary.append(f"  Learning rate: {self.learning_rate}")
        summary.append(f"  Mixed precision: {self.mixed_precision}")
        summary.append(f"  Device: {self.device}")
        summary.append("=" * 80)

        return "\n".join(summary)


# Pre-defined configurations for different use cases

def get_default_config() -> JointConfig:
    """Get default configuration for steel production forecasting."""
    return JointConfig()


def get_small_config() -> JointConfig:
    """Get configuration for testing with smaller model/data."""
    return JointConfig(
        context_length=256,
        prediction_length=48,
        batch_size=4,
        chronos_num_layers=3,
        num_epochs=10,
        moment_checkpoint="AutonLab/MOMENT-1-small",
        moment_d_model=512,
    )


def get_large_config() -> JointConfig:
    """Get configuration for large-scale training."""
    return JointConfig(
        context_length=1024,
        prediction_length=192,
        batch_size=16,
        chronos_num_layers=12,
        chronos_d_model=768,
        chronos_d_ff=3072,
        num_epochs=100,
        lora_r=16,
    )


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")

    config = get_default_config()
    print(config.summary())

    print("\nChronos Core Config:")
    print(config.get_chronos_core_config_dict())

    print("\nChronos Forecasting Config:")
    print(config.get_chronos_forecasting_config_dict())

    print("\nLoRA Config:")
    print(config.get_lora_config_dict())

    print("\n✓ Configuration test passed!")
