"""
Training Script for Joint MOMENT + Chronos-2 Model
===================================================
Implements training loop with LoRA fine-tuning for steel production forecasting.

Training Strategy:
- MOMENT: Unfreeze last 1-2 transformer blocks for domain adaptation
- Chronos-2: Apply LoRA to attention layers (Q, K, V, O projections)
- Bridge: Fully trainable (projector + future tokens)
- Optimizer: AdamW with weight decay
- Scheduler: Linear warmup + cosine decay
- Loss: Quantile loss (Pinball loss) for probabilistic forecasting
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType

from joint_model import JointMOMENTChronos2
from joint_config import JointConfig, get_default_config
from data_preprocessing import load_and_prepare_data


def apply_lora_to_chronos(model: JointMOMENTChronos2, config: JointConfig) -> JointMOMENTChronos2:
    """
    Apply LoRA adapters to Chronos-2 encoder.

    Args:
        model: Joint model instance
        config: Configuration with LoRA parameters

    Returns:
        Model with LoRA applied to Chronos-2
    """
    print(f"\n{'='*80}")
    print("Applying LoRA to Chronos-2")
    print(f"{'='*80}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,  # ['q', 'k', 'v', 'o']
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.FEATURE_EXTRACTION,  # We're doing feature extraction for forecasting
    )

    # Apply LoRA to Chronos-2 encoder
    model.chronos.encoder = get_peft_model(model.chronos.encoder, lora_config)

    print("LoRA configuration:")
    print(f"  Rank (r): {config.lora_r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.lora_target_modules}")
    print(f"  Dropout: {config.lora_dropout}")
    print("")

    # Print trainable parameters
    model.chronos.encoder.print_trainable_parameters()

    return model


def create_optimizer(model: JointMOMENTChronos2, config: JointConfig):
    """
    Create AdamW optimizer for trainable parameters.

    Args:
        model: Joint model
        config: Training configuration

    Returns:
        Optimizer instance
    """
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")

    return optimizer


def create_scheduler(optimizer, config: JointConfig, num_training_steps: int):
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer instance
        config: Training configuration
        num_training_steps: Total number of training steps

    Returns:
        Scheduler instance
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int):
        if current_step < config.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, config.warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - config.warmup_steps) / float(max(1, num_training_steps - config.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"\nScheduler: Warmup + Cosine Decay")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Total steps: {num_training_steps}")

    return scheduler


def train_epoch(
    model: JointMOMENTChronos2,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    config: JointConfig,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Joint model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        config: Training configuration
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

    for step, batch in enumerate(pbar):
        # Move batch to device
        batch = {
            k: v.to(config.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with autocast():
                outputs = model(batch)
                loss = outputs.loss
        else:
            outputs = model(batch)
            loss = outputs.loss

        # Backward pass
        if config.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm
            )

            # Optimizer step
            optimizer.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    avg_loss = total_loss / num_batches

    return {
        'train_loss': avg_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate(
    model: JointMOMENTChronos2,
    val_loader: DataLoader,
    config: JointConfig,
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: Joint model
        val_loader: Validation data loader
        config: Training configuration

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation")

    for batch in pbar:
        # Move batch to device
        batch = {
            k: v.to(config.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        outputs = model(batch)
        loss = outputs.loss

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}",
        })

    avg_loss = total_loss / num_batches

    return {
        'val_loss': avg_loss,
    }


def save_checkpoint(
    model: JointMOMENTChronos2,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str,
):
    """
    Save model checkpoint.

    Args:
        model: Joint model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: JointMOMENTChronos2,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
) -> int:
    """
    Load model checkpoint.

    Args:
        model: Joint model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint

    Returns:
        Starting epoch
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous metrics: {checkpoint['metrics']}")

    return start_epoch


def train(config: JointConfig):
    """
    Main training function.

    Args:
        config: Training configuration
    """
    print(f"\n{'='*80}")
    print("JOINT MOMENT + CHRONOS-2 TRAINING")
    print(f"{'='*80}\n")

    # Print configuration
    print(config.summary())

    # ========================================================================
    # Data Loading
    # ========================================================================
    print(f"\n{'='*80}")
    print("Loading Data")
    print(f"{'='*80}")

    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        metadata_path=config.metadata_path,
        json_dir=config.data_dir,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ========================================================================
    # Model Initialization
    # ========================================================================
    print(f"\n{'='*80}")
    print("Initializing Model")
    print(f"{'='*80}")

    model = JointMOMENTChronos2(config)

    # Apply LoRA to Chronos-2
    model = apply_lora_to_chronos(model, config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Move to device
    model = model.to(config.device)

    # ========================================================================
    # Optimizer & Scheduler
    # ========================================================================
    optimizer = create_optimizer(model, config)

    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)

    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision else None

    # ========================================================================
    # Resume from checkpoint if specified
    # ========================================================================
    start_epoch = 0
    if config.resume_from_checkpoint is not None:
        start_epoch = load_checkpoint(model, optimizer, scheduler, config.resume_from_checkpoint)

    # ========================================================================
    # Create output directory
    # ========================================================================
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Training Loop
    # ========================================================================
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, config)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  LR:         {train_metrics['learning_rate']:.2e}")

        # Save checkpoint every save_steps
        if (epoch + 1) % (config.save_steps // len(train_loader)) == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {**train_metrics, **val_metrics},
                str(checkpoint_path)
            )

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_checkpoint_path = output_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {**train_metrics, **val_metrics},
                str(best_checkpoint_path)
            )
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Joint MOMENT + Chronos-2 Model")

    # Config
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    # Data
    parser.add_argument("--data_dir", type=str, default="./json", help="Data directory")
    parser.add_argument("--metadata_path", type=str, default="./info2.xlsx", help="Metadata path")

    # Training
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        # TODO: Implement config file loading
        raise NotImplementedError("Config file loading not yet implemented")
    else:
        config = get_default_config()

    # Override with command-line arguments
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.metadata_path is not None:
        config.metadata_path = args.metadata_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.device is not None:
        config.device = args.device

    # Run training
    train(config)


if __name__ == "__main__":
    main()
