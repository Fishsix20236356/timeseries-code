"""
Evaluation Script for Joint MOMENT + Chronos-2 Model
=====================================================
Evaluates trained model on test set and generates predictions with metrics.

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Quantile Coverage (for probabilistic forecasting)
- CRPS (Continuous Ranked Probability Score)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from joint_model import JointMOMENTChronos2
from joint_config import JointConfig, get_default_config
from data_preprocessing import load_and_prepare_data


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
) -> Dict[str, float]:
    """
    Compute forecasting metrics.

    Args:
        predictions: Quantile predictions (batch_size, num_quantiles, pred_len)
        targets: Ground truth (batch_size, pred_len)
        quantiles: List of quantile levels

    Returns:
        Dictionary with metrics
    """
    # Extract median prediction (quantile 0.5)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_pred = predictions[:, median_idx, :]  # (batch_size, pred_len)

    # Flatten for metric computation
    median_pred_flat = median_pred.reshape(-1)
    targets_flat = targets.reshape(-1)

    # MAE
    mae = torch.mean(torch.abs(median_pred_flat - targets_flat)).item()

    # RMSE
    rmse = torch.sqrt(torch.mean((median_pred_flat - targets_flat) ** 2)).item()

    # MAPE (avoid division by zero)
    mape = torch.mean(
        torch.abs((targets_flat - median_pred_flat) / (torch.abs(targets_flat) + 1e-8))
    ).item() * 100

    # Quantile Coverage (percentage of targets within prediction intervals)
    coverage = {}
    for i, q in enumerate(quantiles):
        if q < 0.5:
            # Lower quantile
            upper_idx = quantiles.index(1.0 - q) if (1.0 - q) in quantiles else -1
            if upper_idx != -1:
                lower_bound = predictions[:, i, :]
                upper_bound = predictions[:, upper_idx, :]
                within_interval = ((targets >= lower_bound) & (targets <= upper_bound)).float()
                coverage[f"coverage_{int(q*100)}_{int((1-q)*100)}"] = within_interval.mean().item() * 100

    # CRPS (Continuous Ranked Probability Score)
    # Approximation using quantiles
    crps = compute_crps(predictions, targets, quantiles)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'CRPS': crps,
        **coverage,
    }

    return metrics


def compute_crps(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS measures the difference between predicted and observed cumulative distributions.

    Args:
        predictions: Quantile predictions (batch_size, num_quantiles, pred_len)
        targets: Ground truth (batch_size, pred_len)
        quantiles: List of quantile levels

    Returns:
        CRPS score
    """
    batch_size, num_quantiles, pred_len = predictions.shape

    # Expand targets for broadcasting
    targets_expanded = targets.unsqueeze(1).expand(-1, num_quantiles, -1)

    # Compute quantile loss for each quantile
    quantiles_tensor = torch.tensor(quantiles, device=predictions.device).view(1, -1, 1)

    errors = targets_expanded - predictions
    quantile_loss = torch.where(
        errors >= 0,
        quantiles_tensor * errors,
        (quantiles_tensor - 1) * errors
    )

    # Average over all dimensions
    crps = quantile_loss.mean().item()

    return crps


@torch.no_grad()
def evaluate(
    model: JointMOMENTChronos2,
    test_loader: DataLoader,
    config: JointConfig,
    save_predictions: bool = False,
    output_dir: str = "./results",
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Trained joint model
        test_loader: Test data loader
        config: Configuration
        save_predictions: Whether to save predictions to file
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"\n{'='*80}")
    print("Evaluating on Test Set")
    print(f"{'='*80}\n")

    pbar = tqdm(test_loader, desc="Evaluation")

    for batch in pbar:
        # Move batch to device
        batch = {
            k: v.to(config.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        outputs = model(batch)

        # Extract predictions and targets
        predictions = outputs.quantile_preds  # (batch_size, num_quantiles, pred_len)
        targets = batch['future_target'][:, 0, :]  # (batch_size, pred_len) - first target variable

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples, num_quantiles, pred_len)
    all_targets = torch.cat(all_targets, dim=0)  # (total_samples, pred_len)

    print(f"\nTotal samples: {all_predictions.shape[0]}")
    print(f"Prediction shape: {all_predictions.shape}")
    print(f"Target shape: {all_targets.shape}")

    # Compute metrics
    print(f"\n{'='*80}")
    print("Computing Metrics")
    print(f"{'='*80}\n")

    metrics = compute_metrics(all_predictions, all_targets, config.chronos_quantiles)

    # Print metrics
    print("Point Forecast Metrics (Median):")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"\nProbabilistic Forecast Metrics:")
    print(f"  CRPS: {metrics['CRPS']:.4f}")
    print(f"\nPrediction Interval Coverage:")
    for key, value in metrics.items():
        if key.startswith('coverage_'):
            print(f"  {key}: {value:.2f}%")

    # Save predictions if requested
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays
        np.save(output_path / "predictions.npy", all_predictions.numpy())
        np.save(output_path / "targets.npy", all_targets.numpy())

        # Save metrics as JSON
        import json
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Predictions saved to: {output_path}")

    return metrics


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float],
    num_samples: int = 5,
    output_dir: str = "./results",
):
    """
    Plot sample predictions with uncertainty bands.

    Args:
        predictions: Quantile predictions (num_samples, num_quantiles, pred_len)
        targets: Ground truth (num_samples, pred_len)
        quantiles: List of quantile levels
        num_samples: Number of samples to plot
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find median index
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    # Find 10th and 90th percentile indices
    q10_idx = quantiles.index(0.1) if 0.1 in quantiles else 0
    q90_idx = quantiles.index(0.9) if 0.9 in quantiles else -1

    # Plot samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        # Time steps
        time_steps = np.arange(predictions.shape[2])

        # Plot ground truth
        ax.plot(time_steps, targets[i], 'k-', label='Ground Truth', linewidth=2)

        # Plot median prediction
        ax.plot(time_steps, predictions[i, median_idx], 'r--', label='Median Prediction', linewidth=2)

        # Plot uncertainty band (10th to 90th percentile)
        ax.fill_between(
            time_steps,
            predictions[i, q10_idx],
            predictions[i, q90_idx],
            alpha=0.3,
            color='red',
            label='80% Prediction Interval'
        )

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Sample {i+1}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "predictions_plot.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {output_path / 'predictions_plot.png'}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Joint MOMENT + Chronos-2 Model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Data
    parser.add_argument("--data_dir", type=str, default="./json", help="Data directory")
    parser.add_argument("--metadata_path", type=str, default="./info2.xlsx", help="Metadata path")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    parser.add_argument("--plot_predictions", action="store_true", help="Plot sample predictions")
    parser.add_argument("--num_plot_samples", type=int, default=5, help="Number of samples to plot")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    # Load configuration (from checkpoint or default)
    config = get_default_config()
    config.device = args.device
    config.batch_size = args.batch_size

    print(f"\n{'='*80}")
    print("JOINT MOMENT + CHRONOS-2 EVALUATION")
    print(f"{'='*80}\n")

    # Load data
    print(f"{'='*80}")
    print("Loading Data")
    print(f"{'='*80}")

    _, _, test_dataset = load_and_prepare_data(
        metadata_path=args.metadata_path,
        json_dir=args.data_dir,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(f"\nTest batches: {len(test_loader)}")

    # Initialize model
    print(f"\n{'='*80}")
    print("Loading Model")
    print(f"{'='*80}")

    model = JointMOMENTChronos2(config)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    # Move to device
    model = model.to(args.device)

    # Evaluate
    metrics = evaluate(
        model,
        test_loader,
        config,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # Plot predictions if requested
    if args.plot_predictions:
        print(f"\n{'='*80}")
        print("Plotting Predictions")
        print(f"{'='*80}")

        # Load saved predictions
        predictions = np.load(Path(args.output_dir) / "predictions.npy")
        targets = np.load(Path(args.output_dir) / "targets.npy")

        # Plot first N samples
        plot_predictions(
            predictions[:args.num_plot_samples],
            targets[:args.num_plot_samples],
            config.chronos_quantiles,
            num_samples=args.num_plot_samples,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
