"""
Chronos-2 Fine-tuning Script for Steel Production Time Series
==============================================================
This script performs LoRA fine-tuning on Chronos-2 model using steel production data
and compares zero-shot vs fine-tuned predictions.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline

# Set matplotlib style - using standard fonts for better compatibility
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================================
# Step 1: Data Loading & Selection
# ============================================================================

def load_metadata(metadata_path='./info2.xlsx'):
    """Load metadata and filter target variables and covariates."""
    print("=" * 80)
    print("Step 1: Loading Metadata")
    print("=" * 80)

    df_meta = pd.read_excel(metadata_path)
    print(f"Total variables in metadata: {len(df_meta)}")

    # Define target variables and covariates based on chnName
    target_keywords = ["铁水温度", "铁水Si", "炉渣二元碱度"]
    covariate_keywords = ["喷煤量", "冷水流量", "氧气流量", "风压力"]

    # Filter targets
    targets = df_meta[df_meta['chnName'].str.contains('|'.join(target_keywords), na=False)]
    print(f"\nTarget variables found: {len(targets)}")
    for _, row in targets.iterrows():
        print(f"  - {row['chnName']} (code: {row['code']})")

    # Filter covariates
    covariates = df_meta[df_meta['chnName'].str.contains('|'.join(covariate_keywords), na=False)]
    print(f"\nCovariate variables found: {len(covariates)}")
    for _, row in covariates.iterrows():
        print(f"  - {row['chnName']} (code: {row['code']})")

    return targets, covariates


def load_timeseries_data(codes, names, json_dir='./json'):
    """Load time series data from JSON files and merge into a DataFrame."""
    print("\n" + "=" * 80)
    print("Loading Time Series Data")
    print("=" * 80)

    all_data = {}

    for code, name in zip(codes, names):
        json_path = Path(json_dir) / f"{code}.json"

        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping...")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to DataFrame
        df_temp = pd.DataFrame(data)
        df_temp['time'] = pd.to_datetime(df_temp['time'])
        df_temp = df_temp.set_index('time').sort_index()

        # Use chnName as column name for better readability
        all_data[name] = df_temp['value']

        print(f"Loaded {name}: {len(df_temp)} records from {df_temp.index.min()} to {df_temp.index.max()}")

    # Merge all data into a single DataFrame
    df_combined = pd.DataFrame(all_data)
    print(f"\nCombined DataFrame shape: {df_combined.shape}")
    print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    print(f"Missing values per column:\n{df_combined.isnull().sum()}")

    return df_combined


# ============================================================================
# Step 2: Preprocessing
# ============================================================================

def preprocess_data(df):
    """
    Preprocess time series data:
    1. Linear interpolation for missing values
    2. Z-score outlier detection and replacement
    3. Rolling mean smoothing
    """
    print("\n" + "=" * 80)
    print("Step 2: Preprocessing Data")
    print("=" * 80)

    df_processed = df.copy()

    # 1. Handle missing values with linear interpolation
    print("\n1. Handling missing values with linear interpolation...")
    missing_before = df_processed.isnull().sum().sum()
    df_processed = df_processed.interpolate(method='linear', limit_direction='both')
    missing_after = df_processed.isnull().sum().sum()
    print(f"   Missing values: {missing_before} -> {missing_after}")

    # Fill any remaining NaNs with forward/backward fill
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')

    # 2. Outlier detection and replacement using Z-score
    print("\n2. Detecting and replacing outliers (Z-score threshold = 3)...")
    for col in df_processed.columns:
        z_scores = np.abs(stats.zscore(df_processed[col]))
        outliers = z_scores > 3
        n_outliers = outliers.sum()

        if n_outliers > 0:
            print(f"   {col}: {n_outliers} outliers detected")
            # Replace outliers with median
            median_val = df_processed[col].median()
            df_processed.loc[outliers, col] = median_val

    # 3. Smoothing with rolling mean (window=3)
    print("\n3. Applying rolling mean smoothing (window=3)...")
    df_smoothed = df_processed.rolling(window=3, center=True, min_periods=1).mean()

    print("\nPreprocessing completed!")
    print(f"Final shape: {df_smoothed.shape}")
    print(f"Remaining NaN values: {df_smoothed.isnull().sum().sum()}")

    return df_smoothed


# ============================================================================
# Step 2.5: Time Series Visualization Module (English Labels)
# ============================================================================

def visualize_preprocessing_comparison(df_original, df_processed, output_dir='./visualization'):
    """
    Visualize the comparison between original and preprocessed time series data.
    """
    print("\n" + "=" * 80)
    print("Step 2.5: Visualizing Preprocessing Results")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Individual comparison plots for each variable
    print("\n1. Creating individual comparison plots...")
    for col in df_original.columns:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Comparison: {col}', fontsize=16, fontweight='bold')

        # Plot 1: Original vs Processed overlay
        axes[0].plot(df_original.index, df_original[col],
                    label='Original', alpha=0.7, linewidth=1.5, color='steelblue')
        axes[0].plot(df_processed.index, df_processed[col],
                    label='Processed', alpha=0.8, linewidth=1.5, color='orangered')
        axes[0].set_title('Original vs Processed Data', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Difference (residuals)
        difference = df_processed[col] - df_original[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        axes[1].plot(df_original.index, difference,
                    color='green', linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].fill_between(df_original.index, difference, 0, alpha=0.3, color='green')
        axes[1].set_title('Pre-processing Delta (Processed - Interpolated)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Difference')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Distribution comparison
        axes[2].hist(df_original[col].dropna(), bins=50, alpha=0.5,
                    label='Original', color='steelblue', density=True, edgecolor='black')
        axes[2].hist(df_processed[col].dropna(), bins=50, alpha=0.5,
                    label='Processed', color='orangered', density=True, edgecolor='black')

        try:
            df_original[col].dropna().plot.kde(ax=axes[2], linewidth=2,
                                              color='steelblue', label='Original KDE')
            df_processed[col].dropna().plot.kde(ax=axes[2], linewidth=2,
                                               color='orangered', label='Processed KDE')
        except:
            pass

        axes[2].set_title('Data Distribution Comparison', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Density')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = output_path / f'preprocessing_comparison_{col.replace("/", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {save_path}")

    # 2. Statistical comparison summary
    print("\n2. Creating statistical comparison summary...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Preprocessing Statistical Summary', fontsize=16, fontweight='bold')

    stats_orig = df_original.describe().T
    stats_proc = df_processed.describe().T
    x = np.arange(len(df_original.columns))
    width = 0.35

    # Mean Comparison
    axes[0, 0].bar(x - width/2, stats_orig['mean'], width, label='Original', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, stats_proc['mean'], width, label='Processed', alpha=0.8, color='orangered')
    axes[0, 0].set_title('Mean Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df_original.columns, rotation=45, ha='right')
    axes[0, 0].legend()

    # Std Comparison
    axes[0, 1].bar(x - width/2, stats_orig['std'], width, label='Original', alpha=0.8, color='steelblue')
    axes[0, 1].bar(x + width/2, stats_proc['std'], width, label='Processed', alpha=0.8, color='orangered')
    axes[0, 1].set_title('Std Deviation Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_original.columns, rotation=45, ha='right')
    axes[0, 1].legend()

    # Range Comparison
    range_orig = stats_orig['max'] - stats_orig['min']
    range_proc = stats_proc['max'] - stats_proc['min']
    axes[1, 0].bar(x - width/2, range_orig, width, label='Original', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x + width/2, range_proc, width, label='Processed', alpha=0.8, color='orangered')
    axes[1, 0].set_title('Value Range (Max - Min)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df_original.columns, rotation=45, ha='right')
    axes[1, 0].legend()

    # Quality Comparison
    missing_orig = df_original.isnull().sum()
    outliers_orig = []
    outliers_proc = []
    for col in df_original.columns:
        z_orig = np.abs(stats.zscore(df_original[col].dropna()))
        z_proc = np.abs(stats.zscore(df_processed[col].dropna()))
        outliers_orig.append((z_orig > 3).sum())
        outliers_proc.append((z_proc > 3).sum())

    axes[1, 1].bar(x - width, missing_orig, width/2, label='Missing (Orig)', alpha=0.8, color='red')
    axes[1, 1].bar(x, outliers_orig, width/2, label='Outliers (Orig)', alpha=0.8, color='orange')
    axes[1, 1].bar(x + width, outliers_proc, width/2, label='Outliers (Proc)', alpha=0.8, color='green')
    axes[1, 1].set_title('Data Quality Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df_original.columns, rotation=45, ha='right')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path / 'preprocessing_statistics_summary.png', dpi=300)
    plt.close()

    # 3. Correlation matrix comparison
    print("\n3. Creating correlation matrix comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Correlation Matrix Comparison', fontsize=16, fontweight='bold')

    sns.heatmap(df_original.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title('Original Correlation', fontsize=12)

    sns.heatmap(df_processed.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title('Processed Correlation', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / 'preprocessing_correlation_comparison.png', dpi=300)
    plt.close()

    # 4. Multi-variable overview
    print("\n4. Creating multi-variable overview plot...")
    n_cols = len(df_original.columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 4 * n_cols), sharex=True)
    if n_cols == 1: axes = [axes]
    fig.suptitle('Multivariate Time Series Preprocessing Overview', fontsize=16, fontweight='bold')

    for idx, col in enumerate(df_original.columns):
        axes[idx].plot(df_original.index, df_original[col], label='Original', alpha=0.5, color='steelblue')
        axes[idx].plot(df_processed.index, df_processed[col], label='Processed', alpha=0.8, color='orangered')
        axes[idx].set_ylabel(col, fontsize=10)
        axes[idx].legend(loc='upper right')

        # Mark detected outliers
        z_scores = np.abs(stats.zscore(df_original[col].fillna(df_original[col].median())))
        mask = z_scores > 3
        if mask.any():
            axes[idx].scatter(df_original.index[mask], df_original[col][mask], color='red', s=20, marker='x', label='Outliers')

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(output_path / 'preprocessing_multivariate_overview.png', dpi=300)
    plt.close()

    print(f"\nCharts saved to: {output_path}")
    return output_path


def create_preprocessing_report(df_original, df_processed, output_dir='./visualization'):
    """Create a detailed text report summarizing preprocessing changes."""
    output_path = Path(output_dir)
    report_path = output_path / 'preprocessing_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TIME SERIES PREPROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {pd.Timestamp.now()}\n")
        f.write(f"Data Points: {len(df_original)}\n\n")

        for col in df_original.columns:
            f.write(f"Variable: {col}\n" + "-"*30 + "\n")
            f.write(f"Original Missing: {df_original[col].isnull().sum()}\n")
            f.write(f"Original Mean: {df_original[col].mean():.4f}\n")
            f.write(f"Processed Mean: {df_processed[col].mean():.4f}\n\n")

    print(f"\nReport saved: {report_path}")
    return report_path


# ============================================================================
# Step 3: Format for Chronos-2
# ============================================================================

def prepare_chronos_data(df, target_cols, covariate_cols, test_hours=12):
    """Prepare data in Chronos-2 format (List of Dicts)."""
    print("\n" + "=" * 80)
    print("Step 3: Formatting Data for Chronos-2")
    print("=" * 80)

    split_idx = len(df) - test_hours
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    train_inputs, test_contexts, ground_truths = [], [], []

    for target_col in target_cols:
        if target_col not in df.columns: continue

        data_dict = {
            "target": df_train[target_col].values.astype(np.float32),
            "past_covariates": {cov: df_train[cov].values.astype(np.float32) for cov in covariate_cols if cov in df_train.columns},
            "future_covariates": {}
        }
        train_inputs.append(data_dict)
        test_contexts.append(data_dict)
        ground_truths.append(df_test[target_col].values.astype(np.float32))

    return train_inputs, test_contexts, ground_truths, df_train, df_test


# ============================================================================
# Step 4: Model & Fine-tuning (LoRA)
# ============================================================================

def load_and_finetune_model(train_inputs, prediction_length=12):
    """Load Chronos-2 model and perform LoRA fine-tuning."""
    print("\n" + "=" * 80)
    print("Step 4: Loading Model and Fine-tuning with LoRA")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = "/home/jovyan/document/chronos2/chronos_model_local"

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model path not found: {local_model_path}")

    pipeline = BaseChronosPipeline.from_pretrained(
        local_model_path,
        device_map=device,
        dtype="auto"
    )

    print("\nStarting LoRA Fine-tuning (500 steps)...")
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=prediction_length,
        finetune_mode="lora",
        learning_rate=1e-4,
        batch_size=4,
        num_steps=500,
        logging_steps=50,
    )

    return pipeline, finetuned_pipeline

def generate_predictions_and_visualize(pipeline, finetuned_pipeline,
                                       test_contexts, ground_truths,
                                       target_cols, df_train, df_test,
                                       prediction_length=12):
    """Generate predictions and save numerical results."""
    print("\n" + "=" * 80)
    print("Step 5: Generating Predictions")
    print("=" * 80)

    quantile_levels = [0.1, 0.5, 0.9]
    os.makedirs("./data", exist_ok=True)

    for idx, (target_col, test_context, ground_truth) in enumerate(zip(target_cols, test_contexts, ground_truths)):
        inputs = [{
            "target": test_context["target"],
            "past_covariates": test_context["past_covariates"],
            "future_covariates": {}
        }]

        # Zero-shot
        zs_res, _ = pipeline.predict_quantiles(inputs, prediction_length, quantile_levels)
        zs = zs_res[0].cpu().numpy() if hasattr(zs_res[0], 'cpu') else zs_res[0]

        # Fine-tuned
        ft_res, _ = finetuned_pipeline.predict_quantiles(inputs, prediction_length, quantile_levels)
        ft = ft_res[0].cpu().numpy() if hasattr(ft_res[0], 'cpu') else ft_res[0]

        # Extract Median (index 1) or fall back
        z_p = zs[1,:] if zs.shape[0] >= 3 else zs[0,:]
        f_p = ft[1,:] if ft.shape[0] >= 3 else ft[0,:]

        with open(f"./data/data{idx}.txt", "w") as f:
            f.write(f"Target: {target_col}\n")
            f.write(f"Zero-Shot Pred: {z_p.tolist()}\n")
            f.write(f"Fine-Tuned Pred: {f_p.tolist()}\n")
            f.write(f"Ground Truth: {ground_truth.tolist()}\n")

        print(f"Predictions saved for {target_col}")


def main():
    print("\nCHRONOS-2 FINE-TUNING FOR STEEL PRODUCTION")

    # 1. Load
    targets, covariates = load_metadata('./info2.xlsx')
    df_combined = load_timeseries_data(
        targets['code'].tolist() + covariates['code'].tolist(),
        targets['chnName'].tolist() + covariates['chnName'].tolist()
    )

    # 2. Preprocess & Visualize (English)
    df_processed = preprocess_data(df_combined)
    visualize_preprocessing_comparison(df_combined, df_processed)
    create_preprocessing_report(df_combined, df_processed)

    # 3. Prepare
    train_in, test_ctx, gt, _, _ = prepare_chronos_data(
        df_processed, targets['chnName'].tolist(), covariates['chnName'].tolist()
    )

    # 4. Train
    base_m, ft_m = load_and_finetune_model(train_in)

    # 5. Predict
    generate_predictions_and_visualize(base_m, ft_m, test_ctx, gt, targets['chnName'].tolist(), None, None)

    print("\nEXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()