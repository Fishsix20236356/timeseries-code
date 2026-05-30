"""
Data Preprocessing Module for MOMENT + Chronos-2 Joint Training
================================================================
Implements high-fidelity data loading and conservative cleaning for steel production time series.

Key Features:
- Metadata parsing and variable selection
- Time series loading from JSON files
- Conservative cleaning (linear interpolation only)
- Dual-track dataset formatting for MOMENT and Chronos-2
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class MetadataParser:
    """Parse metadata from Excel file and filter target/covariate variables."""

    def __init__(self, metadata_path: str = './info2.xlsx'):
        """
        Initialize metadata parser.

        Args:
            metadata_path: Path to info2.xlsx metadata file
        """
        self.metadata_path = metadata_path
        self.target_keywords = ["铁水温度", "铁水Si", "炉渣二元碱度"]
        self.covariate_keywords = ["喷煤量", "冷水流量", "氧气流量", "风压力"]

    def parse(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse metadata and filter variables.

        Returns:
            targets: DataFrame with target variable metadata
            covariates: DataFrame with covariate metadata
        """
        print("=" * 80)
        print("Step 1: Loading Metadata")
        print("=" * 80)

        df_meta = pd.read_excel(self.metadata_path)
        print(f"Total variables in metadata: {len(df_meta)}")

        # Filter targets
        targets = df_meta[df_meta['chnName'].str.contains('|'.join(self.target_keywords), na=False)]
        print(f"\nTarget variables found: {len(targets)}")
        for _, row in targets.iterrows():
            print(f"  - {row['chnName']} (code: {row['code']})")

        # Filter covariates
        covariates = df_meta[df_meta['chnName'].str.contains('|'.join(self.covariate_keywords), na=False)]
        print(f"\nCovariate variables found: {len(covariates)}")
        for _, row in covariates.iterrows():
            print(f"  - {row['chnName']} (code: {row['code']})")

        return targets, covariates


class TimeSeriesLoader:
    """Load time series data from JSON files and merge into DataFrame."""

    def __init__(self, json_dir: str = './json'):
        """
        Initialize time series loader.

        Args:
            json_dir: Directory containing JSON files
        """
        self.json_dir = Path(json_dir)

    def load(self, codes: List[str], names: List[str]) -> pd.DataFrame:
        """
        Load time series data from JSON files.

        Args:
            codes: List of variable codes (matching JSON filenames)
            names: List of variable names (chnName from metadata)

        Returns:
            DataFrame with all variables merged by timestamp
        """
        print("\n" + "=" * 80)
        print("Step 2: Loading Time Series Data")
        print("=" * 80)

        all_data = {}

        for code, name in zip(codes, names):
            json_path = self.json_dir / f"{code}.json"

            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping...")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df_temp = pd.DataFrame(data)
            df_temp['time'] = pd.to_datetime(df_temp['time'])
            df_temp = df_temp.set_index('time').sort_index()

            # Use chnName as column name
            all_data[name] = df_temp['value']

            print(f"Loaded {name}: {len(df_temp)} records from {df_temp.index.min()} to {df_temp.index.max()}")

        # Merge all data into single DataFrame
        df_combined = pd.DataFrame(all_data)
        print(f"\nCombined DataFrame shape: {df_combined.shape}")
        print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
        print(f"Missing values per column:\n{df_combined.isnull().sum()}")

        return df_combined


class ConservativeCleaner:
    """
    Conservative data cleaning preserving fault signatures.

    Philosophy: Fix data corruption while preserving real operational fluctuations.
    """

    def __init__(self, max_gap: Optional[int] = None):
        """
        Initialize cleaner.

        Args:
            max_gap: Maximum continuous gap for linear interpolation (None = unlimited)
        """
        self.max_gap = max_gap

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply conservative cleaning pipeline.

        Args:
            df: Raw time series DataFrame

        Returns:
            Cleaned DataFrame in raw physical scale
        """
        print("\n" + "=" * 80)
        print("Step 3: Conservative Data Cleaning")
        print("=" * 80)

        df_cleaned = df.copy()

        # Step 1: Linear interpolation for missing values
        print("\n1. Handling missing values with linear interpolation...")
        missing_before = df_cleaned.isnull().sum().sum()

        if self.max_gap is not None:
            df_cleaned = df_cleaned.interpolate(
                method='linear',
                limit=self.max_gap,
                limit_direction='both'
            )
        else:
            df_cleaned = df_cleaned.interpolate(
                method='linear',
                limit_direction='both'
            )

        missing_after = df_cleaned.isnull().sum().sum()
        print(f"   Missing values: {missing_before} -> {missing_after}")

        # Step 2: Boundary filling
        print("\n2. Filling boundary NaN values...")
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
        missing_final = df_cleaned.isnull().sum().sum()
        print(f"   Remaining NaN values: {missing_final}")

        # Step 3: Physical constraint verification
        print("\n3. Removing sensor fault values (0, -999, inf)...")
        for col in df_cleaned.columns:
            # Count fault values
            fault_mask = (df_cleaned[col] == 0) | (df_cleaned[col] == -999) | np.isinf(df_cleaned[col])
            n_faults = fault_mask.sum()

            if n_faults > 0:
                print(f"   {col}: {n_faults} sensor faults detected")
                # Replace with NaN then interpolate
                df_cleaned.loc[fault_mask, col] = np.nan
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill').fillna(method='bfill')

        # Note: We DO NOT remove statistical outliers - they are fault signatures!
        print("\n4. Preserving statistical outliers (fault signatures)...")
        print("   Skipping aggressive outlier removal to preserve fault precursors")

        # Verify quality
        print("\n" + "=" * 80)
        print("Data Quality Report")
        print("=" * 80)
        print(f"Final shape: {df_cleaned.shape}")
        print(f"NaN count: {df_cleaned.isnull().sum().sum()} (target: 0)")
        print(f"Duplicate timestamps: {df_cleaned.index.duplicated().sum()} (target: 0)")
        print(f"Monotonic timestamps: {df_cleaned.index.is_monotonic_increasing}")
        print("\nOutput: Raw physical scale (NO normalization)")

        return df_cleaned


class JointDataset(Dataset):
    """
    Dataset for joint MOMENT + Chronos-2 training.

    Formats data as dictionary with targets and covariates.
    Normalization is handled in the model, not here.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        covariate_cols: List[str],
        context_length: int = 512,
        prediction_length: int = 96,
        stride: int = 1,
    ):
        """
        Initialize dataset.

        Args:
            df: Cleaned time series DataFrame
            target_cols: List of target variable names
            covariate_cols: List of covariate variable names
            context_length: Historical window size
            prediction_length: Forecast horizon
            stride: Stride for sliding window
        """
        self.df = df
        self.target_cols = target_cols
        self.covariate_cols = covariate_cols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride

        # Generate valid indices
        self.indices = self._generate_indices()

        print(f"\nDataset initialized:")
        print(f"  Total samples: {len(self.indices)}")
        print(f"  Context length: {context_length}")
        print(f"  Prediction length: {prediction_length}")
        print(f"  Targets: {target_cols}")
        print(f"  Covariates: {covariate_cols}")

    def _generate_indices(self) -> List[int]:
        """Generate valid start indices for sliding window."""
        total_length = len(self.df)
        required_length = self.context_length + self.prediction_length

        indices = []
        for i in range(0, total_length - required_length + 1, self.stride):
            indices.append(i)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - target: (n_targets, context_length) historical targets
                - past_covariates: dict of (context_length,) covariates
                - future_target: (n_targets, prediction_length) future targets
                - future_covariates: empty dict (no known future covariates)
        """
        start_idx = self.indices[idx]
        context_end = start_idx + self.context_length
        future_end = context_end + self.prediction_length

        # Extract context window
        context_df = self.df.iloc[start_idx:context_end]
        future_df = self.df.iloc[context_end:future_end]

        # Extract targets
        target_context = context_df[self.target_cols].values.T  # (n_targets, context_length)
        target_future = future_df[self.target_cols].values.T  # (n_targets, prediction_length)

        # Extract covariates
        past_covariates = {}
        for cov in self.covariate_cols:
            past_covariates[cov] = context_df[cov].values  # (context_length,)

        # Convert to tensors
        return {
            'target': torch.from_numpy(target_context).float(),
            'past_covariates': {k: torch.from_numpy(v).float() for k, v in past_covariates.items()},
            'future_target': torch.from_numpy(target_future).float(),
            'future_covariates': {},  # No known future covariates
        }


def load_and_prepare_data(
    metadata_path: str = './info2.xlsx',
    json_dir: str = './json',
    context_length: int = 512,
    prediction_length: int = 96,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[JointDataset, JointDataset, JointDataset]:
    """
    Complete data loading and preparation pipeline.

    Args:
        metadata_path: Path to metadata Excel file
        json_dir: Directory containing JSON files
        context_length: Historical window size
        prediction_length: Forecast horizon
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Step 1: Parse metadata
    parser = MetadataParser(metadata_path)
    targets, covariates = parser.parse()

    # Step 2: Load time series
    loader = TimeSeriesLoader(json_dir)
    all_codes = targets['code'].tolist() + covariates['code'].tolist()
    all_names = targets['chnName'].tolist() + covariates['chnName'].tolist()
    df_combined = loader.load(all_codes, all_names)

    # Step 3: Clean data
    cleaner = ConservativeCleaner(max_gap=10)
    df_cleaned = cleaner.clean(df_combined)

    # Step 4: Split into train/val/test
    total_len = len(df_cleaned)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df_cleaned.iloc[:train_end]
    df_val = df_cleaned.iloc[train_end:val_end]
    df_test = df_cleaned.iloc[val_end:]

    print(f"\n" + "=" * 80)
    print("Data Split")
    print("=" * 80)
    print(f"Train: {len(df_train)} samples ({train_ratio*100:.1f}%)")
    print(f"Val:   {len(df_val)} samples ({val_ratio*100:.1f}%)")
    print(f"Test:  {len(df_test)} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")

    # Step 5: Create datasets
    target_cols = targets['chnName'].tolist()
    covariate_cols = covariates['chnName'].tolist()

    train_dataset = JointDataset(
        df_train, target_cols, covariate_cols,
        context_length, prediction_length, stride=1
    )

    val_dataset = JointDataset(
        df_val, target_cols, covariate_cols,
        context_length, prediction_length, stride=prediction_length
    )

    test_dataset = JointDataset(
        df_test, target_cols, covariate_cols,
        context_length, prediction_length, stride=prediction_length
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test data loading pipeline
    print("Testing data preprocessing pipeline...")

    try:
        train_ds, val_ds, test_ds = load_and_prepare_data()

        print("\n" + "=" * 80)
        print("Testing dataset access...")
        print("=" * 80)

        sample = train_ds[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Future target shape: {sample['future_target'].shape}")
        print(f"Past covariates: {list(sample['past_covariates'].keys())}")

        print("\n✓ Data preprocessing pipeline test passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
