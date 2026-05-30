# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Python-based time series forecasting research project** that implements a joint training framework combining:
- **MOMENT**: A universal time series feature extractor (T5-based encoder)
- **Chronos-2**: A probabilistic forecasting model with context learning capabilities

The goal is to leverage MOMENT's strong feature extraction with Chronos-2's probabilistic forecasting and quantile prediction abilities.

**Language**: Python 3.10+ (uses modern type hints like `|` for union types)
**Framework**: PyTorch with Hugging Face Transformers
**License**: Apache-2.0 (Amazon)

## Core Architecture

### Five-Stage Joint Training Framework

The architecture follows a detailed design documented in [方案.md](方案.md):

1. **Dual-Track Preprocessing**: Separate input pipelines
   - MOMENT input: Uses RevIN (Reversible Instance Normalization)
   - Chronos-2 target: Uses standardization + arcsinh transformation

2. **History Encoding via MOMENT**: Feature extraction from historical data
   - Freezes patch embedding and early transformer layers
   - Unfreezes only last 1-2 transformer blocks for domain adaptation

3. **Feature Projection & Temporal Reconstruction**: Bridge between models
   - Linear projection to align MOMENT output dimension to Chronos-2 input dimension
   - Learnable future tokens as placeholders for prediction horizon
   - Meta-feature injection: Time index embeddings and mask embeddings
   - REG token insertion as separator between history and future

4. **Forecasting via Chronos-2**: T5 encoder with dual attention
   - LoRA (Low-Rank Adaptation) fine-tuning on attention layers
   - Time attention for temporal dependencies
   - Group attention for cross-variate information sharing

5. **Loss Calculation**: Quantile loss (Pinball loss)
   - Outputs 21 quantile predictions for probabilistic forecasting
   - Backpropagation through all unfrozen components

### Key Model Components

**[model.py](model.py)** - Chronos-2 implementation
- `Chronos2Model`: Main model class (PreTrainedModel)
- `Chronos2Encoder`: T5-style encoder with dual attention mechanisms
- `Chronos2EncoderBlock`: Individual encoder block containing:
  - `TimeSelfAttention`: Temporal dependency modeling
  - `GroupSelfAttention`: Cross-variate information sharing
  - `FeedForward`: Feed-forward network
- Key methods:
  - `encode()`: Feature extraction from context
  - `forward()`: Full forward pass with loss computation
  - `_prepare_patched_context()`: Context preprocessing with patching
  - `_compute_loss()`: Quantile loss (Pinball loss) calculation

**[moment.py](moment.py)** - MOMENT model integration
- `MOMENT`: Wrapper around T5 encoder for time series
- Task-specific heads: `PretrainHead`, `ClassificationHead`, `ForecastingHead`
- Supports patch embedding and RevIN normalization

**[layers.py](layers.py)** - Transformer components
- `RoPE`: Rotary Position Embedding
- `Chronos2LayerNorm`: T5-style layer normalization
- `MHA`: Multi-Head Attention
- `TimeSelfAttention`, `GroupSelfAttention`: Specialized attention mechanisms
- `ResidualBlock`: Residual connections with layer norm

**[dataset.py](dataset.py)** - Data handling
- `validate_and_prepare_single_dict_task()`: Task validation and preparation
- `left_pad_and_cat_2D()`: Tensor padding and concatenation
- Supports multivariate time series with covariates

**[revin.py](revin.py)** - Reversible Instance Normalization
- `RevIN`: Normalization/denormalization for time series
- Handles NaN values and optional learnable affine parameters

**[embed.py](embed.py)** - Embedding utilities
- `PositionalEmbedding`: Sinusoidal positional encoding
- `TokenEmbedding`: Learnable token embeddings
- `TemporalEmbedding`: Temporal feature embeddings

**[config.py](config.py)** - Configuration classes
- `Chronos2CoreConfig`: Model architecture configuration
  - Default: `d_model=512`, `d_kv=64`, `d_ff=2048`, `num_layers=6`, `num_heads=8`
  - Uses RoPE with `rope_theta=10000.0`
  - SDPA (Scaled Dot-Product Attention) implementation
  - Special tokens: `vocab_size=2` (PAD and REG tokens)
- `Chronos2ForecastingConfig`: Forecasting task configuration
  - `context_length`: Historical window size
  - `input_patch_size` / `output_patch_size`: Patch dimensions
  - `quantiles`: List of quantile levels (default 21 quantiles)
  - `use_reg_token`: Whether to use REG token separator
  - `use_arcsinh`: Whether to use inverse hyperbolic sine transformation

## Model Flow

```
Input Context (batch_size, context_length)
    ↓
Instance Normalization (scaling)
    ↓
Patching (divide into patches)
    ↓
Input Patch Embedding (ResidualBlock)
    ↓
Encoder Stack (6 layers of dual attention)
    ├─ TimeSelfAttention (temporal dependencies)
    ├─ GroupSelfAttention (cross-variate mixing)
    └─ FeedForward
    ↓
Output Patch Embedding (ResidualBlock)
    ↓
Quantile Predictions (batch_size, num_quantiles, prediction_length)
    ↓
Instance Normalization Inverse (unscaling)
```

## Key Design Decisions

### Fine-tuning Strategy
- **MOMENT**: Freeze patch embedding and early layers; unfreeze last 1-2 transformer blocks
- **Chronos-2**: Use LoRA on attention layers (Q, K, V, Output projections); freeze base weights

### Data Processing
- **MOMENT input**: Preserve RevIN normalization (matches pre-training distribution)
- **Chronos-2 target**: Apply standardization + arcsinh transformation for quantile head

### Feature Bridge
- Linear projector aligns MOMENT output dimension to Chronos-2 input dimension
- Learnable future tokens initialized as trainable parameters
- Explicit time index and mask embeddings added via element-wise addition
- REG token separates historical and future sequences

### Probabilistic Forecasting
- Outputs 21 quantiles by default for uncertainty quantification
- Uses Pinball loss (quantile loss) for training
- Instance normalization handles non-stationary time series

## Reference Documentation

- **[方案.md](方案.md)**: Comprehensive architecture design (in Chinese) with detailed 5-stage framework
- **[参考代码.txt](参考代码.txt)**: Module mapping and key file references (in Chinese)

## Dependencies

Core libraries:
- `torch` & `torch.nn`: Deep learning framework
- `transformers`: Hugging Face model utilities (PreTrainedModel, PretrainedConfig)
- `einops`: Tensor operations (rearrange, repeat)
- `momentfm`: MOMENT model library
- `numpy`: Numerical operations
- `sklearn`: Preprocessing utilities

## Important Notes

- This is a **research-grade implementation** focused on joint training of two pre-trained models
- The codebase uses **patch-based processing** to reduce sequence length while preserving temporal information
- **Dual attention mechanism** separates temporal and cross-variate attention for better interpretability
- All documentation in [方案.md](方案.md) and [参考代码.txt](参考代码.txt) is in Chinese
- Model architecture is based on T5 encoder with custom modifications for time series forecasting
