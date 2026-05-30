"""
Feature Bridge Module for MOMENT + Chronos-2 Joint Training
============================================================
Bridges MOMENT's 768-dim output to Chronos-2's 512-dim input space.

Key Components:
- Linear projector for dimension alignment
- Learnable future tokens as prediction placeholders
- Integration with Chronos-2's time encoding mechanism
"""

import torch
import torch.nn as nn
from typing import Tuple


class FeatureBridge(nn.Module):
    """
    Feature projection bridge between MOMENT and Chronos-2.

    This module:
    1. Projects MOMENT output from 768-dim to 512-dim (Chronos-2 input dim)
    2. Provides learnable future tokens as placeholders for prediction
    3. Enables gradient flow from Chronos-2 back to MOMENT
    """

    def __init__(
        self,
        moment_dim: int = 768,
        chronos_dim: int = 512,
        num_future_patches: int = 12,
        use_residual: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize feature bridge.

        Args:
            moment_dim: MOMENT output dimension (768 for MOMENT-1-large)
            chronos_dim: Chronos-2 input dimension (512 by default)
            num_future_patches: Number of future patches (prediction_length / patch_size)
            use_residual: Whether to use residual connection (requires moment_dim == chronos_dim)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.moment_dim = moment_dim
        self.chronos_dim = chronos_dim
        self.num_future_patches = num_future_patches
        self.use_residual = use_residual

        # 1. Linear projector for dimension alignment
        self.projector = nn.Linear(moment_dim, chronos_dim)

        # Optional: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(chronos_dim)

        # Optional: Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # 2. Learnable future tokens (placeholders for prediction horizon)
        # Shape: (num_future_patches, chronos_dim)
        # These are learnable parameters that will be optimized during training
        self.future_tokens = nn.Parameter(
            torch.randn(num_future_patches, chronos_dim) * 0.02  # Small initialization
        )

        # Residual connection (if dimensions match)
        if use_residual:
            assert moment_dim == chronos_dim, \
                f"Residual connection requires moment_dim == chronos_dim, got {moment_dim} != {chronos_dim}"
            self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        else:
            self.residual_weight = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projector weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def forward(
        self,
        moment_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project MOMENT features and provide future tokens.

        Args:
            moment_features: MOMENT encoder output
                Shape: (batch_size, num_history_patches, moment_dim) or
                       (batch_size, num_history_patches, num_channels, moment_dim)

        Returns:
            projected_history: Projected historical features
                Shape: (batch_size, num_history_patches, chronos_dim)
            future_tokens: Learnable future tokens expanded for batch
                Shape: (batch_size, num_future_patches, chronos_dim)
        """
        # Handle different input shapes
        if moment_features.ndim == 4:
            # Shape: (batch_size, num_patches, num_channels, moment_dim)
            # Average over channel dimension
            batch_size, num_patches, num_channels, moment_dim = moment_features.shape
            moment_features = moment_features.mean(dim=2)  # (B, N, D)
        elif moment_features.ndim == 3:
            # Shape: (batch_size, num_patches, moment_dim)
            batch_size, num_patches, moment_dim = moment_features.shape
        else:
            raise ValueError(
                f"Unexpected moment_features shape: {moment_features.shape}. "
                f"Expected 3D or 4D tensor."
            )

        # Project MOMENT output to Chronos-2 dimension
        projected = self.projector(moment_features)  # (B, H, chronos_dim)

        # Apply residual connection if enabled
        if self.use_residual and self.residual_weight is not None:
            projected = self.residual_weight * projected + (1 - self.residual_weight) * moment_features

        # Apply layer normalization
        projected = self.layer_norm(projected)

        # Apply dropout
        projected = self.dropout(projected)

        # Expand learnable future tokens for the batch
        # Future tokens: (num_future_patches, chronos_dim) -> (B, F, chronos_dim)
        future = self.future_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        return projected, future

    def get_combined_sequence(
        self,
        moment_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get combined sequence of history + future tokens.

        This is a convenience method that combines projected history and future tokens.

        Args:
            moment_features: MOMENT encoder output
                Shape: (batch_size, num_history_patches, moment_dim)

        Returns:
            combined_sequence: Concatenated history and future
                Shape: (batch_size, num_history_patches + num_future_patches, chronos_dim)
        """
        projected_history, future_tokens = self.forward(moment_features)

        # Concatenate along sequence dimension
        combined = torch.cat([projected_history, future_tokens], dim=1)

        return combined


class ResidualBridge(FeatureBridge):
    """
    Feature bridge with enhanced residual connections.

    This variant uses a residual block similar to Chronos-2's ResidualBlock
    for more robust projection.
    """

    def __init__(
        self,
        moment_dim: int = 768,
        chronos_dim: int = 512,
        num_future_patches: int = 12,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize residual bridge.

        Args:
            moment_dim: MOMENT output dimension
            chronos_dim: Chronos-2 input dimension
            num_future_patches: Number of future patches
            hidden_dim: Hidden dimension for residual block (default: chronos_dim * 2)
            dropout: Dropout rate
        """
        # Don't call super().__init__() directly, we'll override the projector
        nn.Module.__init__(self)

        self.moment_dim = moment_dim
        self.chronos_dim = chronos_dim
        self.num_future_patches = num_future_patches

        hidden_dim = hidden_dim or chronos_dim * 2

        # Residual block projector (inspired by Chronos-2's ResidualBlock)
        self.hidden_layer = nn.Linear(moment_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, chronos_dim)
        self.residual_layer = nn.Linear(moment_dim, chronos_dim)

        self.layer_norm = nn.LayerNorm(chronos_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable future tokens
        self.future_tokens = nn.Parameter(
            torch.randn(num_future_patches, chronos_dim) * 0.02
        )

        # Initialize weights
        self._init_residual_weights()

    def _init_residual_weights(self):
        """Initialize residual block weights."""
        # Xavier initialization for all linear layers
        for layer in [self.hidden_layer, self.output_layer, self.residual_layer]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        moment_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project using residual block.

        Args:
            moment_features: MOMENT encoder output
                Shape: (batch_size, num_history_patches, moment_dim) or
                       (batch_size, num_history_patches, num_channels, moment_dim)

        Returns:
            projected_history: Projected historical features
            future_tokens: Learnable future tokens
        """
        # Handle different input shapes
        if moment_features.ndim == 4:
            # Shape: (batch_size, num_patches, num_channels, moment_dim)
            # Average over channel dimension
            moment_features = moment_features.mean(dim=2)  # (B, N, D)
        elif moment_features.ndim != 3:
            raise ValueError(
                f"Unexpected moment_features shape: {moment_features.shape}. "
                f"Expected 3D or 4D tensor."
            )

        batch_size = moment_features.shape[0]

        # Residual block forward pass
        hidden = self.activation(self.hidden_layer(moment_features))
        output = self.dropout(self.output_layer(hidden))
        residual = self.residual_layer(moment_features)

        # Combine with residual
        projected = output + residual

        # Layer norm
        projected = self.layer_norm(projected)

        # Expand future tokens
        future = self.future_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        return projected, future


def create_bridge(
    moment_dim: int = 768,
    chronos_dim: int = 512,
    num_future_patches: int = 12,
    bridge_type: str = "simple",
    **kwargs
) -> FeatureBridge:
    """
    Factory function to create feature bridge.

    Args:
        moment_dim: MOMENT output dimension
        chronos_dim: Chronos-2 input dimension
        num_future_patches: Number of future patches
        bridge_type: Type of bridge ('simple' or 'residual')
        **kwargs: Additional arguments for bridge

    Returns:
        FeatureBridge instance
    """
    if bridge_type == "simple":
        return FeatureBridge(
            moment_dim=moment_dim,
            chronos_dim=chronos_dim,
            num_future_patches=num_future_patches,
            **kwargs
        )
    elif bridge_type == "residual":
        return ResidualBridge(
            moment_dim=moment_dim,
            chronos_dim=chronos_dim,
            num_future_patches=num_future_patches,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown bridge_type: {bridge_type}. Choose 'simple' or 'residual'.")


if __name__ == "__main__":
    # Test feature bridge
    print("Testing FeatureBridge...")

    # Create bridge
    bridge = FeatureBridge(
        moment_dim=768,
        chronos_dim=512,
        num_future_patches=12
    )

    # Test forward pass
    batch_size = 4
    num_patches = 64
    moment_output = torch.randn(batch_size, num_patches, 768)

    print(f"Input shape: {moment_output.shape}")

    projected_history, future_tokens = bridge(moment_output)

    print(f"Projected history shape: {projected_history.shape}")
    print(f"Future tokens shape: {future_tokens.shape}")

    # Test combined sequence
    combined = bridge.get_combined_sequence(moment_output)
    print(f"Combined sequence shape: {combined.shape}")

    # Verify shapes
    assert projected_history.shape == (batch_size, num_patches, 512)
    assert future_tokens.shape == (batch_size, 12, 512)
    assert combined.shape == (batch_size, num_patches + 12, 512)

    print("\n✓ Simple bridge test passed!")

    # Test residual bridge
    print("\nTesting ResidualBridge...")

    residual_bridge = ResidualBridge(
        moment_dim=768,
        chronos_dim=512,
        num_future_patches=12
    )

    projected_history, future_tokens = residual_bridge(moment_output)

    print(f"Projected history shape: {projected_history.shape}")
    print(f"Future tokens shape: {future_tokens.shape}")

    assert projected_history.shape == (batch_size, num_patches, 512)
    assert future_tokens.shape == (batch_size, 12, 512)

    print("\n✓ Residual bridge test passed!")

    # Test factory function
    print("\nTesting bridge factory...")

    simple_bridge = create_bridge(bridge_type="simple")
    residual_bridge = create_bridge(bridge_type="residual")

    print("✓ Factory function test passed!")

    print("\n✓ All bridge tests passed!")
