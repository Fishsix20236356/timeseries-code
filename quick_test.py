"""Quick test to verify the fix"""
import torch
from joint_model import JointMOMENTChronos2
from joint_config import get_default_config

print("Quick test of JointMOMENTChronos2...")

config = get_default_config()
print("Creating model...")
model = JointMOMENTChronos2(config)

print("\nTesting forward pass...")
batch_size = 2
batch = {
    'target': torch.randn(batch_size, 1, config.context_length),
    'past_covariates': {},
    'future_target': torch.randn(batch_size, 1, config.prediction_length),
    'future_covariates': {},
}

model.eval()
with torch.no_grad():
    outputs = model(batch)

print(f"✓ Forward pass successful!")
print(f"  Input shape: {batch['target'].shape}")
print(f"  Output shape: {outputs.quantile_preds.shape}")
print(f"  Expected: ({batch_size}, {len(config.chronos_quantiles)}, {config.prediction_length})")

assert outputs.quantile_preds.shape == (batch_size, len(config.chronos_quantiles), config.prediction_length)
print("\n✓ All tests passed!")