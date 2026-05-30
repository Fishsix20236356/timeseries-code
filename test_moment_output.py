"""Test script to check MOMENT output shape"""

import torch
from momentfm import MOMENTPipeline
from joint_config import get_default_config

print("Testing MOMENT output shape...")
config = get_default_config()

# Load MOMENT
moment = MOMENTPipeline.from_pretrained(
    config.moment_checkpoint,
    model_kwargs={'task_name': 'embedding'},
)
moment.init()

# Create dummy input
batch_size = 2
context_length = config.context_length
moment_input = torch.randn(batch_size, 1, context_length)

print(f"\nInput shape: {moment_input.shape}")

# Forward pass
moment_output = moment.embed(
    x_enc=moment_input,
    reduction='none'
)

print(f"\nMOMENT output type: {type(moment_output)}")
print(f"MOMENT output attributes: {dir(moment_output)}")

# Try to access embeddings
if hasattr(moment_output, 'embeddings'):
    embeddings = moment_output.embeddings
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")

    # Check if it's 4D
    if embeddings.ndim == 4:
        print(f"\nEmbeddings is 4D: (batch_size, num_patches, num_channels, hidden_dim)")
        print(f"  batch_size: {embeddings.shape[0]}")
        print(f"  num_patches: {embeddings.shape[1]}")
        print(f"  num_channels: {embeddings.shape[2]}")
        print(f"  hidden_dim: {embeddings.shape[3]}")

        # Squeeze the channel dimension if it's 1
        if embeddings.shape[2] == 1:
            embeddings_squeezed = embeddings.squeeze(2)
            print(f"\nAfter squeezing channel dim: {embeddings_squeezed.shape}")