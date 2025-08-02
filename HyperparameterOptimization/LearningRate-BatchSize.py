import numpy as np
import matplotlib.pyplot as plt

# Function to apply the Linear Scaling Rule
def linear_scaling_rule(base_batch_size, base_lr, batch_size_range):
    scaling_factor = np.sqrt(batch_size_range / base_batch_size)
    scaled_lr = base_lr * scaling_factor
    return scaled_lr

# Define various base values for batch sizes and learning rates
base_batch_sizes = [16, 32, 64, 128]
base_lrs = [0.005, 0.01, 0.02, 0.05]

# Batch sizes to experiment with
batch_size_range = np.array([2**i for i in range(4, 11)])  # Batch sizes: [16, 32, 64, ..., 1024]

# Set up the plot grid
fig, axes = plt.subplots(len(base_batch_sizes), len(base_lrs), figsize=(15, 10))

# Iterate through all combinations of base batch sizes and learning rates
for i, base_batch_size in enumerate(base_batch_sizes):
    for j, base_lr in enumerate(base_lrs):
        # Get the scaled learning rates for the given base batch size and learning rate
        scaled_lrs = linear_scaling_rule(base_batch_size, base_lr, batch_size_range)
        
        ax = axes[i, j]
        ax.plot(batch_size_range, scaled_lrs, marker='o', linestyle='-', color='b')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"B={base_batch_size}, α={base_lr}")
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.show()
