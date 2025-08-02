import numpy as np
import matplotlib.pyplot as plt

# Simulate training and validation errors based on capacity and regularization
def simulate_errors(model_capacity, reg_strength):
    # Training error: Decreases as the model capacity increases (more layers/units => more fitting ability)
    training_error = 1 / (model_capacity + 1) + np.random.normal(0, 0.02)
    
    # Validation error: 
    # - Decreases with increasing capacity, but with weak regularization it increases due to overfitting
    # - With strong regularization, the model is penalized more, balancing the error
    validation_error = 1 / (model_capacity + 1) + np.random.normal(0, 0.1) + (reg_strength * 0.1)
    
    return training_error, validation_error

# Model capacity and regularization values (small to large models and weak to strong regularization)
model_capacities = [2, 4, 8, 16, 32]  # Simulating different model sizes (layers/units)
regularization_strengths = [0.001, 0.01, 0.1, 1.0]  # Weak to strong regularization (λ)

fig, axes = plt.subplots(len(model_capacities), len(regularization_strengths), figsize=(15, 12))

# Plotting training vs validation errors for each combination of model capacity and regularization
for i, model_capacity in enumerate(model_capacities):
    for j, reg_strength in enumerate(regularization_strengths):
        # Simulate the errors for the current configuration
        training_error, validation_error = simulate_errors(model_capacity, reg_strength)
        
        # Plot the errors in the correct subplot
        ax = axes[i, j]
        ax.plot([1, model_capacity], [training_error, training_error], label='Training Error', color='blue', marker='o')
        ax.plot([1, model_capacity], [validation_error, validation_error], label='Validation Error', color='red', marker='x')
        
        ax.set_title(f"Capacity: {model_capacity}, λ: {reg_strength}")
        ax.set_xlabel('Model Capacity (Layers/Units)')
        ax.set_ylabel('Error')
        ax.set_xticks([1, model_capacity])
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
