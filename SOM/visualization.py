import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Generate 3 clusters in 4D
np.random.seed(0)
data1 = np.random.normal(loc=0, scale=0.5, size=(50, 4))
data2 = np.random.normal(loc=3, scale=0.5, size=(50, 4))
data3 = np.random.normal(loc=-3, scale=0.5, size=(50, 4))
data = np.vstack([data1, data2, data3])

# Normalize data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Train SOM
som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# Visualization: U-Matrix
plt.figure(figsize=(7, 7))
u_matrix = som.distance_map()
plt.imshow(u_matrix, cmap='bone_r')
plt.colorbar(label='Distance between nodes')
plt.title("SOM U-Matrix (Topology Visualization)")
plt.show()
