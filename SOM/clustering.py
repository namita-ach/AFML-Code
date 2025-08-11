import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

np.random.seed(1)
spread = 0.4 # change this and try
data1 = np.random.randn(50, 2) * spread + np.array([0, 0])
data2 = np.random.randn(50, 2) * spread + np.array([8, 8])
data3 = np.random.randn(50, 2) * spread + np.array([-8, 8])
data4 = np.random.randn(50, 2) * spread + np.array([8, -8])
data = np.vstack([data1, data2, data3, data4])

# train SOM
som_x, som_y = 8, 8  # bigger grid
som = MiniSom(x=som_x, y=som_y, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 1000)

# Assign clusters
cluster_map = {}
current_cluster = 0
for i in range(som_x):
    for j in range(som_y):
        cluster_map[(i, j)] = current_cluster
        current_cluster += 1

cluster_ids = []
for point in data:
    winner = som.winner(point)
    cluster_ids.append(cluster_map[winner])
cluster_ids = np.array(cluster_ids)

plt.figure(figsize=(7, 7))
scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_ids, cmap='tab20', s=70, edgecolors='k')
plt.title("SOM Clustering Result")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
