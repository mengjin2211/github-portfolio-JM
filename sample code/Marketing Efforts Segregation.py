#Marketing Efforts
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Load or simulate donor data
data = pd.DataFrame({
    "5-Year Donation Amount": np.random.randint(500, 50000, size=500),
    "Largest Gift": np.random.randint(500, 20000, size=500),
    "Lifetime Giving": np.random.randint(1000, 200000, size=500),
    "Wealth Score": np.random.randint(1, 11, size=500),
    "Donation Frequency": np.random.randint(1, 20, size=500),
    "Engagement Score": np.random.randint(50, 100, size=500)
})

# Step 2: Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Step 4: Fit the K-Means model with the optimal number of clusters
optimal_k = 4  # Choose based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Analyze the clusters
print(data.groupby('Cluster').mean())

# Step 6: Visualize the clusters (optional)
import seaborn as sns
sns.pairplot(data, hue="Cluster", palette="viridis")
plt.show()