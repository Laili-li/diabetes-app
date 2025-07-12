import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("diabetes.csv")

# --- KLASIFIKASI ---
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dan scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_classification = StandardScaler()
X_train_scaled = scaler_classification.fit_transform(X_train)
X_test_scaled = scaler_classification.transform(X_test)

# Train model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Save model klasifikasi
with open("knn_model_diabetes.pkl", "wb") as f:
    pickle.dump(knn_model, f)
with open("scaler_diabetes.pkl", "wb") as f:
    pickle.dump(scaler_classification, f)
with open("X_test_diabetes.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)
with open("Y_test_diabetes.pkl", "wb") as f:
    pickle.dump(y_test, f)

# --- CLUSTERING ---
# Gunakan hanya beberapa fitur numerik (misalnya BMI dan Glucose)
X_cluster = df[['Glucose', 'BMI']]
scaler_clustering = StandardScaler()
X_cluster_scaled = scaler_clustering.fit_transform(X_cluster)

kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_cluster_scaled)

# Save model clustering
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_model, f)
with open("scaler_clustering.pkl", "wb") as f:
    pickle.dump(scaler_clustering, f)
with open("X_test_clustering.pkl", "wb") as f:
    pickle.dump(X_cluster_scaled, f)
