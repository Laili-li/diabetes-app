import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle

# =====================
# STEP 1: Load Dataset
# =====================
data = pd.read_csv("diabetes.csv")  # GUNAKAN FILE TUGAS PERTAMA ANDA

# =====================
# STEP 2: Klasifikasi
# =====================
# Pisahkan fitur dan label
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Normalisasi fitur
scaler_classification = StandardScaler()
X_scaled = scaler_classification.fit_transform(X)

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Buat dan latih model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Simpan model klasifikasi dan datanya
pickle.dump(knn_model, open("knn_model_diabetes.pkl", "wb"))
pickle.dump(scaler_classification, open("scaler_diabetes.pkl", "wb"))
pickle.dump(X_test, open("X_test_diabetes.pkl", "wb"))
pickle.dump(y_test, open("Y_test_diabetes.pkl", "wb"))

# =====================
# STEP 3: Clustering
# =====================
# Clustering menggunakan semua fitur (tanpa Outcome)
scaler_clustering = StandardScaler()
X_cluster = scaler_clustering.fit_transform(X)

# Buat dan latih model KMeans
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_cluster)

# Simpan model clustering dan data
pickle.dump(kmeans_model, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler_clustering, open("scaler_clustering.pkl", "wb"))
pickle.dump(X_cluster, open("X_test_clustering.pkl", "wb"))

print("✅ Semua file model berhasil disimpan.")
