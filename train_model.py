import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Klasifikasi Diabetes (KNN) ---
try:
    diabetes_dataset = pd.read_csv('diabetes.csv')
    print("Dataset diabetes.csv berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'diabetes.csv' tidak ditemukan.")
    exit()

X_diabetes = diabetes_dataset.drop(columns='Outcome', axis=1)
Y_diabetes = diabetes_dataset['Outcome']

scaler_diabetes = StandardScaler()
X_diabetes_scaled = scaler_diabetes.fit_transform(X_diabetes)

X_train_diabetes, X_test_diabetes, Y_train_diabetes, Y_test_diabetes = train_test_split(
    X_diabetes_scaled, Y_diabetes, test_size=0.2, stratify=Y_diabetes, random_state=2
)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_diabetes, Y_train_diabetes)

# Simpan model KNN dan data
joblib.dump(knn_classifier, 'knn_model_diabetes.pkl')
joblib.dump(scaler_diabetes, 'scaler_diabetes.pkl')
joblib.dump(X_test_diabetes, 'X_test_diabetes.pkl')
joblib.dump(Y_test_diabetes, 'Y_test_diabetes.pkl')
print("Model KNN dan scaler berhasil disimpan.")

# --- Clustering KMeans ---
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

clustering_url = 'https://drive.google.com/uc?id=1QuDHd-pebfOY-DoyFACtMyK-_UyuMJP8'
try:
    clustering_data = pd.read_csv(clustering_url)
    print("Dataset clustering berhasil dimuat.")
except:
    print("Gagal memuat dataset clustering.")
    exit()

features = ['x', 'y']
X_clustering = clustering_data[features]

scaler_clustering = StandardScaler()
X_clustering_scaled = scaler_clustering.fit_transform(X_clustering)

X_train_clustering, X_test_clustering = train_test_split(X_clustering_scaled, test_size=0.2, random_state=42)

kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_model.fit(X_train_clustering)

joblib.dump(kmeans_model, 'kmeans_model.pkl')
joblib.dump(scaler_clustering, 'scaler_clustering.pkl')
joblib.dump(X_test_clustering, 'X_test_clustering.pkl')
print("Model KMeans dan scaler clustering berhasil disimpan.")
