import pickle
import numpy as np

# Contoh input manual
sample = np.array([[2, 120, 70, 30, 80, 25.0, 0.5, 30]])

# Load model & scaler
scaler = pickle.load(open("scaler_diabetes.pkl", "rb"))
model = pickle.load(open("knn_model_diabetes.pkl", "rb"))

# Transform & prediksi
scaled = scaler.transform(sample)
pred = model.predict(scaled)

print("Prediksi:", "Diabetes" if pred[0]==1 else "Tidak Diabetes")
