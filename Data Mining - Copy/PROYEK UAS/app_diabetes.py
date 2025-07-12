import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load file .pkl
knn_model = pickle.load(open("knn_model_diabetes.pkl", "rb"))
scaler_diabetes = pickle.load(open("scaler_diabetes.pkl", "rb"))
X_test_classification = pickle.load(open("X_test_diabetes.pkl", "rb"))
y_test_classification = pickle.load(open("Y_test_diabetes.pkl", "rb"))

kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler_clustering = pickle.load(open("scaler_clustering.pkl", "rb"))
X_test_clustering = pickle.load(open("X_test_clustering.pkl", "rb"))

# Judul
st.title("Aplikasi Klasifikasi dan Clustering Diabetes")

# Menu
menu = st.sidebar.selectbox("Pilih Menu", ["Klasifikasi Diabetes", "Clustering Pasien"])

# =============================
# KLASIFIKASI
# =============================
if menu == "Klasifikasi Diabetes":
    st.header("🔍 Klasifikasi Diabetes (KNN)")

    # Input user
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Prediksi"):
        # Buat array dan scaling
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler_diabetes.transform(input_data)

        # Prediksi
        prediction = knn_model.predict(input_scaled)[0]
        hasil = "POSITIF Diabetes" if prediction == 1 else "NEGATIF Diabetes"
        st.success(f"Hasil Prediksi: {hasil}")

# =============================
# CLUSTERING
# =============================
elif menu == "Clustering Pasien":
    st.header("📊 Clustering Pasien Diabetes (K-Means)")

    st.write("Cluster ini dibuat dari semua data pasien berdasarkan fitur (tanpa label Outcome).")
    
    # Prediksi cluster untuk data test
    cluster_labels = kmeans_model.predict(X_test_clustering)
    
    df_cluster = pd.DataFrame(X_test_clustering, columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    df_cluster["Cluster"] = cluster_labels

    st.write("Hasil Cluster dari Sampel Data:")
    st.dataframe(df_cluster.head(15))

    st.bar_chart(df_cluster["Cluster"].value_counts())
