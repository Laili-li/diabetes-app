import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================
# Load Semua File .pkl
# =====================

# Ganti path sesuai direktori kamu kalau diperlukan
path_dir = "."

# Load file klasifikasi
knn_model = pickle.load(open(f"{path_dir}/knn_model_diabetes.pkl", "rb"))
scaler_classification = pickle.load(open(f"{path_dir}/scaler_diabetes.pkl", "rb"))
X_test = pickle.load(open(f"{path_dir}/X_test_diabetes.pkl", "rb"))
y_test = pickle.load(open(f"{path_dir}/Y_test_diabetes.pkl", "rb"))

# Load file clustering
kmeans_model = pickle.load(open(f"{path_dir}/kmeans_model.pkl", "rb"))
scaler_clustering = pickle.load(open(f"{path_dir}/scaler_clustering.pkl", "rb"))
X_cluster = pickle.load(open(f"{path_dir}/X_test_clustering.pkl", "rb"))

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Aplikasi Diabetes", layout="centered")
st.title("🔍 Aplikasi Prediksi dan Clustering Diabetes")

menu = st.sidebar.selectbox("Pilih Fitur", ["Klasifikasi", "Clustering"])

# =====================
# Klasifikasi Diabetes
# =====================
if menu == "Klasifikasi":
    st.header("🧪 Prediksi Diabetes (KNN)")

    # Input data
    pregnancies = st.number_input("Pregnancies", 0)
    glucose = st.number_input("Glucose", 0)
    bp = st.number_input("Blood Pressure", 0)
    skin = st.number_input("Skin Thickness", 0)
    insulin = st.number_input("Insulin", 0)
    bmi = st.number_input("BMI", 0.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0)
    age = st.number_input("Age", 0)

    if st.button("Prediksi"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler_classification.transform(input_data)
        prediction = knn_model.predict(input_scaled)

        hasil = "Positif Diabetes" if prediction[0] == 1 else "Negatif Diabetes"
        st.success(f"Hasil Prediksi: **{hasil}**")

# =====================
# Clustering Data
# =====================
elif menu == "Clustering":
    st.header("📊 Clustering Pasien (KMeans)")

    st.write("Menampilkan 10 data dari hasil clustering:")
    cluster_labels = kmeans_model.predict(X_cluster)
    df_cluster = pd.DataFrame(X_cluster, columns=[f"Fitur {i+1}" for i in range(X_cluster.shape[1])])
    df_cluster["Cluster"] = cluster_labels

    st.dataframe(df_cluster.head(10))

    # Visualisasi Pie Chart
    st.subheader("Distribusi Cluster")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    st.write(cluster_counts)

    st.pyplot(cluster_counts.plot.pie(autopct="%1.1f%%", figsize=(5,5)).get_figure())
