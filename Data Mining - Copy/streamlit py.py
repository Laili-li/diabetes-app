import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Penting jika menggunakan scaler untuk input baru

# --- Memuat Model dan Scaler ---
try:
    knn_model_diabetes = joblib.load('knn_model_diabetes.pkl')
    scaler_diabetes = joblib.load('scaler_diabetes.pkl')
    X_test_diabetes = joblib.load('X_test_diabetes.pkl')
    Y_test_diabetes = joblib.load('Y_test_diabetes.pkl')
    st.success("Model Diabetes dan Scaler berhasil dimuat.") # Ini akan muncul di Streamlit
except FileNotFoundError:
    st.error("Error: File 'knn_model_diabetes.pkl' atau file terkait diabetes tidak ditemukan. Pastikan sudah dijalankan 'train_models.py' dan file .pkl berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

try:
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler_clustering = joblib.load('scaler_clustering.pkl')
    X_test_clustering = joblib.load('X_test_clustering.pkl')
    st.success("Model K-Means dan Scaler Clustering berhasil dimuat.") # Ini akan muncul di Streamlit
except FileNotFoundError:
    st.error("Error: File 'kmeans_model.pkl' atau file terkait clustering tidak ditemukan. Pastikan sudah dijalankan 'train_models.py' dan file .pkl berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Model Deployment UAS")

# --- Sidebar untuk Navigasi ---
st.sidebar.title("Streamlit Apps")
st.sidebar.markdown("Collection of my apps deployed in Streamlit for Data Mining UAS.")
page = st.sidebar.radio("Pilih Tampilan", ["Klasifikasi Diabetes", "Clustering K-Means"])
st.sidebar.write("---")
st.sidebar.header("Dosen Pengampu")
st.sidebar.write("Teuku Rizky Noviandy, S.Kom., M.Kom.")
st.sidebar.write("Universitas Abulyatama")

# --- Konten Halaman ---

if page == "Klasifikasi Diabetes":
    st.title("Klasifikasi Diabetes")
    st.write("""
    Aplikasi ini memprediksi apakah seseorang cenderung menderita diabetes berdasarkan beberapa fitur kesehatan.
    Model ini dilatih menggunakan algoritma *KNeighborsClassifier (KNN)*.
    """)

    st.header("Metrik Klasifikasi pada Data Testing")

    # Prediksi pada data testing untuk metrik
    Y_pred_diabetes = knn_model_diabetes.predict(X_test_diabetes)

    # Akurasi
    accuracy = accuracy_score(Y_test_diabetes, Y_pred_diabetes)
    st.metric(label="Akurasi Model", value=f"{accuracy*100:.2f}%")

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(Y_test_diabetes, Y_pred_diabetes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(Y_test_diabetes, Y_pred_diabetes)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetes (0)', 'Diabetes (1)'], yticklabels=['Non-Diabetes (0)', 'Diabetes (1)'], ax=ax_cm)
    ax_cm.set_xlabel('Prediksi')
    ax_cm.set_ylabel('Aktual')
    ax_cm.set_title('Confusion Matrix Klasifikasi Diabetes')
    st.pyplot(fig_cm)
    plt.close(fig_cm) # Tutup figure untuk menghemat memori

    st.header("Input Data Baru untuk Prediksi Diabetes")
    st.write("Masukkan nilai untuk fitur-fitur berikut:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=17, value=1)
        glucose = st.number_input("Glukosa (mg/dL)", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=0, max_value=122, value=70)

    with col2:
        skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=99, value=20)
        insulin = st.number_input("Insulin (muU/ml)", min_value=0, max_value=846, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0)

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=2.500, format="%.3f", value=0.5)
        age = st.number_input("Usia", min_value=0, max_value=120, value=30)

    # Nama kolom sesuai urutan fitur saat training
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data_diabetes = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                       columns=feature_names)

    # Tombol Prediksi
    if st.button("Prediksi Diabetes"):
        # Skalakan input data baru menggunakan scaler yang sama saat training
        input_data_diabetes_scaled = scaler_diabetes.transform(input_data_diabetes)
        prediction = knn_model_diabetes.predict(input_data_diabetes_scaled)
        prediction_proba = knn_model_diabetes.predict_proba(input_data_diabetes_scaled)

        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.warning("Berdasarkan data yang dimasukkan, orang ini *cenderung menderita diabetes*.")
        else:
            st.success("Berdasarkan data yang dimasukkan, orang ini *tidak cenderung menderita diabetes*.")

        st.write(f"Probabilitas Diabetes: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Probabilitas Tidak Diabetes: {prediction_proba[0][0]*100:.2f}%")

elif page == "Clustering K-Means":
    st.title("Clustering K-Means")
    st.write("""
    Aplikasi ini mendemonstrasikan hasil clustering menggunakan algoritma K-Means.
    Data dikelompokkan menjadi 5 cluster berdasarkan fitur 'x' dan 'y'.
    """)

    st.header("Visualisasi Hasil Clustering Pada Data Testing")
    # Prediksi cluster untuk data testing
    cluster_labels_test = kmeans_model.predict(X_test_clustering)

    # Pastikan X_test_clustering adalah DataFrame untuk memudahkan plot
    # Karena X_test_clustering disave setelah scaling, kita mungkin perlu inversenya
    # Tapi untuk visualisasi sederhana, kita bisa langsung pakai nilai scaled
    # Atau lebih baik lagi, visualisasikan data asli (jika tersedia) atau data test yang belum discale
    # Untuk demo ini, kita akan plot data scaled X_test_clustering
    # Pastikan data ini punya setidaknya 2 fitur untuk scatter plot
    if X_test_clustering.shape[1] >= 2:
        df_plot_clustering = pd.DataFrame(X_test_clustering[:, :2], columns=['Fitur X (Scaled)', 'Fitur Y (Scaled)'])
        df_plot_clustering['Cluster'] = cluster_labels_test

        fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='Fitur X (Scaled)', y='Fitur Y (Scaled)', hue='Cluster',
                        palette='viridis', legend='full', data=df_plot_clustering, ax=ax_cluster)

        # Opsional: Tampilkan centroids jika model K-Means menyediakannya
        # centroids_scaled = kmeans_model.cluster_centers_
        # ax_cluster.scatter(centroids_scaled[:, 0], centroids_scaled[:, 1], s=300, c='red', marker='X', label='Centroids')
        # ax_cluster.legend()

        ax_cluster.set_title('Visualisasi Hasil Clustering K-Means pada Data Testing (5 Cluster)')
        st.pyplot(fig_cluster)
        plt.close(fig_cluster)
    else:
        st.warning("Data testing clustering tidak memiliki cukup fitur (minimal 2) untuk visualisasi scatter plot.")
        st.write("Visualisasi hanya dimungkinkan jika data clustering memiliki setidaknya 2 fitur numerik.")


    st.header("Input Data Baru untuk Clustering")
    st.write("Masukkan nilai untuk fitur 'x' dan 'y' yang akan di-cluster:")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        feature_x = st.number_input("Nilai Fitur X", value=0.0)
    with col_c2:
        feature_y = st.number_input("Nilai Fitur Y", value=0.0)

    input_data_clustering_new = pd.DataFrame([[feature_x, feature_y]], columns=['x', 'y'])

    # Tombol Tentukan Cluster
    if st.button("Tentukan Cluster untuk Data Baru"):
        # Skalakan input data baru menggunakan scaler clustering
        input_data_clustering_scaled = scaler_clustering.transform(input_data_clustering_new)
        cluster_prediction = kmeans_model.predict(input_data_clustering_scaled)

        st.subheader("Hasil Clustering")
st.success(f"Data baru ini termasuk dalam *Cluster: {cluster_prediction[0]}*")
