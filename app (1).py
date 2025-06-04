import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
# For simplicity, assume the data is already loaded in the Colab environment or you can load it again here
# If running this locally, you'll need to adjust the path
try:
    df = pd.read_csv('diabetes.csv') # Assuming diabetes.csv is in the same directory or adjust path
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'diabetes.csv' is in the correct path.")
    st.stop()


# --- Page 1: Dataset, Karakteristik, dan Visualisasi ---
def page_dataset():
    st.title("Analisis Dataset Diabetes")

    st.header("Dataset")
    st.write("Berikut adalah beberapa baris pertama dari dataset:")
    st.dataframe(df.head())

    st.header("Karakteristik Dataset")
    st.write("Informasi umum tentang dataset:")
    st.write(df.info())

    st.write("Statistik deskriptif dataset:")
    st.write(df.describe())

    st.header("Visualisasi Data")

    st.subheader("Distribusi Target (Outcome)")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Beberapa Fitur")
    features_to_plot = ['Glucose', 'BMI', 'Age']
    for feature in features_to_plot:
        st.write(f"Distribusi: {feature}")
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

# --- Page 2: Pelatihan Model ---
def page_training():
    st.title("Pelatihan Model Prediksi Diabetes")

    st.header("Model: Regresi Logistik")
    st.write("Kita akan menggunakan Regresi Logistik untuk memprediksi apakah seseorang menderita diabetes atau tidak.")

    if st.button("Latih Model"):
        # Pisahkan fitur (X) dan target (y)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # Bagi data menjadi set pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi dan latih model
        model = LogisticRegression(max_iter=200) # Tambahkan max_iter jika muncul ConvergenceWarning
        model.fit(X_train, y_train)

        # Lakukan prediksi
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        st.subheader("Hasil Pelatihan")
        st.write(f"Akurasi Model: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        st.code(conf_matrix)

        st.subheader("Classification Report")
        st.code(class_report)

        # Simpan model untuk digunakan di halaman prediksi
        joblib.dump(model, 'logistic_regression_model.pkl')
        st.success("Model berhasil dilatih dan disimpan!")

# --- Page 3: Formulir Prediksi ---
def page_prediction():
    st.title("Formulir Prediksi Diabetes")

    st.write("Masukkan nilai-nilai karakteristik untuk memprediksi apakah seseorang menderita diabetes.")

    # Muat model yang telah dilatih
    try:
        model = joblib.load('logistic_regression_model.pkl')
        st.success("Model prediksi berhasil dimuat.")
    except FileNotFoundError:
        st.error("Model belum dilatih atau disimpan. Silakan latih model di halaman 'Pelatihan Model' terlebih dahulu.")
        st.stop() # Hentikan eksekusi jika model belum tersedia

    # Buat input field untuk setiap fitur
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Konsentrasi Glukosa Plasma (mg/dL)", min_value=0, max_value=300, value=120)
    bloodpressure = st.number_input("Tekanan Darah Diastolik (mm Hg)", min_value=0, max_value=150, value=70)
    skinthickness = st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Serum 2 Jam (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Fungsi Silsilah Diabetes", min_value=0.000, max_value=3.000, value=0.500)
    age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=30)

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi"):
        # Siapkan data input dalam format DataFrame
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [bloodpressure],
            'SkinThickness': [skinthickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Lakukan prediksi
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.error("Diprediksi Menderita Diabetes")
        else:
            st.success("Diprediksi Tidak Menderita Diabetes")

        st.write(f"Probabilitas Menderita Diabetes: {prediction_proba[0][1]:.2f}")
        st.write(f"Probabilitas Tidak Menderita Diabetes: {prediction_proba[0][0]:.2f}")


# --- Navigasi Halaman ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Dataset, Karakteristik, dan Visualisasi", "Pelatihan Model", "Formulir Prediksi"])

if page == "Dataset, Karakteristik, dan Visualisasi":
    page_dataset()
elif page == "Pelatihan Model":
    page_training()
elif page == "Formulir Prediksi":
    page_prediction()
