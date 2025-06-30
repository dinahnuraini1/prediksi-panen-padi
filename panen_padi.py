import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import time
import joblib
import os
# Fungsi untuk memuat objek dari file pickle
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj



# STREAMLIT
def main():
    # Menampilkan gambar menggunakan st.image dengan pengaturan width
   
    
    
       
    # Sidebar Menu
    st.sidebar.image("image/panen.png", width=200) 
   # Menambahkan judul besar di sidebar
    st.sidebar.markdown("<h2 style='font-size: 24px;'> select menu</h2>", unsafe_allow_html=True)

    # Pilihan menu di sidebar tanpa label "Pilih Menu"
    menu = st.sidebar.selectbox(
        "",
        ["Home", "Load Data", "Preprocessing", "Random Forest Modelling","Random Forest + PSO Modelling", "Predictions"]
    )
    if menu == "Home":
        st.markdown("""
            <div style='text-align: center; padding: 50px 0;'>
                <h1 style='font-size: 50px; margin-bottom: 10px; color: #2E7D32;'>🌾 Welcome To 🌾</h1>
                <h2 style='font-size: 40px; margin-bottom: 20px;'>Rice Harvest Prediction System</h2>
                <h4 style='color: #555;'>Using <span style='color: #1E88E5;'>Random Forest Regression</span> + 
                <span style='color: #F9A825;'>Particle Swarm Optimization</span></h4>
            </div>
            <div style='text-align: left; padding: 0 40px; font-size: 18px;'>
            <p><strong>Sistem ini memiliki beberapa menu utama di antaranya:</strong></p>
            <ol>
                <li><strong>Load Data</strong> – Untuk memuat dataset kelompok tani dari Desa Mojorayung. Berformat Comma Separated Values (CSV) </li>
                <li><strong>Preprocessing</strong> – Untuk melakukan preprocessing data</li>
                <li><strong>Random Forest Modelling</strong> – Untuk melatih model menggunakan Random Forest </li>
                <li><strong>Random Forest Modelling + PSO </strong> – Untuk melatih model menggunakan Random Forest dan PSO </li>
                <li><strong>Predictions </strong> – Untuk memprediksi hasil panen berdasarkan input pengguna. Tersedia 6 fitur diantaranya:</li>
                    <ul>
                            <li>Luas Tanam (HA)</li>
                            <li>Pupuk Urea (KG)</li>
                            <li>Pupuk NPK (KG)</li>
                            <li>Pupuk Organik (KG)</li>
                            <li>Jumlah Bibit (KG)</li>
                            <li>Varietas Padi</li>
                    </ul>
            </ol>
            </div>
        """, unsafe_allow_html=True)

    
    # Inisialisasi session_state
    if "data" not in st.session_state:
        st.session_state["data"] = None
        st.session_state["selected_features"] = None
        st.session_state["X"] = None
        st.session_state["y"] = None
        st.session_state["scaler_X"] = None
        st.session_state["scaler_y"] = None

    if menu == "Load Data":
        st.header("1. Load Data")
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        
        if uploaded_file is not None:
            # Baca data
            data = pd.read_csv(uploaded_file)
            
            # Ambil hanya kolom yang dibutuhkan
            required_columns = [
                'luas_tanam', 'urea', 'npk', 'organik',
                'jumlah_bibit', 'varietas', 'hasil_panen'
            ]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
            else:
                data = data[required_columns]
                st.session_state["data"] = data
                st.session_state["X"] = data.drop(columns=["hasil_panen"]).values
                st.session_state["y"] = data["hasil_panen"].values

                st.success("✅ Data berhasil dimuat.")
                st.write("Data yang diunggah:")
                st.dataframe(data.head())

    elif menu == "Preprocessing":
        st.header("2. Preprocessing")

        if "data" not in st.session_state or st.session_state["data"] is None or st.session_state["data"].empty:
            st.warning("Harap upload data terlebih dahulu di menu 'Load Data'.")
        else:
            df = st.session_state["data"].copy()

            # ===== One-Hot Encoding untuk 'varietas' =====
            st.subheader("One-Hot Encoding untuk Kolom varietas")
            try:
                if "varietas" in df.columns:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[["varietas"]])
                    feature_names = encoder.get_feature_names_out(["varietas"])
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                    df_encoded = pd.concat([df.drop(columns=["varietas"]), encoded_df], axis=1)

                    # Simpan hasil encoding & encoder
                    st.session_state["df_processed"] = df_encoded
                    st.session_state["one_hot_encoders"] = {"varietas": encoder}
                    st.session_state["one_hot_encoded"] = True

                    st.success("One-Hot Encoding berhasil dilakukan untuk kolom 'varietas'.")
                    st.dataframe(df_encoded.head())
                else:
                    st.warning("Kolom 'varietas' tidak ditemukan dalam data.")
                    df_encoded = df.copy()
            except Exception as e:
                st.error(f"Terjadi kesalahan saat One-Hot Encoding: {e}")
                df_encoded = df.copy()

            # ===== Normalisasi =====
            st.subheader("Normalisasi Fitur dan Target")
            try:
                df_normalized = df_encoded.copy()

                # Pisahkan kolom numerik, kecuali hasil_panen
                kolom_normalisasi = df_normalized.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if 'hasil_panen' in kolom_normalisasi:
                    kolom_normalisasi.remove("hasil_panen")

                # Normalisasi X
                scaler_X = MinMaxScaler()
                df_normalized[kolom_normalisasi] = scaler_X.fit_transform(df_normalized[kolom_normalisasi])

                # Normalisasi y
                scaler_y = MinMaxScaler()
                if "hasil_panen" in df_normalized.columns:
                    df_normalized["hasil_panen"] = scaler_y.fit_transform(df_normalized[["hasil_panen"]])

                # Simpan hasil
                st.session_state["scaler_X"] = scaler_X
                st.session_state["scaler_y"] = scaler_y
                st.session_state["df_normalized"] = df_normalized
                st.session_state["normalized"] = True

                # Simpan X dan y
                st.session_state["X"] = df_normalized.drop(columns=["hasil_panen"])
                st.session_state["y"] = df_normalized["hasil_panen"]

                st.success("Normalisasi berhasil dilakukan.")
                st.write("Data setelah Normalisasi:")
                st.dataframe(df_normalized.head())
            except Exception as e:
                st.error(f"Terjadi kesalahan saat normalisasi: {e}")


    elif menu == "Random Forest Modelling":
        st.header("Random Forest Modelling")

        if "X" not in st.session_state or "y" not in st.session_state:
            st.warning("Harap lakukan preprocessing terlebih dahulu.")
        elif "normalized" not in st.session_state or not st.session_state["normalized"]:
            st.warning("⚠️ Harap lakukan normalisasi data terlebih dahulu sebelum melanjutkan ke pemodelan Random Forest.")
        else:
            # Mapping rasio -> test_size dan file model
            rasio_opsi = {
                "50:50": {"test_size": 0.5, "model_data": "model/rf5.pkl"},
                "60:40": {"test_size": 0.4, "model_data": "model/rf4.pkl"},
                "70:30": {"test_size": 0.3, "model_data": "model/rf3.pkl"},
                "80:20": {"test_size": 0.2, "model_data": "model/rf2.pkl"},
                "90:10": {"test_size": 0.1, "model_data": "model/rf1.pkl"},
            }

            # Pilihan rasio dari dropdown
            selected_rasio_label = st.selectbox("Pilih rasio data latih dan uji:", list(rasio_opsi.keys()))
            selected_rasio = rasio_opsi[selected_rasio_label]

            # Hitung jumlah data
            total_data = len(st.session_state["X"])
            train_count = int((1 - selected_rasio["test_size"]) * total_data)
            test_count = int(selected_rasio["test_size"] * total_data)

            st.info(f"Jumlah data latih: {train_count}")
            st.info(f"Jumlah data uji: {test_count}")

            # Split data hanya untuk info, bukan buat latih ulang
            X = st.session_state["X"]
            y = st.session_state["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=selected_rasio["test_size"], random_state=42
            )

            model_path = selected_rasio["model_data"]  # path file model

            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)

                    model_rf = model_data.get("model")
                    params = model_data.get("params", {})
                    mape_train = params.get("mape_train")
                    mape_test = params.get("mape_test")

                    if model_rf and params and mape_train is not None and mape_test is not None:
                        # Tampilkan parameter model dalam input field yang tidak bisa diubah (read-only)
                        st.subheader("📌 Parameter Model Random Forest")

                        st.number_input("Jumlah pohon (n_estimators)", value=params.get("n_estimators", 0), disabled=True)
                        st.number_input("Kedalaman maksimum pohon (max_depth)", value=params.get("max_depth", 0), disabled=True)
                        st.number_input("Fitur maksimum (max_features)", value=params.get("max_features", 0), disabled=True)
                        # st.success("Model berhasil dimuat!")
                        st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                        st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")
                        # Tambahan: jika MAPE < 10%, sarankan untuk optimasi
                        if mape_test > 10:
                            st.warning("📈 MAPE Testing > 10%. Lakukan optimasi menggunakan PSO.")
                    else:
                        st.error("Beberapa parameter model tidak ditemukan dalam file.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memuat model: {e}")
            else:
                st.error("File model tidak ditemukan.")

    elif menu == "Random Forest + PSO Modelling":
        st.header("Random Forest + PSO Modelling")

        if "X" not in st.session_state or "y" not in st.session_state:
            st.warning("Harap lakukan preprocessing terlebih dahulu.")
        elif "normalized" not in st.session_state or not st.session_state["normalized"]:
            st.warning("⚠️ Harap lakukan normalisasi data terlebih dahulu sebelum melanjutkan ke pemodelan Random Forest.")

        else:
            st.info("Silakan lakukan proses optimasi Random Forest menggunakan PSO di sini.")

            # Mapping rasio ke file model hasil optimasi
            rasio_opsi_pso = {
                "50:50": "model/rfpso_1.pkl",
                "60:40": "model/rfpso_2.pkl",
                "70:30": "model/rfpso_3.pkl",
                "80:20": "model/rfpso_4.pkl",
                "90:10": "model/rfpso_5.pkl",
            }

            # Dropdown untuk pilih rasio
            selected_rasio_label = st.selectbox("Pilih rasio data latih dan uji:", list(rasio_opsi_pso.keys()))
            model_path_pso = rasio_opsi_pso[selected_rasio_label]

            # Hitung dan tampilkan jumlah data train-test
            total_data = len(st.session_state["X"])
            test_size = {
                "50:50": 0.5,
                "60:40": 0.4,
                "70:30": 0.3,
                "80:20": 0.2,
                "90:10": 0.1
            }[selected_rasio_label]
            train_count = int((1 - test_size) * total_data)
            test_count = int(test_size * total_data)

            st.info(f"Jumlah data latih: {train_count}")
            st.info(f"Jumlah data uji: {test_count}")

            # Cek dan load file model PSO
            if os.path.exists(model_path_pso):
                try:
                    with open(model_path_pso, "rb") as f:
                        model_data = pickle.load(f)

                    model_rf_pso = model_data.get("model")
                    params = model_data.get("params", {})
                    mape_train = params.get("mape_train")
                    mape_test = params.get("mape_test")

                    if model_rf_pso and mape_train is not None and mape_test is not None:
                        st.subheader("📌 Parameter Hasil Optimasi (PSO)")

                        # Tampilkan parameter hasil PSO (readonly)
                        st.number_input("Jumlah pohon (n_estimators)", value=params.get("n_estimators", 0), disabled=True)
                        st.number_input("Kedalaman maksimum pohon (max_depth)", value=params.get("max_depth", 0), disabled=True)
                        st.number_input("Max features", value=params.get("max_features", 1), disabled=True)

                        st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                        st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")

                        # Evaluasi kategori berdasarkan nilai MAPE Testing
                        if mape_test < 10:
                            st.success("🎯 Nilai MAPE Testing dalam kategori **SANGAT BAIK**")
                        elif 10 <= mape_test < 20:
                            st.success("✅ Nilai MAPE Testing dalam kategori **BAIK** ")
                        elif 20 <= mape_test <= 50:
                            st.warning("⚠️ Nilai MAPE Testing dalam kategori **CUKUP BAIK**")
                        else:
                            st.error("❌ Nilai MAPE Testing dalam kategori **BURUK** ")

                    else:
                        st.error("Parameter model atau nilai MAPE tidak ditemukan.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memuat model: {e}")
            else:
                st.error("File model hasil optimasi PSO tidak ditemukan.")

    elif menu == "Predictions":
        st.header("Prediksi Hasil Panen")
        st.markdown("""
            <div style='text-align: center;'>
                <h4 style='color: #555;'>
                    <span style='color: #1E88E5;'> Menggunakan Model Terbaik dari Random Forest Regression</span> + 
                    <span style='color: #F9A825;'>Particle Swarm Optimization dengan Rasio Data Testing 0.1</span>
                </h4>
            </div> 
        """, unsafe_allow_html=True)


        # === 1. One-Hot Encoder untuk Varietas (default jika tidak dari training) ===
        def create_default_varietas_encoder():
            list_varietas = ["serang bentis", "ciherang", "toyoarum", "inpari 32", "inpari 13"]
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoder.fit(pd.DataFrame(list_varietas, columns=["varietas"]))
            return encoder

        if "one_hot_encoders" not in st.session_state:
            st.session_state["one_hot_encoders"] = {}

        if "varietas" not in st.session_state["one_hot_encoders"]:
            st.session_state["one_hot_encoders"]["varietas"] = create_default_varietas_encoder()

        encoder = st.session_state["one_hot_encoders"]["varietas"]

        # === 2. Load Model dan Scaler ===
        if "model_rf_pso_best" not in st.session_state:
            model_path = "model/rfpso_5.pkl"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                st.session_state["model_rf_pso_best"] = model_data.get("model")
                st.session_state["scaler_X"] = model_data.get("scaler_X")
                st.session_state["scaler_y"] = model_data.get("scaler_y")
            else:
                st.session_state["model_rf_pso_best"] = None

        # === 3. Input Fitur ===
        st.subheader("Masukkan Nilai Fitur:")

        luas_tanam = st.number_input("Luas Tanam (HA)", min_value=0.0)
        urea = st.number_input("Pupuk Urea (KG)", min_value=0.0)
        npk = st.number_input("Pupuk NPK (KG)", min_value=0.0)
        organik = st.number_input("Pupuk Organik (KG)", min_value=0.0)
        jumlah_bibit = st.number_input("Jumlah Bibit (KG)", min_value=0.0)

        varietas_padi = st.selectbox(
            "Varietas Padi",
            ["serang bentis", "ciherang", "toyoarum", "inpari 32", "inpari 13"]
        )

        if st.button("Prediksi Hasil Panen"):
            try:
                # === 4. Siapkan DataFrame input ===
                input_dict = {
                    "luas_tanam": luas_tanam,
                    "urea": urea,
                    "npk": npk,
                    "organik": organik,
                    "jumlah_bibit": jumlah_bibit,
                    "varietas": varietas_padi
                }
                input_df = pd.DataFrame([input_dict])

                # === 5. One-hot encoding varietas ===
                encoded = encoder.transform(input_df[["varietas"]])
                encoded_df = pd.DataFrame(
                    encoded, columns=encoder.get_feature_names_out(["varietas"])
                )
                input_df.drop(columns=["varietas"], inplace=True)
                input_df = pd.concat([input_df, encoded_df], axis=1)

                # === 6. Pastikan semua fitur lengkap dan urut ===
                final_features = [
                    "luas_tanam", "urea", "npk", "organik", "jumlah_bibit",
                    "varietas_ciherang", "varietas_inpari 13", "varietas_inpari 32",
                    "varietas_serang bentis", "varietas_toyoarum"
                ]
                for col in final_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[final_features]

                # === 7. Normalisasi ===
                scaler_X = st.session_state.get("scaler_X")
                if scaler_X is not None:
                    input_scaled = scaler_X.transform(input_df)
                else:
                    input_scaled = input_df.values

                # === 8. Prediksi ===
                model = st.session_state.get("model_rf_pso_best")
                if model is None:
                    st.warning("Model belum tersedia.")
                    st.stop()

                hasil_normalized = model.predict(input_scaled).reshape(-1, 1)

                # === 9. Inverse transform hasil prediksi ===
                scaler_y = st.session_state.get("scaler_y")
                if scaler_y is not None:
                    hasil_panen = scaler_y.inverse_transform(hasil_normalized)
                else:
                    hasil_panen = hasil_normalized

                st.success(f"🌾 Prediksi Hasil Panen Padi Adalah: **{hasil_panen[0][0]:,.2f}** Ton")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")


if __name__ == "__main__":
    main()

