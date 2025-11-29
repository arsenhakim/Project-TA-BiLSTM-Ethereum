import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="ETH-BiLSTM PREDICTION APP",
    page_icon="💎",
    layout="wide"
)

# --- 2. FUNGSI UTILS ---
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False), 
            input_shape=(10, 2)
        ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    model.predict(np.zeros((1, 10, 2))) 
    model.compile(optimizer='adam', loss='mae')
    return model

@st.experimental_singleton
def load_assets():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, 'model_weights_safe.json')
        scaler_param_path = os.path.join(base_dir, 'scalers_param.json')
        seed_path = os.path.join(base_dir, 'DATA_CONTOH.csv')
        
        model = build_model()
        with open(weights_path, 'r') as f:
            weights_list = json.load(f)
        weights_np = [np.array(w) for w in weights_list]
        model.set_weights(weights_np)
        
        with open(scaler_param_path, 'r') as f:
            scaler_params = json.load(f)
            
        scaler_features = MinMaxScaler()
        scaler_features.min_ = np.array(scaler_params['features']['min'])
        scaler_features.scale_ = np.array(scaler_params['features']['scale'])
        scaler_features.data_min_ = np.array(scaler_params['features']['data_min'])
        scaler_features.data_max_ = np.array(scaler_params['features']['data_max'])
        
        scaler_target = MinMaxScaler()
        scaler_target.min_ = np.array(scaler_params['target']['min'])
        scaler_target.scale_ = np.array(scaler_params['target']['scale'])
        scaler_target.data_min_ = np.array(scaler_params['target']['data_min'])
        scaler_target.data_max_ = np.array(scaler_params['target']['data_max'])
        
        try:
            seed_df = pd.read_csv(seed_path)
        except:
            seed_df = None
        
        return model, scaler_features, scaler_target, seed_df
        
    except Exception as e:
        return None, None, None, None

model, scaler_feat, scaler_target, seed_df = load_assets()

# --- 3. UI HEADER UTAMA ---
st.title("💎 Aplikasi Prediksi Harga Ethereum (Bi-LSTM)")
st.markdown("---")

if model is None:
    st.error("⚠️ Gagal memuat Model. Pastikan file JSON (Weights & Scaler) ada di folder.")
    st.stop()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 Input Data")
    st.info("Upload data historis untuk memulai analisis dan prediksi di Dashboard.")
    uploaded_file = st.file_uploader("Upload File (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
    st.markdown("---")
    
    # HAPUS INPUT FORECAST DAYS DAN GANTI DENGAN PESAN
    st.info("Prediksi dilakukan untuk 1 hari ke depan (t+1).")
    
    predict_btn = False
    if uploaded_file is not None:
        predict_btn = st.button("Proses & Prediksi 🚀")
    st.markdown("---")
    st.caption("Developed by: **Arsen Awali R.H (2043221129)**")

# --- 5. MAIN CONTENT (TABS) ---
tab_home, tab_dash = st.tabs(["🏠 Beranda & Info Model", "📈 Dashboard Analisis"])

# === TAB 1: BERANDA & INFO MODEL ===
with tab_home:
    st.header("Sistem Pendukung Keputusan Investasi Ethereum")
    st.markdown("""
    Aplikasi ini dikembangkan sebagai luaran **Proyek Akhir** untuk membantu investor memprediksi 
    pergerakan harga **Ethereum (ETH)** satu hari ke depan ($t+1$). Sistem menggunakan algoritma **Bi-LSTM** yang mampu 
    mempelajari pola data historis jangka panjang dengan mempertimbangkan sentimen pasar global (**S&P 500**).
    """)
    

    # B. VALIDASI PERFORMA (Langsung di root, bukan dalam kolom)
    st.subheader("🏆 Validasi Performa Model")
    st.write("Hasil evaluasi akurasi (MAPE) dari proses pengujian:")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Backtesting", "4.39%", "Sangat Akurat")
    m2.metric("Fwd Test (Agust)", "5.95%", "Sangat Akurat")
    m3.metric("Fwd Test (Sept)", "4.86%", "Robust")
    
    st.caption("*Mean Absolute Percentage Error (MAPE) < 10% dikategorikan Sangat Akurat.*")

    

    # C. SPESIFIKASI & PANDUAN
    col_kiri_bawah, col_kanan_bawah = st.columns(2)

    with col_kiri_bawah:
        st.subheader("🛠️ Spesifikasi Model")
        st.info("""
        * **Algoritma:** Bi-Directional LSTM
        * **Input:** Harga Ethereum & S&P 500
        * **Window Size:** 10 Hari
        * **Neurons:** 64 Unit (Dropout 0.1)
        * **Optimizer:** Adam (LR=0.005)
        """)

    with col_kanan_bawah:
        st.subheader("📖 Cara Penggunaan")
        st.markdown("""
        1. **Siapkan Data:** File Excel/CSV dengan format 3 kolom A,B,C (Date, ETH, S&P500).
        2. **Upload:** Gunakan panel di sebelah kiri.
        3. **Prediksi:** Klik tombol 'Proses & Prediksi'.
        4. **Analisis:** Hasil muncul di tab 'Dashboard'.
        """)

    if seed_df is not None:
        st.markdown("---")
        with st.expander("Lihat Contoh Format Data yang Dibutuhkan"):
            st.dataframe(seed_df)

# === TAB 2: DASHBOARD ANALISIS ===
with tab_dash:
    if uploaded_file is not None:
        try:
            # 1. BACA FILE
            file_ext = uploaded_file.name.split('.')[-1].lower() 
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] < 2:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')

            # 2. MAPPING KOLOM
            if df.shape[1] < 3:
                st.error("File minimal harus memiliki 3 kolom (Date, ETH, S&P500).")
                st.stop()
            
            df = df.iloc[:, 0:3]
            df.columns = ['Date', 'ETHUSD', 'S&P500']
            
            # 3. CLEANING
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            
            for col in ['ETHUSD', 'S&P500']:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

            # --- VISUALISASI DATA HISTORIS (ALTAIR - Scale Zero False) ---
            st.subheader("📊 Visualisasi Data Historis (Input)")
            
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("**Harga Ethereum (ETHUSD)**")
                chart_eth = alt.Chart(df).mark_line(color='#2962FF').encode(
                    x=alt.X('Date', axis=alt.Axis(format='%d %b')),
                    y=alt.Y('ETHUSD', scale=alt.Scale(zero=False), title='Harga (USD)'),
                    tooltip=['Date', 'ETHUSD']
                ).interactive()
                st.altair_chart(chart_eth, use_container_width=True)
                
            with c_chart2:
                st.markdown("**Indeks S&P 500**")
                chart_sp = alt.Chart(df).mark_line(color='#FF5252').encode(
                    x=alt.X('Date', axis=alt.Axis(format='%d %b')),
                    y=alt.Y('S&P500', scale=alt.Scale(zero=False), title='Indeks Points'),
                    tooltip=['Date', 'S&P500']
                ).interactive()
                st.altair_chart(chart_sp, use_container_width=True)

            with st.expander("🔍 Lihat Data Mentah"):
                st.dataframe(df.tail(10))

            # --- PREDIKSI (SINGLE STEP) ---
            if predict_btn:
                st.markdown("---")
                
                if len(df) < 10:
                    st.error("Data input kurang dari 10 baris. Model membutuhkan minimal 10 hari data historis.")
                else:
                    with st.spinner('Menghitung prediksi harga esok hari...'):
                        
                        last_10_days = df.tail(10)[['ETHUSD', 'S&P500']].values
                        
                        input_scaled = scaler_feat.transform(last_10_days)
                        model_input = input_scaled.reshape(1, 10, 2)
                        
                        pred_scaled = model.predict(model_input, verbose=0)
                        pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
                        
                        last_actual_price = last_10_days[-1][0]
                        delta_val = pred_price - last_actual_price
                        
                        last_date = df['Date'].iloc[-1]
                        pred_date = last_date + timedelta(days=1)
                        
                        # TAMPILKAN HASIL
                        st.success("✅ Prediksi Selesai!")
                        st.subheader(f"🔮 Hasil Prediksi ({pred_date.date()})")
                        
                        col_res, col_ket = st.columns([1, 2])
                        
                        with col_res:
                            st.metric(
                                label="Estimasi Harga ETH (t+1)",
                                value=f"${pred_price:,.2f}",
                                delta=f"{delta_val:,.2f} USD",
                                delta_color="normal"
                            )
                        
                        with col_ket:
                            st.info(f"""
                            Prediksi ini didasarkan pada pola data historis dari tanggal **{df['Date'].iloc[-10].date()}** sampai **{last_date.date()}**.
                            Model memproyeksikan pergerakan harga satu hari ke depan.
                            """)
                        
                        # --- GRAFIK GABUNGAN (ALTAIR) ---
                        st.markdown("### 📈 Tren & Titik Prediksi")
                        
                        # Data Historis (30 Hari Terakhir)
                        df_chart_hist = df[['Date', 'ETHUSD']].tail(30).copy()
                        df_chart_hist['Tipe'] = 'Historis'
                        
                        # Data Prediksi (2 Titik: Akhir Historis + Prediksi)
                        df_chart_pred = pd.DataFrame({
                            'Date': [last_date, pred_date],
                            'ETHUSD': [last_actual_price, pred_price],
                            'Tipe': ['Prediksi', 'Prediksi']
                        })
                        
                        # Gabungkan
                        df_final_chart = pd.concat([df_chart_hist, df_chart_pred], ignore_index=True)
                        
                        # Visualisasi Gabungan (Altair - Scale Zero False)
                        chart_final = alt.Chart(df_final_chart).mark_line(point=True).encode(
                            x=alt.X('Date', axis=alt.Axis(format='%d %b', title='Tanggal')),
                            y=alt.Y('ETHUSD', scale=alt.Scale(zero=False), title='Harga ETH (USD)'),
                            color=alt.Color('Tipe', scale=alt.Scale(domain=['Historis', 'Prediksi'], range=['#2962FF', '#00C853'])),
                            tooltip=['Date', 'ETHUSD', 'Tipe']
                        ).interactive()
                        
                        st.altair_chart(chart_final, use_container_width=True)

        except Exception as e:
            st.error("❌ Terjadi kesalahan membaca file.")
            st.code(str(e))
            st.warning("Pastikan format file benar (Excel/CSV) dengan minimal 3 kolom.")

    else:
        st.info("👈 Silakan upload file Excel/CSV di panel sebelah kiri untuk melihat Dashboard.")
        st.markdown("**Menunggu Input Data...**")