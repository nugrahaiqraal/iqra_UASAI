import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# ============================================
# 1. KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Gagal Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 2. CSS
# =============================
st.markdown("""
<style>
    /* Background utama putih */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar background putih */
    .stSidebar {
        background-color: #F8F9FA !important;
    }
    
    /* Semua teks hitam */
    .stApp, .stSidebar, .stMarkdown, .stText, .stTitle, .stSubheader {
        color: #000000 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Label */
    .stTextInput label, .stNumberInput label, 
    .stSelectbox label, .stSlider label {
        color: #000000 !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: #FF4B4B !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        background-color: #FF6B6B !important;
        color: white !important;
        transform: scale(1.02) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #FFFFFF !important;
    }
    
    /* Metric */
    .stMetric {
        background-color: #F8F9FA !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #F8F9FA !important;
        color: #000000 !important;
    }
    
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    
    /* Divider */
    hr {
        border-color: #DDDDDD !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 3rem !important;
        color: #FF4B4B !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        font-weight: bold !important;
    }
    
    .sub-header {
        font-size: 1.5rem !important;
        color: #666666 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 2rem !important;
        border-radius: 15px !important;
        text-align: center !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #FFE5E5, #FFCCCC) !important;
        border: 3px solid #FF4B4B !important;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #E5FFE5, #CCFFCC) !important;
        border: 3px solid #00CC66 !important;
    }
    
    .prediction-title {
        font-size: 1.8rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .prediction-prob {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        margin: 0.5rem 0 !important;
    }
    
    .prediction-note {
        font-size: 1rem !important;
        color: #555555 !important;
        margin-top: 0.5rem !important;
    }
    
    /* Feature boxes */
    .feature-box {
        background-color: #F0F2F6 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 10px !important;
        margin: 0.3rem 0 !important;
        border-left: 4px solid #FF4B4B !important;
    }
    
    .feature-box strong {
        color: #1E1E1E !important;
    }
    
    .feature-box .value {
        float: right !important;
        color: #FF4B4B !important;
        font-weight: bold !important;
    }
    
    .health-warning {
        background: linear-gradient(135deg, #FEE2E2, #FECACA) !important;
        border-left: 6px solid #DC2626 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        margin: 0.8rem 0 !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15) !important;
    }
    
    .health-warning strong {
        color: #991B1B !important;
    }
    
    .health-warning ol, .health-warning ul {
        color: #1F2937 !important;
        padding-left: 1.5rem !important;
    }
    
    .health-warning li {
        margin: 0.4rem 0 !important;
        color: #1F2937 !important;
        font-size: 0.95rem !important;
    }
    
    .health-success {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0) !important;
        border-left: 6px solid #059669 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        margin: 0.8rem 0 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.15) !important;
    }
    
    .health-success strong {
        color: #065F46 !important;
    }
    
    .health-success ol, .health-success ul {
        color: #1F2937 !important;
        padding-left: 1.5rem !important;
    }
    
    .health-success li {
        margin: 0.4rem 0 !important;
        color: #1F2937 !important;
        font-size: 0.95rem !important;
    }
    
    .health-title {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center !important;
        color: #888888 !important;
        padding: 1.5rem 0 !important;
        margin-top: 2rem !important;
        border-top: 1px solid #DDDDDD !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 3. LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        feature_names = joblib.load('model/feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error("""
        ❌ Model tidak ditemukan!
        Pastikan file model ada di folder 'model/' 
        dan semua file (model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl) ada.
        """)
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_model()

# ============================================
# 4. HEADER
# ============================================
st.markdown('<h1 class="main-header">❤️ PREDIKSI RISIKO PENYAKIT GAGAL JANTUNG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi risiko penyakit gagal jantung menggunakan AI</p>', unsafe_allow_html=True)

# ============================================
# 5. SIDEBAR - INPUT FORM
# ============================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4726/4726095.png", width=100)
    st.markdown("### Input Data Pasien")
    st.markdown("---")
    
    # Input fields
    age = st.slider("Usia (Age)", 20, 100, 50)
    sex = st.selectbox("Jenis Kelamin (Sex)", ["M", "F"])
    chest_pain = st.selectbox("Tipe Nyeri Dada (Chest Pain Type)", 
                              ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Tekanan Darah Istirahat (Resting BP)", 
                                 min_value=80, max_value=250, value=120)
    cholesterol = st.number_input("Kolesterol (Cholesterol)", 
                                  min_value=0, max_value=700, value=200)
    fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl (Fasting BS)", [0, 1])
    resting_ecg = st.selectbox("Hasil EKG Istirahat (Resting ECG)", 
                               ["Normal", "ST", "LVH"])
    max_hr = st.slider("Detak Jantung Maksimum (Max HR)", 60, 220, 150)
    exercise_angina = st.selectbox("Angina saat Olahraga (Exercise Angina)", ["N", "Y"])
    oldpeak = st.slider("Depresi ST (Oldpeak)", -5.0, 10.0, 1.0, 0.1)
    st_slope = st.selectbox("Kemiringan ST (ST Slope)", ["Up", "Flat", "Down"])
    
    st.markdown("---")
    predict_button = st.button("🔍 Prediksi", use_container_width=True)

# ============================================
# 6. MAIN CONTENT
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Fitur Penting")
    
    if model is not None:
        importance = model.feature_importances_
        features = feature_names
        importance_df = pd.DataFrame({
            'Fitur': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(10), 
                     x='Importance', 
                     y='Fitur',
                     orientation='h',
                     title='Top 10 Fitur yang Paling Berpengaruh',
                     color='Importance',
                     color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model belum dimuat. Silakan upload file model.")
    
    st.subheader("📈 Statistik Dataset")
    if model is not None:
        stats_data = {
            'Jumlah sampel': 918,
            'Jumlah fitur': 11,
            'Pasien dengan penyakit': '499 (54.4%)',
            'Pasien tanpa penyakit': '419 (45.6%)'
        }
        stats_df = pd.DataFrame(list(stats_data.items()), 
                                columns=['Metrik', 'Nilai'])
        st.dataframe(stats_df, hide_index=True)

with col2:
    st.subheader("📝 Informasi Pasien")
    
    input_data = {
        'Usia': age,
        'Jenis Kelamin': sex,
        'Tipe Nyeri Dada': chest_pain,
        'Tekanan Darah': resting_bp,
        'Kolesterol': cholesterol,
        'Gula Darah Puasa': 'Ya' if fasting_bs == 1 else 'Tidak',
        'Hasil EKG': resting_ecg,
        'Detak Maksimum': max_hr,
        'Angina Olahraga': 'Ya' if exercise_angina == 'Y' else 'Tidak',
        'Depresi ST': oldpeak,
        'Kemiringan ST': st_slope
    }
    
    for key, value in input_data.items():
        st.markdown(f"""
        <div class="feature-box">
            <strong>{key}</strong>
            <span class="value">{value}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# 7. PREDIKSI
# ============================================
if predict_button:
    if model is not None:
        try:
            # Prepare input data
            input_df = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'ChestPainType': [chest_pain],
                'RestingBP': [resting_bp],
                'Cholesterol': [cholesterol],
                'FastingBS': [fasting_bs],
                'RestingECG': [resting_ecg],
                'MaxHR': [max_hr],
                'ExerciseAngina': [exercise_angina],
                'Oldpeak': [oldpeak],
                'ST_Slope': [st_slope]
            })
            
            # Encode categorical columns
            categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            for col in categorical_cols:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Scale numerical columns
            numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Ensure correct order
            input_df = input_df[feature_names]
            
            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box prediction-positive">
                    <div class="prediction-title">⚠️ Risiko Tinggi Penyakit Gagal Jantung</div>
                    <div class="prediction-prob" style="color: #FF4B4B;">Probabilitas: {probability[1]:.2%}</div>
                    <div class="prediction-note">Pasien memiliki indikasi penyakit gagal jantung. Segera konsultasikan ke dokter!</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box prediction-negative">
                    <div class="prediction-title">✅ Risiko Rendah Penyakit Gagal Jantung</div>
                    <div class="prediction-prob" style="color: #00CC66;">Probabilitas: {probability[0]:.2%}</div>
                    <div class="prediction-note">Pasien tidak memiliki indikasi penyakit gagal jantung. Tetap jaga kesehatan!</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan probabilitas dalam chart
            fig_proba = go.Figure(data=[
                go.Bar(name='Tanpa Penyakit', x=['Probabilitas'], y=[probability[0]], 
                       marker_color='#00CC66'),
                go.Bar(name='Dengan Penyakit', x=['Probabilitas'], y=[probability[1]], 
                       marker_color='#FF4B4B')
            ])
            fig_proba.update_layout(
                title='Probabilitas Prediksi',
                yaxis_title='Probabilitas',
                barmode='group',
                height=300
            )
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # ============================================
            # 7.1 SARAN KESEHATAN 
            # ============================================
            st.markdown("---")
            st.markdown("### 💡 Saran Kesehatan")
            
            if prediction == 1:
                st.markdown("""
                <div class="health-warning">
                    <div class="health-title">🔴 Rekomendasi untuk Pasien Berisiko Tinggi</div>
                    <ol>
                        <li>🏥 <strong>Segera konsultasi</strong> ke dokter spesialis jantung</li>
                        <li>🩺 Lakukan <strong>pemeriksaan lanjutan</strong> (EKG, Echo, Treadmill)</li>
                        <li>🥗 Terapkan <strong>pola makan sehat</strong> (kurangi lemak jenuh, garam)</li>
                        <li>🏃 <strong>Olahraga teratur</strong> dengan intensitas sedang</li>
                        <li>🚭 <strong>Hindari rokok</strong> dan alkohol</li>
                        <li>😴 <strong>Istirahat cukup</strong> (7-8 jam per hari)</li>
                        <li>📊 Pantau <strong>tekanan darah</strong> dan <strong>kolesterol</strong> secara rutin</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="health-success">
                    <div class="health-title">🟢 Tips Menjaga Kesehatan Jantung</div>
                    <ol>
                        <li>🥗 Konsumsi <strong>makanan bergizi</strong> seimbang</li>
                        <li>🏃 <strong>Olahraga</strong> minimal 30 menit per hari</li>
                        <li>🚫 <strong>Hindari stres</strong> berlebihan</li>
                        <li>😴 <strong>Istirahat</strong> yang cukup (7-8 jam)</li>
                        <li>🔄 Lakukan <strong>medical check-up</strong> rutin</li>
                        <li>🚭 <strong>Hindari rokok</strong> dan minuman beralkohol</li>
                        <li>📱 Pantau <strong>detak jantung</strong> dan <strong>tekanan darah</strong></li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            st.info("Pastikan semua input sudah diisi dengan benar.")
    else:
        st.error("❌ Model tidak tersedia. Silakan upload file model.")

# ============================================
# 8. FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 1rem; color: #aaa;">
        ⚠️ Prediksi ini hanya untuk tujuan edukasi, bukan untuk diagnosis medis.<br>
        Selalu konsultasikan dengan tenaga medis profesional.
    </p>
</div>
""", unsafe_allow_html=True)