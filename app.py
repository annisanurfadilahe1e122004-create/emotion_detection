import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import cv2
import time
import requests
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# -------------------------
# Configuration & Default Values
# -------------------------
# PERBAIKAN KRUSIAL: Mengubah link sharing menjadi link direct download Google Drive
# File ID: 1nfiNR6XWVPzpNYHTXnL4lMR3PjnGPSx1
MODEL_URL = "https://huggingface.co/datasets/sdadwdas/ed/resolve/main/facial_emotion_recognition_model.h5"
MODEL_DEFAULT_PATH = "facial_emotion_recognition_model.h5"
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
TARGET_FPS = 30

EMOTION_EMOJI = {
    'Anger': '😠', 'Contempt': '😒', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprised': '😲'
}

# Default Parameters for optimized Real-time Detection
DEFAULT_CAM_INDEX = 0
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 8  # Ditingkatkan untuk robust
DEFAULT_MIN_FACE_SIZE = 80
DEFAULT_PADDING_PERCENT = 40 # Padding Wajah 40%

# -------------------------
# Page Config - Modern Design
# -------------------------
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .emotion-result {
        font-size: 3.5rem;
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 20px;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box_shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    .stProgress > div > div {
        border-radius: 10px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Model Architecture
# -------------------------
def build_fer_model(input_shape=(224,224,3), n_classes=8):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

@st.cache_resource
def load_emotion_model(url: str, filename: str):
    try:
        if not os.path.exists(filename):
            st.info(f"Mengunduh model dari cloud. Ini mungkin butuh waktu beberapa menit.")
            
            # Melakukan permintaan unduhan
            response = requests.get(url, stream=True)
            response.raise_for_status() # Cek jika unduhan berhasil
            
            # Menyimpan file yang diunduh secara lokal
            with open(filename, "wb") as f:
                f.write(response.content)
            st.success("✅ Model berhasil diunduh!")

        # Setelah diunduh (atau jika sudah ada), model dimuat
        try:
            # Karena model dilatih dengan arsitektur yang dikustomisasi, kita perlu membangun arsitektur dulu
            if "ResNet50" in filename: 
                 model = build_fer_model()
                 model.load_weights(filename)
            else:
                 model = load_model(filename)

            return model, None
        
        except Exception as e:
            return None, f"Gagal memuat model Keras: {str(e)}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Gagal mengunduh model. Pastikan **MODEL_URL** menggunakan format direct download Google Drive yang benar (gunakan format 'uc?export=download...'). Error: {str(e)}"
    except Exception as e:
        return None, str(e)

# -------------------------
# Face Detection
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image, scale_factor, min_neighbors, min_size):
    """Detect faces with optimized parameters"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

# -------------------------
# CRITICAL PREPROCESSING (MATCHING TRAINING)
# -------------------------
def preprocess_face(face_bgr, size=(224, 224)):
    """
    Menggunakan BGR [0, 255] non-normalized karena model dilatih dengan input ini.
    """
    # Resize dengan INTER_LINEAR (sama dengan training)
    face_resized = cv2.resize(face_bgr, size, interpolation=cv2.INTER_LINEAR)
    
    # PENTING: TIDAK mengubah BGR ke RGB dan TIDAK ada normalisasi
    face_array = face_resized.astype(np.float32)
    face_batch = np.expand_dims(face_array, axis=0)
    
    return face_batch

# -------------------------
# Sidebar - Streamlined Settings
# -------------------------
with st.sidebar: 	
    st.markdown("### ⚙️ Pengaturan Default")
    
    st.markdown("#### ⚙️ Deteksi")
    st.caption(f"**Scale Factor:** {DEFAULT_SCALE_FACTOR}")
    st.caption(f"**Min Neighbors:** {DEFAULT_MIN_NEIGHBORS}")
    st.caption(f"**Padding:** {DEFAULT_PADDING_PERCENT}%")
    
    st.markdown("---")
    st.markdown("### 📊 Info Model")
    
    st.markdown("**Computer Vision - Kelompok 6**") 
    
    st.caption(f"**Model:** ResNet50 (Frozen Base)")
    st.caption(f"**Preprocessing:** BGR [0-255] (Matching Training)")
    st.caption(f"**Target FPS:** {TARGET_FPS}") 

# -------------------------
# Main Header
# -------------------------
st.markdown('<h1 class="main-header">😊 Face Emotion Recognition</h1>', unsafe_allow_html=True)

# Load Model
with st.spinner("🔄 Memuat model..."):
    model, load_err = load_emotion_model(MODEL_URL, MODEL_DEFAULT_PATH)

if model is None:
    st.error(f"❌ {load_err}")
    st.info("💡 Pastikan file model ada di direktori yang sama dengan script ini")
    st.stop()

# -------------------------
# State Management
# -------------------------
if "running" not in st.session_state:
    st.session_state.running = False

# -------------------------
# Controls
# -------------------------
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ Mulai Deteksi", type="primary"):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Berhenti"):
        st.session_state.running = False

st.markdown("---")

# -------------------------
# Prediction Function
# -------------------------
def predict_emotion(frame_bgr, show_box=True):
    """Predict emotion from frame using fixed default parameters."""
    
    # Gunakan nilai default yang sudah dioptimalkan
    scale_factor = DEFAULT_SCALE_FACTOR
    min_neighbors = DEFAULT_MIN_NEIGHBORS
    min_face_size = DEFAULT_MIN_FACE_SIZE
    padding_percent = DEFAULT_PADDING_PERCENT
    
    # 1. Detect Faces
    faces = detect_faces(frame_bgr, scale_factor, min_neighbors, min_face_size)
    
    # Set default/fallback ke Neutral (sesuai permintaan user)
    result = {
        'frame': frame_bgr,
        'emotion': 'Neutral', 
        'probabilities': None,
        'face_count': len(faces),
        'emoji': EMOTION_EMOJI['Neutral'], 
        'confidence': 0.0
    }
    
    if len(faces) > 0:
        # Ambil wajah terbesar (paling jelas)
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        x_orig, y_orig, w_orig, h_orig = faces[largest_idx]
        
        
        # --- Terapkan PADDING (MARGIN) ---
        padding_ratio = padding_percent / 100.0
        px = int(w_orig * padding_ratio / 2)
        py = int(h_orig * padding_ratio / 2)
        
        # Hitung koordinat crop baru dengan padding, pastikan tidak keluar batas frame
        x1 = max(0, x_orig - px)
        y1 = max(0, y_orig - py)
        x2 = min(frame_bgr.shape[1], x_orig + w_orig + px)
        y2 = min(frame_bgr.shape[0], y_orig + h_orig + py)
        
        # Ambil crop wajah yang sudah diperbesar (padded)
        face_roi = frame_bgr[y1:y2, x1:x2]
        
        # 2. Preprocess (BGR [0-255])
        face_input = preprocess_face(face_roi, size=(224, 224))
        
        # 3. Predict
        predictions = model.predict(face_input, verbose=0)[0]
        
        # 4. Get results
        emotion_idx = int(np.argmax(predictions))
        emotion_label = EMOTIONS[emotion_idx]
        confidence = float(predictions[emotion_idx])
        
        result['emotion'] = emotion_label
        result['emoji'] = EMOTION_EMOJI[emotion_label]
        result['confidence'] = confidence
        result['probabilities'] = {
            EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))
        }
        
        # 5. Draw bounding box (gunakan koordinat asli untuk box yang akurat)
        if show_box:
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.5:
                color = (0, 200, 255)  # Orange - medium confidence
            else:
                color = (0, 100, 255)  # Red - low confidence
            
            cv2.rectangle(frame_bgr, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 3)
            
            # Label with background
            label_text = f"{emotion_label} ({confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame_bgr, (x_orig, y_orig-45), (x_orig+text_w+15, y_orig), color, -1)
            cv2.putText(frame_bgr, label_text, (x_orig+7, y_orig-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    result['frame'] = frame_bgr
    return result

# -------------------------
# Main Display - Hanya Kamera Real-time
# -------------------------
# Real-time Camera Mode
video_col, info_col = st.columns([2.5, 1])

with video_col:
    video_placeholder = st.empty()

with info_col:
    emotion_display = st.empty()
    metrics_placeholder = st.empty()
    confidence_placeholder = st.empty()

# Tambahkan checkbox Tampilkan kotak deteksi di luar sidebar (dekat tombol control)
show_bbox = st.checkbox("Tampilkan kotak deteksi", value=True)

if st.session_state.running:
    st.success(f"✅ Model siap! Mengaktifkan Kamera Index {DEFAULT_CAM_INDEX}.")
    
    # Gunakan index kamera default
    cap = cv2.VideoCapture(DEFAULT_CAM_INDEX)
    
    if not cap.isOpened():
        st.error("❌ Tidak dapat membuka kamera!")
        st.info("💡 Tips: Cek izin kamera atau coba index kamera lain")
        st.session_state.running = False
    else:
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        frame_count = 0
        start_time = time.time()
        prev_time = 0
        
        try:
            while st.session_state.running:
                # FPS control
                time_elapsed = time.time() - prev_time
                if time_elapsed < 1.0 / TARGET_FPS:
                    time.sleep(max(0, (1.0 / TARGET_FPS) - time_elapsed))
                prev_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Gagal membaca frame dari kamera")
                    break
                
                # Predict emotion (menggunakan default yang sudah disetel)
                result = predict_emotion(frame.copy(), show_box=show_bbox)
                
                # Display video
                frame_rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Display emotion with animation
                emotion_display.markdown(
                    f'<div class="emotion-result">'
                    f'{result["emoji"]}<br>'
                    f'<strong>{result["emotion"]}</strong><br>'
                    f'<small style="font-size:1.2rem; color:#666;">Confidence: {result["confidence"]:.1%}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Display metrics
                with metrics_placeholder.container():
                    m1, m2 = st.columns(2)
                    m1.metric("⚡ FPS", f"{fps:.1f}")
                    m2.metric("👤 Wajah", result['face_count'])
                
                # Display confidence bars
                if result['probabilities']:
                    with confidence_placeholder.container():
                        st.markdown("#### 📊 Probabilitas Emosi")
                        
                        # Sort and show top 5
                        sorted_probs = sorted(
                            result['probabilities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        
                        for emotion, prob in sorted_probs:
                            emoji = EMOTION_EMOJI[emotion]
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(prob, text=f"{emoji} {emotion}")
                            with col2:
                                st.markdown(f"**{prob:.1%}**")
                
        finally:
            cap.release()
            st.session_state.running = False
            st.success("✅ Kamera berhasil dihentikan")
            
else:
    video_placeholder.info("👆 Klik tombol **Mulai Deteksi** untuk mengaktifkan kamera")
    
    # Tampilkan status Neutral sebagai default saat aplikasi dimuat/dihentikan
    emotion_display.markdown(
        f'<div class="emotion-result">'
        f'{EMOTION_EMOJI["Neutral"]}<br>'
        f'<strong>Neutral</strong><br>'
        f'<small style="font-size:1.2rem; color:#666;">Confidence: 0.0%</small>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    metrics_placeholder.empty()
    confidence_placeholder.empty()
    st.success("✅ Model berhasil dimuat dan siap digunakan!")


# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 2rem;'>"
    "🤖 Face Emotion Recognition System • Powered by ResNet50 • 2025<br>"
    "<small>Deteksi 8 emosi: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised</small>"
    "</div>",
    unsafe_allow_html=True
)
