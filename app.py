import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import io

# --- 1. CORE BACKEND INTEGRATION ---
try:
    from anomalydetector import load_trained_model, detect_anomaly, get_anomaly_threshold
except ImportError:
    st.error("Neural Backend 'anomalydetector.py' not found.")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CAE Maritime Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 3. ENHANCED VISUAL CSS (FIXED & CLEANED) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }

    /* Heading Styling */
    .header-style {
        color: #002b5b;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .sub-header-style {
        color: #007bff;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }

    /* Analytics Container */
    .analytics-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 40px;
    }

    /* Professional Solid Color Cards */
    .stat-card {
        flex: 1;
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }

    .blue-solid { background-color: #002b5b; } /* Deep Navy */
    .teal-solid { background-color: #00695c; } /* Industrial Teal */
    .slate-solid { background-color: #37474f; } /* Dark Slate */
    
    .card-label { font-size: 0.75rem; font-weight: 600; opacity: 0.8; text-transform: uppercase; margin-bottom: 5px; }
    .card-value { font-size: 1.8rem; font-weight: 800; }

    /* Sidebar Logic Cards */
    .logic-card {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #002b5b;
        margin-bottom: 15px;
        font-size: 0.85rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #333;
    }
    .step-num {
        color: #002b5b;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    model = load_trained_model()
    threshold = get_anomaly_threshold()
    return model, threshold

model, base_threshold = load_resources()

# --- 5. SIDEBAR: HOW IT WORKS ---
with st.sidebar:
    st.markdown("<h2 style='color: #002b5b; font-size:1.4rem;'>🛰️ CONTROL PANEL</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.write("#### ⚙️ ADJUST SENSITIVITY")
    sensitivity = st.slider("Multiplier", 0.5, 2.0, 1.0, 0.05)
    
    st.markdown("---")
    st.write("#### 🧠 HOW IT WORKS")
    
    st.markdown("""
    <div class="logic-card">
        <div class="step-num">STEP 1: COMPRESSION</div>
        The Encoder scans the 80x80 patch, stripping noise to create a 128-dim 'Neural Fingerprint' of the sea.
    </div>
    <div class="logic-card">
        <div class="step-num">STEP 2: REBUILDING</div>
        The Decoder attempts to 'repaint' the image using only its learned knowledge of water textures.
    </div>
    <div class="logic-card">
        <div class="step-num">STEP 3: COMPARISON</div>
        We calculate <b>MSE</b>. If the AI cannot rebuild a shape (like a ship), a high error signal is triggered.
    </div>
    <div class="logic-card">
        <div class="step-num">STEP 4: DETECTION</div>
        Pixels exceeding the 95th percentile threshold are flagged in RED as anomalies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if model:
        st.success("ENGINE: ONLINE")
    else:
        st.error("ENGINE: OFFLINE")

# --- 6. MAIN DASHBOARD ---
st.markdown("<h1 class='header-style'>MARITIME ANOMALY INTELLIGENCE</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header-style'>Convolutional Autoencoder (CAE) Detection Pipeline</p>", unsafe_allow_html=True)

# Calculate active threshold
adj_threshold = (base_threshold if base_threshold else 0.001246) * sensitivity

# Dashboard Cards with Solid Colors
st.markdown(f"""
    <div class="analytics-container">
        <div class="stat-card blue-solid">
            <div class="card-label">SYSTEM STATUS</div>
            <div class="card-value">OPERATIONAL</div>
        </div>
        <div class="stat-card teal-solid">
            <div class="card-label">NEURAL THRESHOLD</div>
            <div class="card-value">{adj_threshold:.6f}</div>
        </div>
        <div class="stat-card slate-solid">
            <div class="card-label">ANALYSIS MODE</div>
            <div class="card-value">UNSUPERVISED</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 7. DATA UPLOAD ---
st.markdown("### 📥 DATA INGESTION")
col_up, col_prev = st.columns([1.5, 1], gap="large")

with col_up:
    uploaded_file = st.file_uploader("Upload Satellite Image (80x80 RGB Patch)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        if st.button("▶️ EXECUTE CONVOLUTIONAL SCAN", use_container_width=True):
            run_scan = True
        else:
            run_scan = False
    else:
        run_scan = False

with col_prev:
    if uploaded_file:
        st.image(uploaded_file, caption="Input Data Preview", use_container_width=True)

# --- 8. RESULTS DISPLAY ---
if uploaded_file and run_scan:
    with st.spinner('Neural Network modeling oceanic normalcy...'):
        original, reconstructed, error_map, mask = detect_anomaly(uploaded_file, model, adj_threshold)
        
        if original is not None:
            st.markdown("---")
            st.markdown("### 🔍 NEURAL INSPECTION REPORT")
            
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.markdown("**RECONSTRUCTION**")
                st.image(reconstructed, use_container_width=True)
                st.caption("AI-generated sea normalcy.")
            
            with r2:
                st.markdown("**ERROR HEATMAP**")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(error_map, cmap='hot')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Reconstruction failure zones.")

            with r3:
                st.markdown("**LOCALIZATION**")
                overlay = (original * 255).astype(np.uint8)
                overlay[mask > 0] = [255, 0, 0] 
                st.image(overlay, use_container_width=True)
                
                score = (np.sum(mask > 0) / mask.size) * 100
                if score > 0:
                    st.error(f"🚨 TARGET IDENTIFIED: {score:.2f}%")
                else:
                    st.success("✅ SCAN STATUS: CLEAR")

            # --- 9. EXPORT & DOWNLOAD ---
            st.markdown("---")
            st.markdown("#### 💾 INTELLIGENCE EXPORT")
            
            export_buf = io.BytesIO()
            plt.imsave(export_buf, error_map, cmap='hot', format='png')
            
            st.download_button(
                label="📥 DOWNLOAD ANOMALY RADIOGRAPH (PNG)",
                data=export_buf.getvalue(),
                file_name=f"maritime_scan_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )

# Footer
st.markdown("<br><hr><center>GITAM School of Technology | B.Tech Capstone Project 2026</center>", unsafe_allow_html=True)