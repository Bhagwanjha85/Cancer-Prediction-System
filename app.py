import os
import streamlit as st
import config
import utils

# Set page config FIRST
st.set_page_config(
    page_title="OncoVision - AI Cancer Detection Portal",
    page_icon="https://cdn-icons-png.flaticon.com/512/2877/2877840.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = os.path.join(config.ASSETS_DIR, "css", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Diagnostic Sidebar Info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2877/2877840.png", width=70)
st.sidebar.title("OncoVision Portal")
st.sidebar.markdown("*Next-Gen Deep Learning Diagnostics*")

st.sidebar.markdown("---")
st.sidebar.subheader("System Diagnostics")
gpu_info = utils.check_gpu_support()
if gpu_info["gpu_available"]:
    st.sidebar.success(f"GPU Accelerated\nDevice: {gpu_info['active_device'].split(':')[-1]}")
else:
    st.sidebar.info("Running on CPU Fallback")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Quick Help:**
    Use the menu options to access:
    1. **Prediction Portal** (Upload & Analyze)
    2. **Model Performance** (Architecture comparisons)
    3. **Dataset Information** (Metrics & Splits)
    4. **Tech Stack Details** (Deep Learning concepts)
    5. **Developer Contact** (Inquiries)
    """
)

# Header Section
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0px 40px 0px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px; background: linear-gradient(135deg, #ffffff 0%, #1b285c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            OncoVision: Deep Learning Cancer Detection Portal
        </h1>
        <p style="font-size: 1.2rem; color: #94A3B8; max-width: 800px; margin: 0 auto;">
            An advanced, real-time medical screening tool leveraging transfer learning and explainable AI (Grad-CAM) to identify Oral and Skin Malignancies.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Dashboard Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">Oral Cancer Screening</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem;">
                Detects Oral Squamous Cell Carcinoma (OSCC) and other mucosal malignancies from intraoral camera captures or medical photographs.
            </p>
            <ul style="color: #94A3B8; font-size: 0.9rem; padding-left: 20px;">
                <li>Identifies leukoplakia & erythroplakia</li>
                <li>Highlights early dysplastic margins</li>
                <li>Differentiates benign aphthous ulcers</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Launch Oral Predictor", key="btn_oral"):
        st.session_state["cancer_type"] = "Oral"
        st.switch_page("pages/1_Prediction.py")

with col2:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">Skin Cancer Screening</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem;">
                Analyzes dermoscopic and clinical images of skin lesions to detect Melanoma, Basal Cell Carcinoma, and other suspicious moles.
            </p>
            <ul style="color: #94A3B8; font-size: 0.9rem; padding-left: 20px;">
                <li>ABCDE dermoscopic criteria analysis</li>
                <li>Locates irregular lesion borders</li>
                <li>Screens for malignant melanoma signs</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Launch Skin Predictor", key="btn_skin"):
        st.session_state["cancer_type"] = "Skin"
        st.switch_page("pages/1_Prediction.py")

st.markdown("---")

# Project Workflow Section
st.subheader("Comprehensive Diagnostic Workflow")
w_col1, w_col2, w_col3, w_col4 = st.columns(4)

with w_col1:
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center; min-height: 220px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1b285c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:15px;"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z"/><circle cx="12" cy="13" r="3"/></svg>
            <h4 style="margin-bottom: 8px;">1. Image Upload</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Patient uploads a close-up skin lesion or oral cavity image via a secure drag-and-drop panel.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with w_col2:
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center; min-height: 220px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1b285c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:15px;"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 15h3"/><path d="M1 9h3"/><path d="M1 15h3"/></svg>
            <h4 style="margin-bottom: 8px;">2. Human Validation</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                MobileNetV2 filters out irrelevant objects (animals, buildings, items) before classification to prevent false positives.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with w_col3:
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center; min-height: 220px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1b285c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:15px;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3.001 3.001 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3.001 3.001 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/></svg>
            <h4 style="margin-bottom: 8px;">3. Deep CNN Inference</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                The best-performing model runs inference across 4 classes to determine disease presence and type.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with w_col4:
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center; min-height: 220px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1b285c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:15px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            <h4 style="margin-bottom: 8px;">4. Explainable AI</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Grad-CAM highlights the exact features guiding predictions, providing visual trust for medical evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Medical Disclaimer Alert Box
st.markdown(
    """
    <div class="medical-disclaimer-banner">
        <h5 style="color: #1b285c; margin: 0 0 5px 0; font-weight: bold; font-family: 'Outfit';">CRITICAL CLINICAL DISCLAIMER</h5>
        <p style="margin: 0; font-size: 0.85rem; color: #F3F4F6; line-height: 1.4;">
            This system is designed exclusively for educational, research, and self-screening awareness purposes. 
            It is NOT a medical diagnostic platform. All machine learning predictions must be reviewed and 
            confirmed by a licensed dermatologist, oncologist, or qualified clinician.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
