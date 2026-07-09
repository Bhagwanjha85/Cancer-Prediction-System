import os
import io
import streamlit as st
import datetime
import numpy as np
from PIL import Image

import config
import utils
import body_detector
import predict
from predict import CancerPredictor
import plots

# Load Logo Image
logo_path = os.path.join(config.ASSETS_DIR, "images", "logo-main.png")
logo_img = Image.open(logo_path) if os.path.exists(logo_path) else None

# Configure page
st.set_page_config(
    page_title="Prediction Portal - RMNIHR Virology Dept.",
    page_icon=logo_img if logo_img else "https://cdn-icons-png.flaticon.com/512/2877/2877840.png",
    layout="wide"
)

# Load CSS
css_path = os.path.join(config.ASSETS_DIR, "css", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading Cancer Diagnostic Models...")
def load_cached_predictor():
    # Force reload of predict module on initialization to pick up any signature changes
    import importlib
    import predict
    importlib.reload(predict)
    return predict.CancerPredictor()

predictor = load_cached_predictor()

# Page title
st.markdown(
    """
    <div style="padding-bottom: 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">Cancer Detection Portal</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; margin-top: 0;">
            Upload skin lesion or oral cavity photographs for automated cancer screening.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for options
st.sidebar.title("Analysis Options")

# Target tissue selection defaulting to home page click
default_index = 0
if st.session_state.get("cancer_type") == "Skin":
    default_index = 1

selected_type = st.sidebar.radio(
    "Select Target Tissue & Model",
    options=["Oral Cavity", "Skin Lesion"],
    index=default_index,
    help="Select which specialized model to use for prediction."
)

st.session_state["cancer_type"] = "Oral" if selected_type == "Oral Cavity" else "Skin"
cancer_key = st.session_state["cancer_type"].lower()
model_file_name = "best_model_oral.pth" if cancer_key == "oral" else "best_model_skin.pth"



st.sidebar.info(
    f"**Specialized {st.session_state['cancer_type']} Model Active**\n\n"
    f"The portal will now accept and analyze {st.session_state['cancer_type']} images specifically using the dedicated model."
)

# Preprocessing Option
apply_denoising = st.sidebar.checkbox(
    "Apply Bilateral Image Denoising",
    value=False,
    help="Reduces background noise while preserving lesion edge boundaries for cleaner input. Off by default to match neural network training patterns."
)

skip_validation_check = st.sidebar.checkbox(
    "Bypass Pre-Validation Filters",
    value=False,
    help="Force model prediction even if the input image is flagged as non-human or incorrect tissue type by pre-validation filters."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Security Protocol:**\n"
    "- Upload limit: 10 MB\n"
    "- Valid formats: PNG, JPG, JPEG, WEBP\n"
    "- High-resolution input is downscaled automatically."
)



# File uploader tailored to the active selection
uploader_label = f"Upload a high-quality photograph of the { 'oral cavity / mouth' if cancer_key == 'oral' else 'skin lesion / mole' }"
uploader_help = f"Make sure the photograph is in focus, well-lit, and close to the { 'oral tissue.' if cancer_key == 'oral' else 'skin lesion.' }"

uploaded_file = st.file_uploader(
    uploader_label,
    type=["png", "jpg", "jpeg", "webp"],
    help=uploader_help
)

if uploaded_file is not None:
    # 1. Size Validation
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    
    if file_size_mb > config.MAX_CONTENT_LENGTH_MB:
        st.error(f"Upload rejected. File size ({file_size_mb:.2f} MB) exceeds the {config.MAX_CONTENT_LENGTH_MB} MB limit.")
    else:
        # Load and display original uploaded image
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception:
            st.error("Failed to decode the uploaded image. The file may be corrupted.")
            image = None
            
        if image is not None:
            # Pre-validate the image structure and anatomy immediately on upload
            import cv2
            img_np = np.array(image)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
            original_resized = cv2.resize(img_np, (224, 224))
            
            if not skip_validation_check:
                is_human = predictor.body_detector.is_human_body(original_resized)
                is_correct_tissue = True
                tissue_err_msg = ""
                if is_human:
                    is_correct_tissue, tissue_err_msg = predictor.body_detector.validate_tissue_type(original_resized, st.session_state["cancer_type"])
            else:
                is_human = True
                is_correct_tissue = True
                tissue_err_msg = ""
                
            if not is_human:
                st.error("Validation Failed")
                st.markdown(
                     f"""
                     <div class="result-box-cancerous">
                         <h4 style="margin: 0; font-weight: bold;">Please upload relevent image, this image is outside the content.</h4>
                         <p style="margin: 5px 0 0 0; font-size: 0.95rem;">The pre-validation model identified the input image as non-human or outside the medical domain.</p>
                     </div>
                     """,
                     unsafe_allow_html=True
                )
                st.warning("Please upload relevent image, this image is outside the content.")
            elif not is_correct_tissue:
                st.error("Validation Failed")
                st.markdown(
                     f"""
                     <div class="result-box-cancerous">
                         <h4 style="margin: 0; font-weight: bold;">{tissue_err_msg}</h4>
                     </div>
                     """,
                     unsafe_allow_html=True
                )
                st.warning("Please upload relevent image, this image is outside the content.")
            else:
                st.toast("Image uploaded successfully!")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Uploaded Image Preview")
                    st.image(image, use_container_width=True)
                    
                with col2:
                    st.subheader("Analysis Action")
                    st.write("Ready to initiate Deep Learning pipeline. This includes:")
                    st.markdown(
                        "- **Image Quality & Structure Check** (Human body check)\n"
                        "- **Noise Reduction Filter** (Optional edge-preserving blur)\n"
                        "- **Multi-model prediction evaluation**"
                    )
                    
                    predict_btn = st.button("Analyze Uploaded Image", use_container_width=True)
                    
                if predict_btn:
                    # Prediction Loop
                    with st.spinner("Executing analysis... Please wait..."):
                        # Call Predictor
                        pred_res = predictor.predict_image(
                            file_bytes, 
                            selected_detection_type=st.session_state["cancer_type"],
                            remove_noise=apply_denoising,
                            skip_validation=True
                        )
                        
                    if not pred_res["is_human"]:
                        # Rejected Image
                        st.error("Validation Failed")
                        st.markdown(
                             f"""
                             <div class="result-box-cancerous">
                                 <h4 style="margin: 0; font-weight: bold;">Please upload relevent image, this image is outside the content.</h4>
                                 <p style="margin: 5px 0 0 0; font-size: 0.95rem;">The pre-validation model identified the input image as non-human or outside the medical domain.</p>
                             </div>
                             """,
                             unsafe_allow_html=True
                        )
                        st.warning("Please upload relevent image, this image is outside the content.")
                    elif not pred_res["success"]:
                        # Prediction failed
                        st.error(f"Analysis failed: {pred_res['message']}")
                    else:
                        # Successful Analysis
                        st.success("Diagnostic analysis complete!")
                        st.toast("Analysis finished!")
                    
                    pred_class = pred_res["predicted_class"]
                    confidence = pred_res["confidence"]
                    is_cancer = pred_res["is_cancerous"]
                    probs = pred_res["probabilities"]
                    
                    # For oral predictions, always merge Oral_Ulcer probability into Oral_Normal
                    # so that the UI graph only shows Cancer and Normal classes.
                    display_class = pred_class
                    probs_for_graph = probs.copy()
                    if "Oral_Normal" in probs_for_graph or "Oral_Ulcer" in probs_for_graph:
                        probs_for_graph["Oral_Normal"] = probs_for_graph.get("Oral_Normal", 0.0) + probs_for_graph.get("Oral_Ulcer", 0.0)
                        if "Oral_Ulcer" in probs_for_graph:
                            probs_for_graph["Oral_Ulcer"] = 0.0
                            
                    if pred_class == "Oral_Ulcer":
                        display_class = "Oral Normal (Benign Ulcer)"
                    
                    detected_tissue = "Oral Cavity" if "Oral" in pred_class else "Skin Lesion"
                    
                    # Display Results Highlights
                    st.markdown("---")
                    st.subheader("Primary Classification Result")
                    
                    res_class = "result-box-cancerous" if is_cancer else "result-box-normal"
                    res_title = "MALIGNANCY SUSPECTED" if is_cancer else "BENIGN / HEALTHY STATUS DETECTED"
                    
                    st.markdown(
                         f"""
                         <div class="{res_class}">
                             <h3 style="margin: 0 0 8px 0; font-weight: bold; font-family: 'Outfit';">{res_title}</h3>
                             <p style="margin: 0; font-size: 1.15rem; line-height: 1.6;">
                                 <b>Detected Tissue:</b> {detected_tissue}<br/>
                                 <b>Predicted Diagnosis:</b> {display_class.replace('_', ' ')}<br/>
                                 <b>Confidence Score:</b> {confidence * 100:.2f}%
                             </p>
                         </div>
                         """,
                         unsafe_allow_html=True
                    )
                    
                    # Visualizations (Original, Heatmap, and Bar chart)
                    st.subheader("Visual Diagnostics & Class Probabilities")
                    
                    viz_col1, viz_col2 = st.columns([1, 1.2])
                    
                    with viz_col1:
                        # Grad-CAM Heatmap display
                        if pred_res["superimposed"] is not None:
                            st.write(" **Grad-CAM Saliency Map**")
                            st.image(
                                pred_res["superimposed"], 
                                caption="Model Focus Highlight (Red indicates maximum influence)",
                                use_container_width=True
                            )
                        else:
                            st.info("Grad-CAM visualization is unavailable for this backbone.")
                            
                    with viz_col2:
                        st.write(" **Class Confidence Distribution**")
                        # Plot Matplotlib Bar chart
                        if detected_tissue == "Oral Cavity":
                            # Always hide Oral_Ulcer from the graph, as it is considered normal/benign
                            active_classes = ["Oral_Cancer", "Oral_Normal"]
                        else:
                            active_classes = ["Skin_Cancer", "Skin_Normal"]
                            
                        prob_list = [probs_for_graph.get(c, 0.0) for c in active_classes]
                        fig = plots.plot_probability_graph(np.array(prob_list), active_classes)
                        st.pyplot(fig)
                        
                    # Clinical Details and Recommendations
                    st.markdown("---")
                    st.subheader("Clinical Details & Recommendations")
                    
                    info = pred_res["medical_info"]
                    
                    info_tabs = st.tabs([
                        "Clinical Description", 
                        "Symptoms & Risks", 
                        "Prevention & Treatment", 
                        "Clinical Consultation"
                    ])
                    
                    with info_tabs[0]:
                        st.markdown(f"#### {info.get('title', 'Disease Details')}")
                        st.write(info.get("description", "No description available."))
                        
                    with info_tabs[1]:
                        col_sym, col_risk = st.columns(2)
                        with col_sym:
                            st.write("##### Possible Symptoms")
                            for s in info.get("symptoms", []):
                                st.markdown(f"- {s}")
                            if not info.get("symptoms"):
                                st.write("No typical symptoms listed.")
                        with col_risk:
                            st.write("##### Key Risk Factors")
                            for r in info.get("risk_factors", []):
                                st.markdown(f"- {r}")
                            if not info.get("risk_factors"):
                                st.write("No critical risk factors listed.")
                                
                    with info_tabs[2]:
                        col_prev, col_treat = st.columns(2)
                        with col_prev:
                            st.write("##### Preventive Actions")
                            for p in info.get("preventive_measures", []):
                                st.markdown(f"- {p}")
                            if not info.get("preventive_measures"):
                                st.write("No preventive guidelines listed.")
                        with col_treat:
                            st.write("##### Standard Treatments")
                            for t in info.get("treatment_suggestions", []):
                                st.markdown(f"- {t}")
                            if not info.get("treatment_suggestions"):
                                st.write("Consult a specialist for options.")
                                
                    with info_tabs[3]:
                        st.write("##### Clinical Consultation Guidelines")
                        st.write(info.get("consultation_advice", "Seek immediate expert consultation."))
                        
                    # Log prediction history to CSV
                    record = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "detection_type": f"{detected_tissue} Detection",
                        "predicted_class": display_class,
                        "confidence": confidence,
                        "is_cancerous": is_cancer,
                        "device_used": utils.check_gpu_support()["active_device"],
                        "medical_info": info,
                        "disclaimer": pred_res["disclaimer"]
                    }
                    utils.append_prediction_to_csv(os.path.join(config.BASE_DIR, "prediction_history.csv"), record)
                    
                    # PDF Export Block
                    st.markdown("---")
                    st.subheader("Export & Share Results")
                    
                    col_pdf, col_hist = st.columns(2)
                    
                    with col_pdf:
                        st.write("Download a detailed clinical report of this analysis in PDF format.")
                        # Generate PDF bytes
                        pdf_bytes = utils.generate_pdf_report(record)
                        st.download_button(
                            label="Download Clinical PDF Report",
                            data=pdf_bytes,
                            file_name=f"Clinical_Detection_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    with col_hist:
                        st.write("View or download full local screening history in CSV format.")
                        if st.button("Export Diagnostic History (CSV)", use_container_width=True):
                            csv_path = os.path.join(config.BASE_DIR, "prediction_history.csv")
                            if os.path.exists(csv_path):
                                with open(csv_path, "r", encoding="utf-8") as f:
                                    csv_data = f.read()
                                st.download_button(
                                    label="Save History File",
                                    data=csv_data,
                                    file_name="diagnostic_history.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.info("No prediction history recorded yet.")
                                
                    # Clinical Disclaimer Display
                    st.markdown(
                        f"""
                        <div class="medical-disclaimer-banner">
                            <h5 style="color: #EF4444; margin: 0 0 5px 0; font-weight: bold;">MEDICAL DISCLAIMER</h5>
                            <p style="margin: 0; font-size: 0.85rem; color: #F3F4F6;">{pred_res['disclaimer']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.info("Please upload an image using the file uploader above to begin the screening analysis.")
