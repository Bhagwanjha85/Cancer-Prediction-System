import os
import streamlit as st
from PIL import Image
import config

# Load Logo Image
logo_path = os.path.join(config.ASSETS_DIR, "images", "logo-main.png")
logo_img = Image.open(logo_path) if os.path.exists(logo_path) else None

# Configure page
st.set_page_config(
    page_title="About - RMRIMS Virology Dept.",
    page_icon=logo_img if logo_img else "https://cdn-icons-png.flaticon.com/512/2877/2877840.png",
    layout="wide"
)

# Load CSS
css_path = os.path.join(config.ASSETS_DIR, "css", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding-bottom: 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">Project Architecture & Methodology</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; margin-top: 0;">
            Scientific insights into Transfer Learning, computer vision pipelines, and system architectures.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">Deep Learning in Medical Imaging</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6;">
                Deep Convolutional Neural Networks (CNNs) have revolutionized computerized diagnostics. 
                By utilizing layered convolutions, pooling, and feature extraction layers, CNNs learn 
                hierarchical representations of image details. In dermatological and oral pathology:
            </p>
            <ul style="color: #94A3B8; font-size: 0.9rem; padding-left: 20px; line-height: 1.6;">
                <li><b>Early Layers:</b> Recognize edges, contrast boundaries, and basic shapes.</li>
                <li><b>Intermediate Layers:</b> Recognize texture alignments, pigments, and lesion margins.</li>
                <li><b>Deep Layers:</b> Capture complex pathological indicators (asymmetry, nodular formations).</li>
            </ul>
        </div>
        
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">The Power of Transfer Learning</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6;">
                Medical datasets are frequently limited in size, making training models from scratch highly susceptible to overfitting. 
                <b>Transfer Learning</b> resolves this by utilizing models pre-trained on massive open-source visual datasets (like ImageNet).
            </p>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                By freezing these pre-trained feature extractors (e.g., DenseNet121, ResNet50) and attaching custom, regularized classification 
                heads, the network applies general visual features (shapes, borders, colors) directly to malignant cellular tissues. 
                This leads to high diagnostic accuracy even with limited sample counts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">Explainable AI (Grad-CAM)</h3>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6;">
                Deep Learning models are often criticized as "black boxes" because their decision-making logic is hidden. 
                To ensure clinical utility, RMRIMS Virology Dept. implements <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b>.
            </p>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                Grad-CAM tracks the gradient flow of the winning class backward to the final convolutional layer. It overlays a 
                heat map highlighting the regions that most strongly influenced the final classification. This helps doctors confirm 
                if the AI focused on the actual lesion or irrelevant background artifacts.
            </p>
        </div>
        
        <div class="glass-card">
            <h3 style="color: #1b285c; margin-top:0;">Portal Technology Stack</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #E2E8F0;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; font-weight: bold; color: #1b285c;">Frontend Framework</td>
                    <td style="padding: 8px 0;">Streamlit Dashboard</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; font-weight: bold; color: #1b285c;">Deep Learning Engine</td>
                    <td style="padding: 8px 0;">PyTorch & Torchvision</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; font-weight: bold; color: #1b285c;">Computer Vision API</td>
                    <td style="padding: 8px 0;">OpenCV (CV2)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; font-weight: bold; color: #1b285c;">Report Compiler</td>
                    <td style="padding: 8px 0;">ReportLab PDF Generator</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight: bold; color: #1b285c;">Pre-validation model</td>
                    <td style="padding: 8px 0;">MobileNetV2 (ImageNet)</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("Complete Diagnostics Pipeline Flowchart")
st.markdown(
    """
    ```mermaid
    graph TD
        A[User Uploads Image] --> B[File size and extension check]
        B --> C[Bilateral Denoising Filter]
        C --> D[MobileNetV2 Validation]
        D -- Non-human Image --> E[Reject & Request Body Image]
        D -- Valid Human Image --> F[Best-Model Inference]
        F --> G[Extract Class Softmax Scores]
        F --> H[Grad-CAM Gradients Computation]
        G --> I[Format Medical Guidelines & Advice]
        H --> J[Overlay Heatmap on Image]
        I --> K[Display Prediction Dashboard]
        J --> K
        K --> L[Generate clinical PDF / CSV reports]
    ```
    """,
    unsafe_allow_html=True
)
