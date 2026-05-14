"""
Streamlit web application for LULC Classification inference.

What: Interactive web dashboard for uploading satellite images and running inference
       with dual visualization (original + false color composite).
Why: Enables non-technical users to interact with models, visualizes predictions effectively.

Usage:
    streamlit run app.py
"""
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import rasterio
import cv2
from pathlib import Path
import tempfile

import config
from model import ResNet50Classifier


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LULC Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State & Caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(modality):
    """
    Load trained model checkpoint.

    What: Caches model in memory to avoid reloading on every interaction.
    Why: Improves app responsiveness, reduces memory usage.
    """
    try:
        input_channels = 3 if modality == 'RGB' else 13
        model = ResNet50Classifier(
            input_channels=input_channels,
            num_classes=config.NUM_CLASSES,
            freeze_backbone=False
        )

        model_path = config.MODELS_DIR / f'{modality.lower()}_best.pth'
        if not model_path.exists():
            st.error(f"❌ Model not found: {model_path}\n\nPlease run `python train.py` first.")
            return None

        state_dict = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(config.DEVICE)

        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def load_rgb_image(uploaded_file):
    """Load RGB JPG image."""
    try:
        img = Image.open(uploaded_file).convert('RGB')
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load RGB image: {e}")


def load_ms_image(uploaded_file):
    """Load MS TIFF image (13 bands)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            arr = src.read().astype(np.float32)  # (13, H, W)

        Path(tmp_path).unlink()  # Clean up temp file
        return arr
    except Exception as e:
        raise ValueError(f"Failed to load MS TIFF image: {e}")


def get_rgb_composite(ms_array):
    """Extract RGB composite (B4-B3-B2) from 13-band MS array."""
    # Band indices: B04(Red)=3, B03(Green)=2, B02(Blue)=1
    rgb = np.stack([ms_array[3], ms_array[2], ms_array[1]], axis=-1)
    # Normalize to 0-1 for display
    vmin, vmax = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return rgb


def get_false_color_composite(ms_array):
    """
    Extract False Color Composite (NIR-Red-Green for vegetation highlighting).

    What: Creates B8-B4-B3 representation, emphasizes healthy vegetation (appears red).
    Why: Helps visualize land cover differences, especially vegetation vs. non-vegetation.
    """
    # Band indices: B08(NIR)=7, B04(Red)=3, B03(Green)=2
    nrg = np.stack([ms_array[7], ms_array[3], ms_array[2]], axis=-1)
    # Normalize to 0-1
    vmin, vmax = np.percentile(nrg, [2, 98])
    nrg = np.clip((nrg - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return nrg


def preprocess_image(img_array, modality):
    """Preprocess image to model input format."""
    from torchvision import transforms

    # Resize and normalize
    if modality == 'RGB':
        mean, std = config.RGB_MEAN, config.RGB_STD
        # PIL Image required for ToTensor
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8)) if img_array.max() <= 1 else Image.fromarray(img_array.astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img_tensor = transform(img_pil)
    else:  # MS
        mean, std = config.MS_MEAN, config.MS_STD
        # img_array is already (13, H, W)
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.Normalize(mean=mean, std=std)
        ])
        img_tensor = torch.from_numpy(img_array)
        img_tensor = transform(img_tensor)

    return img_tensor.unsqueeze(0).to(config.DEVICE)  # Add batch dimension


def run_inference(_img_tensor, _model):
    """Run model inference on image tensor."""
    with torch.no_grad():
        logits = _model(_img_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)

    return {
        'top_classes': [config.CLASS_NAMES[idx.item()] for idx in top_indices[0]],
        'top_probs': top_probs[0].cpu().numpy(),
        'all_probs': probs[0].cpu().numpy()
    }


def main():
    """Main Streamlit app."""
    st.title("🛰️ Land Use & Land Cover Classification")
    st.markdown("Upload a satellite image and get instant classification with visual analysis.")

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar: Model Selection & Info
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        modality = st.radio(
            "Select Model Type",
            options=['RGB', 'MS'],
            help="RGB: 3-channel optical images | MS: 13-channel multispectral"
        )

        st.markdown("---")
        st.subheader("📊 Dataset Info")
        st.write(f"**Classes:** {len(config.CLASS_NAMES)}")
        st.write(f"**Image Size:** {config.IMAGE_SIZE} × {config.IMAGE_SIZE} px")
        st.write(f"**Modality:** {modality}")
        if modality == 'RGB':
            st.write(f"**Channels:** 3 (Red, Green, Blue)")
        else:
            st.write(f"**Channels:** 13 (Sentinel-2 bands)")

        st.markdown("---")
        st.subheader("📌 Classes")
        cols = st.columns(2)
        for i, cls in enumerate(config.CLASS_NAMES):
            with cols[i % 2]:
                st.write(f"• {cls}")

    # ─────────────────────────────────────────────────────────────────────────
    # File Upload
    # ─────────────────────────────────────────────────────────────────────────
    st.header("📁 Upload Image")
    file_ext = "jpg, jpeg" if modality == 'RGB' else "tif, tiff"
    uploaded_file = st.file_uploader(
        f"Upload a {modality} image ({file_ext})",
        type=['jpg', 'jpeg', 'tif', 'tiff'],
        help=f"Supported formats: {file_ext}"
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Load and Process Image
    # ─────────────────────────────────────────────────────────────────────────
    try:
        with st.spinner(f"Loading {modality} image..."):
            if modality == 'RGB':
                img_array = load_rgb_image(uploaded_file)
                display_rgb = img_array / 255.0 if img_array.max() > 1 else img_array
            else:  # MS
                img_array = load_ms_image(uploaded_file)
                display_rgb = get_rgb_composite(img_array)

        st.success("✅ Image loaded successfully")

    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Load Model & Run Inference
    # ─────────────────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {modality} model..."):
        model = load_model(modality)
    if model is None:
        return

    with st.spinner("Running inference..."):
        img_tensor = preprocess_image(img_array, modality)
        results = run_inference(img_tensor, model)

    st.success("✅ Inference complete")

    # ─────────────────────────────────────────────────────────────────────────
    # Dual Dashboard: Visualization + Predictions
    # ─────────────────────────────────────────────────────────────────────────
    st.header("📊 Classification Results")

    col1, col2 = st.columns(2)

    # Left column: Image visualization with toggle
    with col1:
        st.subheader("🖼️ Satellite Image")

        if modality == 'MS':
            view_mode = st.radio("Display Mode", options=['RGB Composite', 'False Color (NIR-R-G)'],
                                help="False Color highlights vegetation health in red")
            if view_mode == 'False Color (NIR-R-G)':
                display_rgb = get_false_color_composite(img_array)
        else:
            st.text("RGB Natural Color")

        st.image(display_rgb, caption="Visualized Patch", width="stretch")

    # Right column: Predictions
    with col2:
        st.subheader("🎯 Top 3 Predictions")

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 4))

        # --- DARK MODE FIXES START ---
        fig.patch.set_alpha(0.0)      # Transparent outer background
        ax.set_facecolor((0,0,0,0))   # Transparent inner background
        
        # Set text and axis colors to be visible in Dark Mode
        label_color = "#808080" 
        plt.rcParams.update({'text.color': label_color, 'axes.labelcolor': label_color})
        ax.tick_params(colors=label_color, which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor(label_color)
        # --- DARK MODE FIXES END ---

        # Bar chart with probabilities
        bars = ax.barh(results['top_classes'], results['top_probs'] * 100)

        # Color bars based on confidence
        for i, bar in enumerate(bars):
            prob = results['top_probs'][i]
            if prob > 0.6:
                bar.set_color('#2ecc71')  # Green (high confidence)
            elif prob > 0.3:
                bar.set_color('#f39c12')  # Orange (medium confidence)
            else:
                bar.set_color('#e74c3c')  # Red (low confidence)

        ax.set_xlabel('Probability (%)', color=label_color)
        ax.set_title('Classification Confidence', color=label_color)
        ax.set_xlim(0, 100)

        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, results['top_probs'])):
            ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                   f'{prob * 100:.1f}%', va='center', fontsize=10, fontweight='bold', color=label_color)

        plt.tight_layout()
        # Important: set transparent=True here
        st.pyplot(fig, width="stretch", transparent=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Detailed Metrics
    # ─────────────────────────────────────────────────────────────────────────
    st.header("📈 Detailed Analysis")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Top Prediction", results['top_classes'][0], f"{results['top_probs'][0]*100:.1f}%")
    with metric_cols[1]:
        st.metric("2nd Place", results['top_classes'][1], f"{results['top_probs'][1]*100:.1f}%")
    with metric_cols[2]:
        st.metric("3rd Place", results['top_classes'][2], f"{results['top_probs'][2]*100:.1f}%")

    st.subheader("All Class Probabilities")
    df_results = {
        'Class': config.CLASS_NAMES,
        'Probability (%)': [f"{p*100:.2f}%" for p in results['all_probs']]
    }
    st.dataframe(df_results, width="stretch")
    # st.header("📊 Classification Results")

    # col1, col2 = st.columns(2)

    # # Left column: Image visualization with toggle
    # with col1:
    #     st.subheader("🖼️ Satellite Image")

    #     if modality == 'MS':
    #         view_mode = st.radio("Display Mode", options=['RGB Composite', 'False Color (NIR-R-G)'],
    #                             help="False Color highlights vegetation health in red")
    #         if view_mode == 'False Color (NIR-R-G)':
    #             display_rgb = get_false_color_composite(img_array)
    #     else:
    #         st.text("RGB Natural Color")

    #     st.image(display_rgb, caption="Visualized Patch", width="stretch")

    # # Right column: Predictions
    # with col2:
    #     st.subheader("🎯 Top 3 Predictions")

    #     # Bar chart with probabilities
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     bars = ax.barh(results['top_classes'], results['top_probs'] * 100)

    #     # Color bars based on confidence
    #     for i, bar in enumerate(bars):
    #         prob = results['top_probs'][i]
    #         if prob > 0.6:
    #             bar.set_color('#2ecc71')  # Green (high confidence)
    #         elif prob > 0.3:
    #             bar.set_color('#f39c12')  # Orange (medium confidence)
    #         else:
    #             bar.set_color('#e74c3c')  # Red (low confidence)

    #     ax.set_xlabel('Probability (%)')
    #     ax.set_title('Classification Confidence')
    #     ax.set_xlim(0, 100)

    #     # Add percentage labels on bars
    #     for i, (bar, prob) in enumerate(zip(bars, results['top_probs'])):
    #         ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
    #                f'{prob * 100:.1f}%', va='center', fontsize=10, fontweight='bold')

    #     plt.tight_layout()
    #     st.pyplot(fig, width="stretch")

    # # ─────────────────────────────────────────────────────────────────────────
    # # Detailed Metrics
    # # ─────────────────────────────────────────────────────────────────────────
    # st.header("📈 Detailed Analysis")

    # metric_cols = st.columns(3)
    # with metric_cols[0]:
    #     st.metric("Top Prediction", results['top_classes'][0], f"{results['top_probs'][0]*100:.1f}%")
    # with metric_cols[1]:
    #     st.metric("2nd Place", results['top_classes'][1], f"{results['top_probs'][1]*100:.1f}%")
    # with metric_cols[2]:
    #     st.metric("3rd Place", results['top_classes'][2], f"{results['top_probs'][2]*100:.1f}%")

    # # All class probabilities table
    # st.subheader("All Class Probabilities")
    # df_results = {
    #     'Class': config.CLASS_NAMES,
    #     'Probability (%)': [f"{p*100:.2f}%" for p in results['all_probs']]
    # }
    # st.dataframe(df_results, width="stretch")
    # ─────────────────────────────────────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
                <style>
    /* Force metric cards to have a transparent background and dynamic text color */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Ensure the label and value are readable in both modes */
    [data-testid="stMetricLabel"] {
        font-weight: bold;
    }
    </style>
        <div style='
            text-align: center; 
            padding: 20px; 
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            margin-top: 30px;'>
            <p style='color: #808080; font-size: 14px; margin-bottom: 5px;'>
                🛰️ <b>LULC Classification System</b> | Powered by ResNet50 & Sentinel-2
            </p>
            <p style='color: #808080; font-size: 12px;'>
                DEPI Graduation Project
            </p>
        </div>
    """, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
