
"""
Streamlit web application for LULC Classification inference.
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
st.set_page_config(
    page_title="LULC Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stMetric {
    background-color: white; padding: 15px;
    border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
 
# ── Color map for each class ───────────────────────────────────────────────────
CLASS_COLORS = {
    "AnnualCrop":           (255, 230, 100),
    "Forest":               (34,  139,  34),
    "HerbaceousVegetation": (144, 238, 144),
    "Highway":              (169, 169, 169),
    "Industrial":           (178,  34,  34),
    "Pasture":              (255, 165,   0),
    "PermanentCrop":        (210, 180, 140),
    "Residential":          (100, 149, 237),
    "River":                (30,  144, 255),
    "SeaLake":              (0,   105, 148),
}
 
def apply_color_overlay(display_rgb, class_name, alpha=0.4):
    """Blend a solid color over the image to indicate predicted class."""
    img_uint8 = (np.clip(display_rgb, 0, 1) * 255).astype(np.uint8)
    r, g, b = CLASS_COLORS.get(class_name, (128, 128, 128))
    color_layer = np.full_like(img_uint8, [r, g, b], dtype=np.uint8)
    blended = ((1 - alpha) * img_uint8 + alpha * color_layer).astype(np.uint8)
    return Image.fromarray(blended)
 
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(modality):
    try:
        input_channels = 3 if modality == 'RGB' else 13
        model = ResNet50Classifier(
            input_channels=input_channels,
            num_classes=config.NUM_CLASSES,
            freeze_backbone=False
        )
        model_path = config.MODELS_DIR / f'resnet50_{modality.lower()}.pth'
        if not model_path.exists():
            st.error(f"❌ Model not found: {model_path}\n\nPlease run `python train.py` first.")
            return None
        state_dict = torch.load(model_path, map_location=config.DEVICE)

        # Fix key mismatch: add 'backbone.' prefix if needed
        if not any(k.startswith('backbone.') for k in state_dict.keys()):
            state_dict = {'backbone.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.to(config.DEVICE)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None
 
def load_rgb_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load RGB image: {e}")
 
def load_ms_image(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with rasterio.open(tmp_path) as src:
            arr = src.read().astype(np.float32)
        Path(tmp_path).unlink()
        return arr
    except Exception as e:
        raise ValueError(f"Failed to load MS TIFF image: {e}")
 
def get_rgb_composite(ms_array):
    rgb = np.stack([ms_array[3], ms_array[2], ms_array[1]], axis=-1)
    vmin, vmax = np.percentile(rgb, [2, 98])
    return np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)
 
def get_false_color_composite(ms_array):
    nrg = np.stack([ms_array[7], ms_array[3], ms_array[2]], axis=-1)
    vmin, vmax = np.percentile(nrg, [2, 98])
    return np.clip((nrg - vmin) / (vmax - vmin + 1e-6), 0, 1)
 
def preprocess_image(img_array, modality):
    from torchvision import transforms
    if modality == 'RGB':
        mean, std = config.RGB_MEAN, config.RGB_STD
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8)) if img_array.max() <= 1 else Image.fromarray(img_array.astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img_tensor = transform(img_pil)
    else:
        mean, std = config.MS_MEAN, config.MS_STD
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.Normalize(mean=mean, std=std)
        ])
        img_tensor = torch.from_numpy(img_array)
        img_tensor = transform(img_tensor)
    return img_tensor.unsqueeze(0).to(config.DEVICE)
 
def run_inference(_img_tensor, _model):
    with torch.no_grad():
        logits = _model(_img_tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)
    return {
        'top_classes': [config.CLASS_NAMES[idx.item()] for idx in top_indices[0]],
        'top_probs':   top_probs[0].cpu().numpy(),
        'all_probs':   probs[0].cpu().numpy()
    }
 
def main():
    st.title("🛰️ Land Use & Land Cover Classification")
    st.markdown("Upload a satellite image and get instant classification with visual analysis.")
 
    with st.sidebar:
        st.header("⚙️ Configuration")
        modality = st.radio("Select Model Type", options=['RGB', 'MS'],
                            help="RGB: 3-channel optical images | MS: 13-channel multispectral")
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        st.write(f"**Classes:** {len(config.CLASS_NAMES)}")
        st.write(f"**Image Size:** {config.IMAGE_SIZE} × {config.IMAGE_SIZE} px")
        st.write(f"**Modality:** {modality}")
        if modality == 'RGB':
            st.write("**Channels:** 3 (Red, Green, Blue)")
        else:
            st.write("**Channels:** 13 (Sentinel-2 bands)")
        st.markdown("---")
        st.subheader("📌 Classes")
        cols = st.columns(2)
        for i, cls in enumerate(config.CLASS_NAMES):
            r, g, b = CLASS_COLORS.get(cls, (128, 128, 128))
            with cols[i % 2]:
                st.markdown(
                    f"<span style='display:inline-block;width:12px;height:12px;"
                    f"background:rgb({r},{g},{b});border-radius:2px;margin-right:5px'></span>{cls}",
                    unsafe_allow_html=True
                )
 
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
 
    try:
        with st.spinner(f"Loading {modality} image..."):
            if modality == 'RGB':
                img_array = load_rgb_image(uploaded_file)
                display_rgb = img_array / 255.0 if img_array.max() > 1 else img_array
            else:
                img_array = load_ms_image(uploaded_file)
                display_rgb = get_rgb_composite(img_array)
        st.success("✅ Image loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return
 
    with st.spinner(f"Loading {modality} model..."):
        model = load_model(modality)
    if model is None:
        return
 
    with st.spinner("Running inference..."):
        img_tensor = preprocess_image(img_array, modality)
        results = run_inference(img_tensor, model)
    st.success("✅ Inference complete")
 
    # ── Results ───────────────────────────────────────────────────────────────
    st.header("📊 Classification Results")
    top_class = results['top_classes'][0]
    r, g, b   = CLASS_COLORS.get(top_class, (128, 128, 128))
 
    col1, col2, col3 = st.columns(3)
 
    # Original image
    with col1:
        st.subheader("🖼️ Original Image")
        if modality == 'MS':
            view_mode = st.radio("Display Mode",
                                 options=['RGB Composite', 'False Color (NIR-R-G)'],
                                 help="False Color highlights vegetation in red")
            if view_mode == 'False Color (NIR-R-G)':
                display_rgb = get_false_color_composite(img_array)
        st.image(display_rgb, caption="Uploaded Patch", width='stretch')
 
    # Color overlay
    with col2:
        st.subheader("🎨 Classification Overlay")
        overlay_img = apply_color_overlay(display_rgb, top_class, alpha=0.45)
        st.image(overlay_img, caption=f"Predicted: {top_class}", width='stretch')
        # Legend badge
        st.markdown(
            f"<div style='background-color:rgb({r},{g},{b});padding:8px 16px;"
            f"border-radius:8px;display:inline-block;color:white;"
            f"font-weight:bold;margin-top:4px'>🏷️ {top_class}</div>",
            unsafe_allow_html=True
        )
 
    # Bar chart
    with col3:
        st.subheader("🎯 Top 3 Predictions")
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        label_color = "#808080"
        ax.tick_params(colors=label_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(label_color)
 
        bar_colors = []
        for p in results['top_probs']:
            if p > 0.6:   bar_colors.append('#2ecc71')
            elif p > 0.3: bar_colors.append('#f39c12')
            else:         bar_colors.append('#e74c3c')
 
        bars = ax.barh(results['top_classes'], results['top_probs'] * 100, color=bar_colors)
        ax.set_xlabel('Probability (%)', color=label_color)
        ax.set_title('Confidence', color=label_color)
        ax.set_xlim(0, 110)
        for bar, prob in zip(bars, results['top_probs']):
            ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f'{prob*100:.1f}%', va='center', fontsize=9,
                    fontweight='bold', color=label_color)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
 
    # ── Detailed metrics ──────────────────────────────────────────────────────
    st.header("📈 Detailed Analysis")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Top Prediction", results['top_classes'][0], f"{results['top_probs'][0]*100:.1f}%")
    with metric_cols[1]:
        st.metric("2nd Place", results['top_classes'][1], f"{results['top_probs'][1]*100:.1f}%")
    with metric_cols[2]:
        st.metric("3rd Place", results['top_classes'][2], f"{results['top_probs'][2]*100:.1f}%")
 
    st.subheader("All Class Probabilities")
    st.dataframe({
        'Class': config.CLASS_NAMES,
        'Probability (%)': [f"{p*100:.2f}%" for p in results['all_probs']]
    }, use_container_width=True)
 
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:20px;opacity:0.7'>
        🛰️ <b>LULC Classification System</b> | ResNet50 & Sentinel-2 | DEPI Graduation Project
    </div>""", unsafe_allow_html=True)
 
if __name__ == '__main__':
    main()
