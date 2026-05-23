"""
Streamlit web application for LULC Classification inference.
What: Interactive web dashboard for uploading satellite images and running inference
      with dual visualization (original + segmentation map using sliding window).
Usage:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import torch
import numpy as np
from PIL import Image
import rasterio
from pathlib import Path
import tempfile
import config
from model import ResNet50Classifier

# ─────────────────────────────────────────────────────────────────────────────
# Class Colors (consistent across the app)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'AnnualCrop':           [255, 255,   0],   # أصفر
    'Forest':               [  0, 100,   0],   # أخضر غامق
    'HerbaceousVegetation': [  0, 200,   0],   # أخضر فاتح
    'Highway':              [128, 128, 128],   # رمادي
    'Industrial':           [255,   0,   0],   # أحمر
    'Pasture':              [144, 238, 144],   # أخضر فاتح جداً
    'PermanentCrop':        [255, 165,   0],   # برتقالي
    'Residential':          [255,  20, 147],   # وردي
    'River':                [  0,   0, 255],   # أزرق غامق
    'SeaLake':              [  0, 191, 255],   # أزرق فاتح
}

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
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.2);
}
[data-testid="stMetricLabel"] {
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Caching & Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(modality):
    """
    Load trained model checkpoint (cached in memory).
    Handles key mismatch: .pth files saved without 'backbone.' prefix
    but ResNet50Classifier expects 'backbone.xxx' keys.
    """
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

        raw_state_dict = torch.load(model_path, map_location=config.DEVICE)

        # ── تحديد الـ keys المتوقعة في الـ model ──────────────────────────
        model_keys   = set(model.state_dict().keys())       # backbone.conv1.weight ...
        saved_keys   = set(raw_state_dict.keys())           # conv1.weight ...

        needs_backbone_prefix = (
            any(k.startswith('backbone.') for k in model_keys) and
            not any(k.startswith('backbone.') for k in saved_keys)
        )

        if needs_backbone_prefix:
            # أضف prefix "backbone." لكل key
            fixed_state_dict = {
                f'backbone.{k}': v
                for k, v in raw_state_dict.items()
            }
        else:
            fixed_state_dict = raw_state_dict

        # load بـ strict=False عشان نتجاهل أي keys زيادة مش محتاجاها
        missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)

        if missing:
            st.warning(f"⚠️ Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            st.warning(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

        model.eval()
        model = model.to(config.DEVICE)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Image Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_rgb_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    return np.array(img)

def load_ms_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        arr = src.read().astype(np.float32)   # (13, H, W)
    Path(tmp_path).unlink()
    return arr

def get_rgb_composite(ms_array):
    """B04-B03-B02 for natural color display."""
    rgb = np.stack([ms_array[3], ms_array[2], ms_array[1]], axis=-1)
    vmin, vmax = np.percentile(rgb, [2, 98])
    return np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)

def get_false_color_composite(ms_array):
    """B08-B04-B03 (NIR-Red-Green) to highlight vegetation."""
    nrg = np.stack([ms_array[7], ms_array[3], ms_array[2]], axis=-1)
    vmin, vmax = np.percentile(nrg, [2, 98])
    return np.clip((nrg - vmin) / (vmax - vmin + 1e-6), 0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & Inference
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_array, modality):
    from torchvision import transforms
    if modality == 'RGB':
        mean, std = config.RGB_MEAN, config.RGB_STD
        img_pil = (Image.fromarray((img_array * 255).astype(np.uint8))
                   if img_array.max() <= 1 else Image.fromarray(img_array.astype(np.uint8)))
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform(img_pil).unsqueeze(0).to(config.DEVICE)
    else:
        mean, std = config.MS_MEAN, config.MS_STD
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform(torch.from_numpy(img_array)).unsqueeze(0).to(config.DEVICE)

def run_inference(img_tensor, model):
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)
    return {
        'top_classes': [config.CLASS_NAMES[i.item()] for i in top_indices[0]],
        'top_probs':   top_probs[0].cpu().numpy(),
        'all_probs':   probs[0].cpu().numpy(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Sliding-Window Segmentation
# ─────────────────────────────────────────────────────────────────────────────
def segment_image_sliding_window(img_array, model, modality, patch_size=64, stride=32):
    """
    تقسيم الصورة لـ patches وتصنيف كل patch بشكل منفصل،
    ثم رسم لون كل class على الخريطة النهائية.
    """
    if modality == 'RGB':
        h, w = img_array.shape[:2]
    else:
        h, w = img_array.shape[1], img_array.shape[2]

    color_map = np.zeros((h, w, 3), dtype=np.float32)
    count_map = np.zeros((h, w),    dtype=np.float32)

    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))
    total = len(y_positions) * len(x_positions)

    progress_bar = st.progress(0, text="جاري تحليل الصورة...")
    step = 0

    for y in y_positions:
        for x in x_positions:
            # استخراج الـ patch
            if modality == 'RGB':
                patch = img_array[y:y+patch_size, x:x+patch_size]
            else:
                patch = img_array[:, y:y+patch_size, x:x+patch_size]

            # تصنيف
            tensor = preprocess_image(patch, modality)
            with torch.no_grad():
                logits = model(tensor)
                pred_idx = torch.argmax(logits, dim=1).item()

            color = CLASS_COLORS[config.CLASS_NAMES[pred_idx]]

            # تلوين المنطقة
            color_map[y:y+patch_size, x:x+patch_size] += color
            count_map[y:y+patch_size, x:x+patch_size] += 1

            step += 1
            progress_bar.progress(step / total,
                                  text=f"جاري التحليل... {step}/{total} patch")

    progress_bar.empty()

    # تطبيع اللون في المناطق المتداخلة
    count_map = np.maximum(count_map, 1)
    color_map /= count_map[:, :, np.newaxis]
    return np.clip(color_map / 255.0, 0, 1)


def build_legend_figure():
    """رسم legend صغير لكل الـ classes."""
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_alpha(0)
    ax.set_visible(False)
    patches = [
        mpatches.Patch(color=[c/255 for c in rgb], label=cls)
        for cls, rgb in CLASS_COLORS.items()
    ]
    fig.legend(handles=patches, loc='center', ncol=5,
               framealpha=0, fontsize=9)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🛰️ Land Use & Land Cover Classification")
    st.markdown("Upload a satellite image and get instant classification with visual analysis.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        modality = st.radio(
            "Select Model Type",
            options=['RGB', 'MS'],
            help="RGB: 3-channel optical | MS: 13-channel Sentinel-2"
        )
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        st.write(f"**Classes:** {len(config.CLASS_NAMES)}")
        st.write(f"**Patch Size:** {config.IMAGE_SIZE} × {config.IMAGE_SIZE} px")
        st.write(f"**Modality:** {modality}")
        st.markdown("---")
        st.subheader("🗺️ Segmentation Settings")
        stride = st.slider(
            "Stride (pixels)",
            min_value=16, max_value=64, value=32, step=8,
            help="قيمة أصغر = تفاصيل أكثر لكن أبطأ"
        )
        alpha = st.slider(
            "Overlay Opacity",
            min_value=0.1, max_value=0.9, value=0.5, step=0.1,
            help="شفافية طبقة الألوان فوق الصورة الأصلية"
        )
        st.markdown("---")
        st.subheader("📌 Classes")
        for cls, rgb in CLASS_COLORS.items():
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
            st.markdown(
                f'<span style="background:{hex_color};'
                f'display:inline-block;width:14px;height:14px;'
                f'border-radius:3px;margin-right:6px;'
                f'border:1px solid #555"></span>{cls}',
                unsafe_allow_html=True
            )

    # ── File Upload ───────────────────────────────────────────────────────────
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader(
        f"Upload a {modality} image",
        type=['jpg', 'jpeg', 'tif', 'tiff'],
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started")
        return

    # ── Load Image ────────────────────────────────────────────────────────────
    try:
        with st.spinner("Loading image..."):
            if modality == 'RGB':
                img_array   = load_rgb_image(uploaded_file)
                display_rgb = img_array / 255.0
            else:
                img_array   = load_ms_image(uploaded_file)
                display_rgb = get_rgb_composite(img_array)
        st.success("✅ Image loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return

    # ── Load Model ────────────────────────────────────────────────────────────
    with st.spinner("Loading model..."):
        model = load_model(modality)
    if model is None:
        return

    # ── Single-Image Inference ────────────────────────────────────────────────
    with st.spinner("Running inference..."):
        img_tensor = preprocess_image(img_array, modality)
        results    = run_inference(img_tensor, model)
    st.success("✅ Inference complete")

    # ── Tabs: Single Prediction | Segmentation Map ───────────────────────────
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "🗺️ Segmentation Map"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Single Prediction (الطريقة القديمة محسّنة)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("📊 Classification Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Satellite Image")
            if modality == 'MS':
                view_mode = st.radio(
                    "Display Mode",
                    ['RGB Composite', 'False Color (NIR-R-G)'],
                    key="tab1_view"
                )
                if view_mode == 'False Color (NIR-R-G)':
                    display_rgb = get_false_color_composite(img_array)
            st.image(display_rgb, caption="Uploaded Patch", use_container_width=True)

        with col2:
            st.subheader("🎯 Top 3 Predictions")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor((0, 0, 0, 0))
            label_color = "#888888"

            bars = ax.barh(results['top_classes'], results['top_probs'] * 100)
            for i, bar in enumerate(bars):
                p = results['top_probs'][i]
                bar.set_color('#2ecc71' if p > 0.6 else '#f39c12' if p > 0.3 else '#e74c3c')

            ax.set_xlabel('Probability (%)', color=label_color)
            ax.set_title('Classification Confidence', color=label_color)
            ax.set_xlim(0, 100)
            ax.tick_params(colors=label_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(label_color)

            for bar, prob in zip(bars, results['top_probs']):
                ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                        f'{prob*100:.1f}%', va='center', fontsize=10,
                        fontweight='bold', color=label_color)
            plt.tight_layout()
            st.pyplot(fig, transparent=True)

        # Metrics
        st.header("📈 Detailed Analysis")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Top Prediction", results['top_classes'][0],
                      f"{results['top_probs'][0]*100:.1f}%")
        with m2:
            st.metric("2nd Place", results['top_classes'][1],
                      f"{results['top_probs'][1]*100:.1f}%")
        with m3:
            st.metric("3rd Place", results['top_classes'][2],
                      f"{results['top_probs'][2]*100:.1f}%")

        st.subheader("All Class Probabilities")
        st.dataframe({
            'Class':            config.CLASS_NAMES,
            'Probability (%)': [f"{p*100:.2f}%" for p in results['all_probs']],
        }, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Segmentation Map
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("🗺️ Segmentation Map")
        st.markdown(
            "الـ model بيقسّم الصورة لـ patches صغيرة (64×64) ويصنف كل واحدة لوحدها، "
            "وبعدين بيرسم لون كل نوع غطاء أرضي على الخريطة."
        )

        if modality == 'MS':
            view_mode2 = st.radio(
                "Background Image",
                ['RGB Composite', 'False Color (NIR-R-G)'],
                key="tab2_view"
            )
            bg = get_false_color_composite(img_array) if view_mode2 == 'False Color (NIR-R-G)' else display_rgb
        else:
            bg = display_rgb

        run_seg = st.button("▶️ ابدأ التحليل", type="primary")

        if run_seg:
            seg_map = segment_image_sliding_window(
                img_array, model, modality,
                patch_size=config.IMAGE_SIZE,
                stride=stride
            )

            # دمج الصورة الأصلية مع الـ segmentation map
            # تأكد إن الـ bg والـ seg_map نفس الأبعاد
            bg_resized   = np.array(Image.fromarray((bg * 255).astype(np.uint8))
                                    .resize((seg_map.shape[1], seg_map.shape[0]))) / 255.0
            overlay      = np.clip((1 - alpha) * bg_resized + alpha * seg_map, 0, 1)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📷 الصورة الأصلية")
                st.image(bg, use_container_width=True)
            with c2:
                st.subheader("🎨 Segmentation Map")
                st.image(overlay, use_container_width=True, caption="Overlay")

            # عرض الـ Segmentation Map بدون overlay
            st.subheader("🗺️ Pure Segmentation Map")
            st.image(seg_map, use_container_width=True)

            # Legend
            st.subheader("🎨 Legend")
            st.pyplot(build_legend_figure(), transparent=True)

            # إحصائيات المساحة
            st.subheader("📊 Land Cover Statistics")
            # تحويل seg_map لـ labels بناءً على أقرب لون
            color_array = np.array(list(CLASS_COLORS.values()), dtype=np.float32) / 255.0
            flat_pixels = seg_map.reshape(-1, 3)
            # حساب أقرب class لكل pixel
            diffs = np.linalg.norm(
                flat_pixels[:, None, :] - color_array[None, :, :], axis=2
            )
            pixel_labels = np.argmin(diffs, axis=1)
            total_pixels = len(pixel_labels)
            stats = {}
            for idx, cls in enumerate(config.CLASS_NAMES):
                count = np.sum(pixel_labels == idx)
                if count > 0:
                    stats[cls] = count / total_pixels * 100

            stats_sorted = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor((0, 0, 0, 0))
            lc = "#888888"
            bars2 = ax2.bar(
                list(stats_sorted.keys()),
                list(stats_sorted.values()),
                color=[[c/255 for c in CLASS_COLORS[k]] for k in stats_sorted]
            )
            ax2.set_ylabel('Coverage (%)', color=lc)
            ax2.set_title('Land Cover Distribution', color=lc)
            ax2.tick_params(colors=lc, axis='both')
            plt.xticks(rotation=30, ha='right')
            for spine in ax2.spines.values():
                spine.set_edgecolor(lc)
            for bar, val in zip(bars2, stats_sorted.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.1f}%', ha='center', va='bottom',
                         fontsize=9, color=lc)
            plt.tight_layout()
            st.pyplot(fig2, transparent=True)

        else:
            st.info("👆 اضغط 'ابدأ التحليل' لتشغيل الـ Segmentation Map")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:15px;opacity:0.7;font-size:13px">
        🛰️ <b>LULC Classification System</b> | Powered by ResNet50 & Sentinel-2<br>
        <span style="font-size:11px">DEPI Graduation Project</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
