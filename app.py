# app.py
import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import torch
import torchvision.transforms as T
from model import GeleNet

# ----------------------------
# Hugging Face helper (downloads model files from HF repo; supports private repos via HF_TOKEN)
# ----------------------------
from huggingface_hub import hf_hub_download

# <-- SET THIS to your HF repo id (case-sensitive)
HF_REPO = "Satya55555/satya-gele-models"

# Map the filenames users select to files stored in the HF repo
HF_MODEL_FILES = {
    "GeleNet_ORSSD.pth": "GeleNet_ORSSD.pth",
    "GeleNet_EORSSD.pth": "GeleNet_EORSSD.pth"
}

def get_model_file_from_hf(filename: str):
    """
    Ensure `filename` exists locally. If not, download it from the HF repo.
    Uses st.secrets['HF_TOKEN'] if present (for private/gated repos).
    Returns local filename or None on failure.
    """
    if os.path.exists(filename):
        return filename

    if filename not in HF_MODEL_FILES:
        st.error(f"{filename} is not configured for Hugging Face download.")
        return None

    hf_filename = HF_MODEL_FILES[filename]
    st.info(f"Ensuring model {hf_filename} is available (HF repo: {HF_REPO}) ...")

    # Read token from Streamlit secrets (works on Streamlit Cloud) or None for public repos
    token = st.secrets.get("HF_TOKEN", None)

    try:
        local_hf_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            use_auth_token=token,
            cache_dir="hf_cache",
            force_download=False
        )
    except Exception as e:
        st.error(f"Failed to download {hf_filename} from Hugging Face: {e}")
        return None

    try:
        # Move HF cache file to working dir with expected filename (atomic replace)
        os.replace(local_hf_path, filename)
    except Exception:
        # Fallback copy if replace fails (cross-device)
        import shutil
        try:
            shutil.copy(local_hf_path, filename)
        except Exception as e:
            st.error(f"Failed to copy downloaded model to working dir: {e}")
            return None

    st.success(f"Downloaded {filename}")
    return filename

# ----------------------------
# Page config and CSS tweaks
# ----------------------------
st.set_page_config(page_title="GeleNet Viewer", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* reduce top padding */
    .css-18e3th9 { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
    /* reduce main block padding */
    .css-1d391kg { padding-top: .5rem; }
    /* responsive images */
    img { max-width: 100% !important; height: auto !important; }
    /* narrow the sidebar control width a bit */
    [data-testid="stSidebar"] { width: 320px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Constants
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE = (352, 352)

# ----------------------------
# Helpers
# ----------------------------
def prepare_tensor(img_pil):
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img_pil).unsqueeze(0)

def probs_to_mask_image(probs, size=None):
    arr = (probs * 255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    if size is not None:
        img = img.resize(size)
    return img

def create_overlay(orig_pil, probs, alpha=0.5):
    """Create an RGBA overlay where `probs` defines alpha channel (0..1)."""
    orig = orig_pil.convert("RGBA")
    mask_u8 = (probs * 255).astype(np.uint8)
    # apply global alpha by scaling mask values
    scaled = (mask_u8.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    alpha_mask = Image.fromarray(scaled).convert("L")
    color_mask = Image.new("RGBA", orig.size, color=(255, 0, 0, 0))
    color_mask.putalpha(alpha_mask)
    overlay = Image.alpha_composite(orig, color_mask)
    return overlay

def get_bytes_from_pil(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ----------------------------
# Cached model loader
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeleNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

# ----------------------------
# Sidebar (left) - navbar style
# ----------------------------
st.sidebar.title("🔥 GeleNet Salient Object Detection")
st.sidebar.markdown("Upload an image and select a model. Then press **Run Inference**.")

uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg","jpeg","png"], help="Drag and drop file here")
st.sidebar.markdown("---")

# Model selection dropdown (assumes these .pth files are in repo or will be downloaded from HF)
available_models = []
for choice in ["GeleNet_ORSSD.pth", "GeleNet_EORSSD.pth"]:
    # show in dropdown regardless; we'll download if not present
    available_models.append(choice)

checkpoint_choice = st.sidebar.selectbox("Select a model", available_models)
st.sidebar.markdown("---")

threshold = st.sidebar.slider("Binarization Threshold", 0.0, 1.0, 0.5, 0.01)
alpha = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")

use_gpu = st.sidebar.checkbox("Prefer GPU (if available)", value=False)
run = st.sidebar.button("Run Inference")

st.sidebar.markdown("\n\n")
st.sidebar.caption("Tip: upload 800×600 or smaller images for faster CPU inference.")

# ----------------------------
# Main area (right)
# ----------------------------
st.header("Result Viewer")

if not uploaded:
    st.info("Upload an image from the left sidebar (Drag and drop supported).")
    st.stop()

# Ensure model file exists locally (download from HF if needed)
model_local_path = None
try:
    model_local_path = get_model_file_from_hf(checkpoint_choice)
except Exception as e:
    st.error(f"Error while ensuring model file: {e}")

if model_local_path is None:
    st.warning(f"Model `{checkpoint_choice}` not found and download failed. Please check HF_REPO and HF_TOKEN settings.")
    st.stop()

# Load model (cached)
try:
    model, device_default = load_model(model_local_path)
except Exception as e:
    st.error(f"Failed loading model `{model_local_path}`: {e}")
    st.stop()

# honor user GPU preference if available
device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
if device != device_default:
    model = model.to(device)

# Run inference
if run:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
        st.stop()

    input_tensor = prepare_tensor(img).to(device)
    with torch.no_grad():
        logits = model(input_tensor)  # [1,1,H,W]
        probs = torch.sigmoid(logits)[0][0].cpu().numpy()  # HxW float in [0,1]

    # Resize prediction to original image size
    prob_img = probs_to_mask_image(probs)  # at model resolution
    prob_img_resized = prob_img.resize(img.size)
    probs_resized = np.array(prob_img_resized).astype(np.float32) / 255.0

    # Binary mask
    bin_mask = (probs_resized >= threshold).astype(np.uint8) * 255
    bin_mask_pil = Image.fromarray(bin_mask)

    # Overlay
    overlay_pil = create_overlay(img, probs_resized, alpha=alpha)

    # 2x2 grid: use two rows of two columns
    row1_col1, row1_col2 = st.columns(2, gap="large")
    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row1_col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    with row1_col2:
        st.subheader("Probability Mask")
        st.image(prob_img_resized.convert("L"), channels="L", use_container_width=True)

    with row2_col1:
        st.subheader("Binary Mask")
        st.image(bin_mask_pil, use_container_width=True)

    with row2_col2:
        st.subheader("Overlay")
        st.image(overlay_pil, use_container_width=True)

    # Downloads below grid
    st.markdown("---")
    st.subheader("Download Results")
    col_dl1, col_dl2, col_dl3 = st.columns([1,1,1])
    with col_dl1:
        st.download_button("Download Probability Mask", get_bytes_from_pil(prob_img_resized), file_name="probability_mask.png", mime="image/png")
    with col_dl2:
        st.download_button("Download Binary Mask", get_bytes_from_pil(bin_mask_pil), file_name="binary_mask.png", mime="image/png")
    with col_dl3:
        st.download_button("Download Overlay", get_bytes_from_pil(overlay_pil), file_name="overlay.png")

else:
    st.info("Click **Run Inference** in the left sidebar after uploading an image and selecting a model.")
