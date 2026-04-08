import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
import importlib.util
import sys
import logging
from pathlib import Path
import shutil
from glob import glob
from torchvision.models import (
    VGG11_Weights, VGG11_BN_Weights, VGG13_Weights, VGG13_BN_Weights,
    VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights,
    AlexNet_Weights, DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights,
    Inception_V3_Weights, GoogLeNet_Weights, MobileNet_V2_Weights, MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights, MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights,
    MNASNet1_3_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights,
    EfficientNet_B7_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights,
    RegNet_X_400MF_Weights, RegNet_X_800MF_Weights, RegNet_X_1_6GF_Weights, RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights, RegNet_X_16GF_Weights, RegNet_X_32GF_Weights,
    RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights,
    RegNet_Y_8GF_Weights, RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights,
    MaxVit_T_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights
)
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, STL10
import tempfile
import io
import zipfile

# ─── Logging & Warnings ────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/app.log", level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Page Config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Model Feature Extractor",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# ─── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

# ─── Models Dictionary ─────────────────────────────────────────────────────────
MODELS_DICT = {
    "VGG11": (models.vgg11, VGG11_Weights.DEFAULT),
    "VGG11_BN": (models.vgg11_bn, VGG11_BN_Weights.DEFAULT),
    "VGG13": (models.vgg13, VGG13_Weights.DEFAULT),
    "VGG13_BN": (models.vgg13_bn, VGG13_BN_Weights.DEFAULT),
    "VGG16": (models.vgg16, VGG16_Weights.DEFAULT),
    "VGG16_BN": (models.vgg16_bn, VGG16_BN_Weights.DEFAULT),
    "VGG19": (models.vgg19, VGG19_Weights.DEFAULT),
    "VGG19_BN": (models.vgg19_bn, VGG19_BN_Weights.DEFAULT),
    "ResNet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "ResNet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "ResNet50": (models.resnet50, ResNet50_Weights.DEFAULT),
    "ResNet101": (models.resnet101, ResNet101_Weights.DEFAULT),
    "ResNet152": (models.resnet152, ResNet152_Weights.DEFAULT),
    "ResNeXt50_32x4d": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.DEFAULT),
    "ResNeXt101_32x8d": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.DEFAULT),
    "ResNeXt101_64x4d": (models.resnext101_64x4d, ResNeXt101_64X4D_Weights.DEFAULT),
    "Wide_ResNet50_2": (models.wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT),
    "Wide_ResNet101_2": (models.wide_resnet101_2, Wide_ResNet101_2_Weights.DEFAULT),
    "AlexNet": (models.alexnet, AlexNet_Weights.DEFAULT),
    "DenseNet121": (models.densenet121, DenseNet121_Weights.DEFAULT),
    "DenseNet161": (models.densenet161, DenseNet161_Weights.DEFAULT),
    "DenseNet169": (models.densenet169, DenseNet169_Weights.DEFAULT),
    "DenseNet201": (models.densenet201, DenseNet201_Weights.DEFAULT),
    "Inception_V3": (models.inception_v3, Inception_V3_Weights.DEFAULT),
    "GoogLeNet": (models.googlenet, GoogLeNet_Weights.DEFAULT),
    "MobileNet_V2": (models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
    "MobileNet_V3_Small": (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
    "MobileNet_V3_Large": (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
    "MNASNet0_5": (models.mnasnet0_5, MNASNet0_5_Weights.DEFAULT),
    "MNASNet0_75": (models.mnasnet0_75, MNASNet0_75_Weights.DEFAULT),
    "MNASNet1_0": (models.mnasnet1_0, MNASNet1_0_Weights.DEFAULT),
    "MNASNet1_3": (models.mnasnet1_3, MNASNet1_3_Weights.DEFAULT),
    "EfficientNet_B0": (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "EfficientNet_B1": (models.efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
    "EfficientNet_B2": (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    "EfficientNet_B3": (models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
    "EfficientNet_B4": (models.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
    "EfficientNet_B5": (models.efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
    "EfficientNet_B6": (models.efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
    "EfficientNet_B7": (models.efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
    "EfficientNet_V2_S": (models.efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
    "EfficientNet_V2_M": (models.efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT),
    "EfficientNet_V2_L": (models.efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT),
    "RegNet_X_400MF": (models.regnet_x_400mf, RegNet_X_400MF_Weights.DEFAULT),
    "RegNet_X_800MF": (models.regnet_x_800mf, RegNet_X_800MF_Weights.DEFAULT),
    "RegNet_X_1_6GF": (models.regnet_x_1_6gf, RegNet_X_1_6GF_Weights.DEFAULT),
    "RegNet_X_3_2GF": (models.regnet_x_3_2gf, RegNet_X_3_2GF_Weights.DEFAULT),
    "RegNet_X_8GF": (models.regnet_x_8gf, RegNet_X_8GF_Weights.DEFAULT),
    "RegNet_X_16GF": (models.regnet_x_16gf, RegNet_X_16GF_Weights.DEFAULT),
    "RegNet_X_32GF": (models.regnet_x_32gf, RegNet_X_32GF_Weights.DEFAULT),
    "RegNet_Y_400MF": (models.regnet_y_400mf, RegNet_Y_400MF_Weights.DEFAULT),
    "RegNet_Y_800MF": (models.regnet_y_800mf, RegNet_Y_800MF_Weights.DEFAULT),
    "RegNet_Y_1_6GF": (models.regnet_y_1_6gf, RegNet_Y_1_6GF_Weights.DEFAULT),
    "RegNet_Y_3_2GF": (models.regnet_y_3_2gf, RegNet_Y_3_2GF_Weights.DEFAULT),
    "RegNet_Y_8GF": (models.regnet_y_8gf, RegNet_Y_8GF_Weights.DEFAULT),
    "RegNet_Y_16GF": (models.regnet_y_16gf, RegNet_Y_16GF_Weights.DEFAULT),
    "RegNet_Y_32GF": (models.regnet_y_32gf, RegNet_Y_32GF_Weights.DEFAULT),
    "RegNet_Y_128GF": (models.regnet_y_128gf, RegNet_Y_128GF_Weights.DEFAULT),
    "ViT_B_16": (models.vit_b_16, ViT_B_16_Weights.DEFAULT),
    "ViT_B_32": (models.vit_b_32, ViT_B_32_Weights.DEFAULT),
    "ViT_L_16": (models.vit_l_16, ViT_L_16_Weights.DEFAULT),
    "ViT_L_32": (models.vit_l_32, ViT_L_32_Weights.DEFAULT),
    "ViT_H_14": (models.vit_h_14, ViT_H_14_Weights.DEFAULT),
    "Swin_T": (models.swin_t, Swin_T_Weights.DEFAULT),
    "Swin_S": (models.swin_s, Swin_S_Weights.DEFAULT),
    "Swin_B": (models.swin_b, Swin_B_Weights.DEFAULT),
    "Swin_V2_T": (models.swin_v2_t, Swin_V2_T_Weights.DEFAULT),
    "Swin_V2_S": (models.swin_v2_s, Swin_V2_S_Weights.DEFAULT),
    "Swin_V2_B": (models.swin_v2_b, Swin_V2_B_Weights.DEFAULT),
    "MaxVit_T": (models.maxvit_t, MaxVit_T_Weights.DEFAULT),
    "ConvNeXt_Tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "ConvNeXt_Small": (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT),
    "ConvNeXt_Base": (models.convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    "ConvNeXt_Large": (models.convnext_large, ConvNeXt_Large_Weights.DEFAULT),
}

# ─── Session State Init ────────────────────────────────────────────────────────
DEFAULTS = {
    "selected_model": None,
    "selected_model_name": "",
    "layer_shapes": {},
    "npz_file_path": None,
    "download_weights": True,
    "pretrained_models": {},
    "model_file_path": None,
    "custom_model_classes": [],
    "custom_module": None,
    "architecture_loaded": False,
    "weights_loaded": False,
    "extraction_done": False,
    "extracted_npz_bytes": None,
    "extracted_npz_name": "",
    "failed_images": [],
    "success_count": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helper Functions ──────────────────────────────────────────────────────────

def to_device(tensor):
    """Move tensor to the appropriate device (GPU/CPU)."""
    return tensor.to(device)

def to_gpu(arr):
    """Transfer array to GPU if available, else keep on CPU."""
    return cp.asarray(arr) if GPU_AVAILABLE else np.asarray(arr)

def to_cpu(arr):
    """Transfer array to CPU for NumPy operations."""
    return cp.asnumpy(arr) if GPU_AVAILABLE and isinstance(arr, cp.ndarray) else np.asarray(arr)

def get_model(model_name, download_weights_flag):
    """Load a pretrained model from torchvision.models with optional weights."""
    if model_name not in MODELS_DICT:
        raise ValueError(f"Unknown model: {model_name}.")
    model_fn, weights = MODELS_DICT[model_name]
    model = model_fn(weights=weights if download_weights_flag else None)
    return model.to(device)

def preprocess_image(image, transform_choice, mode_value, target_size, use_rgb):
    """Preprocess image based on mode and transformation choice."""
    try:
        num_channels = 3 if use_rgb or mode_value == "Pretrained" else 1
        image = image.convert("RGB" if num_channels == 3 else "L")
        steps = []
        if transform_choice == "Crop":
            steps.append(transforms.Resize(max(target_size),
                                           interpolation=transforms.InterpolationMode.BILINEAR))
            steps.append(transforms.CenterCrop(target_size))
        elif transform_choice == "Resize":
            steps.append(transforms.Resize(target_size,
                                           interpolation=transforms.InterpolationMode.BILINEAR))
        # "None" → no spatial transform
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406] if num_channels == 3 else [0.5],
            std=[0.229, 0.224, 0.225]  if num_channels == 3 else [0.5]
        ))
        transform = transforms.Compose(steps)
        return to_device(transform(image).unsqueeze(0))
    except Exception as e:
        logging.error(f"Failed to preprocess image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def compute_layer_shapes(model, mode_value, height, width, use_rgb):
    """
    Run a dummy forward pass and record output shapes for every named module.
    Mirrors the tkinter compute_layer_shapes() exactly.
    """
    layer_shapes = {}
    channels = 3 if mode_value == "Pretrained" or use_rgb else 1
    input_size = (224, 224) if mode_value == "Pretrained" else (height, width)
    dummy_input = to_device(torch.randn(1, channels, *input_size))
    hooks = []

    def make_hook(name):
        def hook(module, inp, output):
            try:
                out = output
                # handle tuple outputs (e.g. Inception)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    out = nn.AdaptiveAvgPool2d((1, 1))(out).squeeze(-1).squeeze(-1)
                if isinstance(out, torch.Tensor):
                    layer_shapes[name] = list(out.shape)
            except Exception:
                pass
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        logging.error(f"Failed to compute layer shapes: {e}")
        st.error(f"Failed to compute layer shapes: {e}")
    finally:
        for h in hooks:
            h.remove()

    return layer_shapes

def show_model_file_contents(file_path):
    """
    Return only the class-definition lines from a .py file.
    Mirrors tkinter show_model_file_contents().
    """
    try:
        with open(file_path, "r") as f:
            lines = [l for l in f if l.strip().startswith("class ")]
        return "".join(lines) if lines else None
    except Exception as e:
        logging.error(f"Could not read model file: {e}")
        return None

def select_pretrained_model(model_name, download_weights_flag):
    """
    Load a pretrained model on-demand and store it in session state.
    Mirrors tkinter select_pretrained_model().
    """
    if model_name not in st.session_state.pretrained_models:
        st.session_state.pretrained_models[model_name] = get_model(
            model_name, download_weights_flag
        )
    model = st.session_state.pretrained_models[model_name]
    model.eval()
    st.session_state.selected_model = model
    st.session_state.selected_model_name = model_name
    st.session_state.npz_file_path = None
    logging.debug(f"Selected pretrained model: {model_name}")


def extract_features(layer_name, dataset_source, transform_choice, dataset_name,
                     uploaded_images, use_rgb, target_size, mode_value):
    """
    Full port of tkinter extract_features().
    - Supports Local (uploaded files via Streamlit) and Standard datasets.
    - Saves individual .npy files per image (same folder structure as tkinter).
    - Saves a combined .npz for all images.
    - Returns (feature_list, label_list, failed_list, npz_path).
    """
    model = st.session_state.selected_model
    if model is None:
        st.error("Please load / select a model first!")
        return [], [], [], None

    # ── Resolve image items ────────────────────────────────────────────────────
    dataset = None
    classes = None
    image_items = []        # list of (pil_image, label_str, filename_str)

    if dataset_source == "Local":
        if not uploaded_images:
            st.warning("No images uploaded.")
            return [], [], [], None
        for uf in uploaded_images:
            try:
                img = Image.open(uf)
                # derive label from the filename prefix before the last '_' or use "unknown"
                label = Path(uf.name).parent.name if hasattr(uf, "name") else "unknown"
                image_items.append((img, label, uf.name))
            except Exception as e:
                image_items.append((None, "unknown", uf.name))
    else:
        # Standard dataset (same as tkinter)
        data_root = "./data"
        os.makedirs(data_root, exist_ok=True)
        with st.spinner(f"Downloading / loading {dataset_name} …"):
            if dataset_name == "CIFAR10":
                dataset = CIFAR10(root=data_root, download=True, train=False)
                classes = dataset.classes
                use_rgb = True
            elif dataset_name == "CIFAR100":
                dataset = CIFAR100(root=data_root, download=True, train=False)
                classes = dataset.classes
                use_rgb = True
            elif dataset_name == "MNIST":
                dataset = MNIST(root=data_root, download=True, train=False)
                classes = [str(i) for i in range(10)]
                use_rgb = False
            elif dataset_name == "FashionMNIST":
                dataset = FashionMNIST(root=data_root, download=True, train=False)
                classes = dataset.classes
                use_rgb = False
            elif dataset_name == "STL10":
                dataset = STL10(root=data_root, download=True, split="test")
                classes = dataset.classes
                use_rgb = True
        for idx in range(len(dataset)):
            img, label_idx = dataset[idx]
            if not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            image_items.append((img, classes[label_idx], f"image_{idx + 1}.jpg"))

    # ── Register forward hook ──────────────────────────────────────────────────
    hook_output = {"value": None}

    def hook_fn(module, inp, output):
        out = output
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            out = nn.AdaptiveAvgPool2d((1, 1))(out).squeeze(-1).squeeze(-1)
        if isinstance(out, torch.Tensor):
            hook_output["value"] = out.detach().cpu().numpy()

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break

    if handle is None:
        st.error(f"Layer '{layer_name}' not found in model!")
        return [], [], [], None

    # ── Process images ─────────────────────────────────────────────────────────
    failed = []
    feature_list = []
    label_list = []

    progress_bar = st.progress(0.0)
    status_text   = st.empty()
    total = len(image_items)

    for i, (image, label, filename) in enumerate(image_items, 1):
        status_text.text(f"Processing {i}/{total} — {filename}")
        progress_bar.progress(i / total)

        if image is None:
            failed.append(f"{filename}: could not open image")
            continue

        try:
            image_tensor = preprocess_image(image, transform_choice, mode_value,
                                            target_size, use_rgb)
            hook_output["value"] = None
            with torch.no_grad():
                model(image_tensor)

            if hook_output["value"] is not None:
                feature_list.append(to_gpu(hook_output["value"].flatten()))
                label_list.append(label)
        except Exception as e:
            failed.append(f"{filename}: {str(e)}")

    handle.remove()
    progress_bar.empty()
    status_text.empty()

    # ── Save combined .npz  (identical to tkinter) ─────────────────────────────
    npz_path = None
    if feature_list:
        try:
            features_array = to_cpu(np.vstack([to_cpu(f) for f in feature_list]))
            labels_array   = np.array(label_list)

            # ── Save directly inside embeddings/ (no subfolders) ──────────────────────────
            embeddings_dir = Path("embeddings")
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            npz_save_path = embeddings_dir / (
                f"{st.session_state.selected_model_name}_"
                f"{layer_name}_{transform_choice}"
                f"_all_classes_{dataset_name}.npz"
            )
            
            # Overwrite if already exists
            if npz_save_path.exists():
                npz_save_path.unlink()
            np.savez(npz_save_path, features=features_array, labels=labels_array)
            npz_path = str(npz_save_path.resolve())

            st.session_state.npz_file_path = npz_path
            with open("logs/last_npz_path.txt", "w") as f:
                f.write(npz_path)
            logging.debug(f"Saved NPZ file: {npz_path}")

            # Cache bytes for download button (no file re-read needed later)
            buf = io.BytesIO()
            np.savez(buf, features=features_array, labels=labels_array)
            st.session_state.extracted_npz_bytes = buf.getvalue()
            st.session_state.extracted_npz_name  = npz_save_path.name

        except Exception as e:
            logging.error(f"Failed to save .npz file: {e}")
            st.error(f"Failed to save .npz file: {e}")

    return feature_list, label_list, failed, npz_path


def resolve_npz_path():
    """
    Mirrors the tkinter run_visualisation() fallback logic:
    1. Use session-state path.
    2. Read last_npz_path.txt.
    3. Find the most-recently-modified .npz under features/.
    """
    path = st.session_state.npz_file_path

    if not path:
        txt = Path("logs/last_npz_path.txt")
        if txt.exists():
            path = txt.read_text().strip()
            st.session_state.npz_file_path = path

    if not path or not Path(path).exists():
        npz_files = glob("embeddings/**/*.npz", recursive=True)
        if npz_files:
            path = max(npz_files, key=os.path.getmtime)
            st.session_state.npz_file_path = path

    return path if path and Path(path).exists() else None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    if GPU_AVAILABLE:
        st.success("Running on GPU ⚡")
    else:
        st.info("Running on CPU 💻")

    st.session_state.download_weights = st.checkbox(
        "Download pretrained weights",
        value=st.session_state.download_weights,
        help="Uncheck to load architecture without ImageNet weights (random init)."
    )

    st.markdown("---")
    st.markdown("**Quick Navigation**")
    st.markdown("- Step 1 → Select Model\n- Step 2 → Architecture & Extract\n- Step 3 → Embedding Analysis")

    # Show currently loaded model info
    if st.session_state.selected_model_name:
        st.markdown("---")
        st.success(f"**Loaded:** {st.session_state.selected_model_name}")
    if st.session_state.npz_file_path:
        st.info(f"**Feature file ready**\n`{Path(st.session_state.npz_file_path).name}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TITLE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🔬 Model Feature Extractor")
st.markdown("---")







# ─── Check existing embeddings (.npz files) ────────────────────────────────────
existing_npz_files = glob("embeddings/**/*.npz", recursive=True)

if existing_npz_files:
    st.success(f"📦 Found {len(existing_npz_files)} existing embedding file(s)")
    
    with st.expander("View existing embeddings"):
        for f in existing_npz_files:
            st.write(f"• {Path(f).name}")
else:
    st.info("No existing embeddings found.")
# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.header("📋 Step 1: Select Model")

mode = st.radio(
    "Model Mode:",
    ["Pretrained", "Custom (.py + .pth)"],
    horizontal=True,
    key="mode_radio"
)

# ── Pretrained Branch ──────────────────────────────────────────────────────────
if mode == "Pretrained":
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        model_name = st.selectbox(
            "Select Pretrained Model:",
            ["— select —"] + sorted(MODELS_DICT.keys()),
            key="pretrained_selectbox"
        )
    with col_btn:
        st.write("")          # vertical alignment spacer
        st.write("")
        load_clicked = st.button("Load Model", key="load_pretrained_btn",
                                 disabled=(model_name == "— select —"))

    if load_clicked and model_name != "— select —":
        with st.spinner(f"Loading {model_name} …"):
            try:
                select_pretrained_model(model_name, st.session_state.download_weights)
                st.session_state.layer_shapes = compute_layer_shapes(
                    st.session_state.selected_model, "Pretrained", 224, 224, True
                )
                st.session_state.architecture_loaded = True
                st.session_state.weights_loaded = True
                st.success(f"✅ Loaded pretrained model: **{model_name}**  "
                           f"({len(st.session_state.layer_shapes)} layers found)")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    # Custom-input fields not used for pretrained; set safe defaults
    custom_width  = 224
    custom_height = 224
    use_rgb       = True

# ── Custom Branch ──────────────────────────────────────────────────────────────
else:
    col_dims, col_files = st.columns(2)

    with col_dims:
        st.subheader("Input Dimensions")
        size_options = ["32", "64", "128", "224", "256", "299", "384"]
        custom_width  = int(st.selectbox("Width:",  size_options, index=3, key="cw"))
        custom_height = int(st.selectbox("Height:", size_options, index=3, key="ch"))
        use_rgb = st.checkbox("Use RGB Channel", value=True, key="use_rgb_cb")

    with col_files:
        st.subheader("Model Files")
        model_py_file  = st.file_uploader(
            "① Select Model Definition (.py)", type=["py"], key="py_uploader"
        )
        model_pth_file = st.file_uploader(
            "② Load Model Weights (.pth)", type=["pth"], key="pth_uploader"
        )

    # ── Load architecture ──────────────────────────────────────────────────────
    if model_py_file is not None:
        if st.button("Load Architecture", key="load_arch_btn"):
            with st.spinner("Importing model architecture …"):
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".py", delete=False
                    ) as tmp:
                        tmp.write(model_py_file.read())
                        tmp_path = tmp.name

                    module_name = Path(model_py_file.name).stem
                    spec = importlib.util.spec_from_file_location(module_name, tmp_path)
                    model_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = model_module
                    spec.loader.exec_module(model_module)

                    # Collect class names (mirror tkinter choose_model_class)
                    class_names = [
                        attr for attr in dir(model_module)
                        if isinstance(getattr(model_module, attr), type)
                           and attr not in dir(__builtins__)
                    ]

                    if not class_names:
                        st.error("No model class found in the uploaded file.")
                    else:
                        st.session_state.custom_model_classes = class_names
                        st.session_state.custom_module = model_module
                        st.session_state.model_file_path = tmp_path
                        st.session_state.architecture_loaded = False  # need class pick
                        st.success("File imported. Choose a class below and click **Confirm**.")
                except Exception as e:
                    logging.error(f"Failed to import model: {e}")
                    st.error(f"Failed to import model: {str(e)}")

    # Class picker (mirrors tkinter choose_model_class dialog)
    if st.session_state.custom_model_classes:
        chosen_class = st.selectbox(
            "Select Model Class:",
            st.session_state.custom_model_classes,
            key="class_picker"
        )
        if st.button("Confirm Class Selection", key="confirm_class_btn"):
            try:
                model_cls = getattr(st.session_state.custom_module, chosen_class)
                st.session_state.selected_model = model_cls().to(device)
                st.session_state.selected_model_name = chosen_class
                st.session_state.npz_file_path = None
                st.session_state.architecture_loaded = True
                st.session_state.weights_loaded = False
                logging.debug(f"Loaded custom model: {chosen_class}")

                # Show class definitions (mirrors show_model_file_contents)
                class_content = show_model_file_contents(st.session_state.model_file_path)
                if class_content:
                    with st.expander("📄 Class definitions found in file"):
                        st.code(class_content, language="python")

                st.success(f"✅ Architecture **{chosen_class}** loaded. Now upload and load weights.")
            except Exception as e:
                st.error(f"Failed to instantiate model: {e}")

    # ── Load weights ───────────────────────────────────────────────────────────
    if (model_pth_file is not None
            and st.session_state.selected_model is not None
            and st.session_state.architecture_loaded):

        if st.button("Load Weights", key="load_weights_btn"):
            with st.spinner("Loading weights …"):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pth", delete=False
                    ) as tmp:
                        tmp.write(model_pth_file.read())
                        tmp_path = tmp.name

                    st.session_state.selected_model.load_state_dict(
                        torch.load(tmp_path, map_location=device)
                    )
                    st.session_state.selected_model.eval()
                    st.session_state.npz_file_path = None
                    st.session_state.weights_loaded = True

                    # Compute layer shapes after weights loaded
                    st.session_state.layer_shapes = compute_layer_shapes(
                        st.session_state.selected_model, "Custom",
                        custom_height, custom_width, use_rgb
                    )

                    logging.debug(f"Loaded weights for {st.session_state.selected_model_name}")
                    os.unlink(tmp_path)

                    st.success(
                        f"✅ Weights loaded for **{st.session_state.selected_model_name}**  "
                        f"({len(st.session_state.layer_shapes)} layers found)"
                    )
                except Exception as e:
                    logging.error(f"Failed to load weights: {e}")
                    st.error(f"Failed to load model weights: {str(e)}")

    # Compute layer shapes even without weights (random init) if arch is ready
    if (st.session_state.architecture_loaded
            and not st.session_state.layer_shapes
            and st.session_state.selected_model is not None):
        st.session_state.layer_shapes = compute_layer_shapes(
            st.session_state.selected_model, "Custom",
            custom_height, custom_width, use_rgb
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MODEL ARCHITECTURE & FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🏗️ Step 2: View Model Architecture & Extract Features")

if st.session_state.selected_model is None:
    st.info("ℹ️ Please load / select a model in Step 1 first.")
else:
    st.caption("**Layer shape convention:** [Batch, Features]")

    # Re-compute on demand if empty
    if not st.session_state.layer_shapes:
        if mode == "Pretrained":
            st.session_state.layer_shapes = compute_layer_shapes(
                st.session_state.selected_model, "Pretrained", 224, 224, True
            )
        else:
            st.session_state.layer_shapes = compute_layer_shapes(
                st.session_state.selected_model, "Custom",
                custom_height, custom_width, use_rgb
            )

    if not st.session_state.layer_shapes:
        st.warning("No layers found in the model.")
    else:
        # ── Layer browser (mirrors the scrollable architecture window) ──────────
        layer_names = list(st.session_state.layer_shapes.keys())

        with st.expander(
            f"📊 Model Layers — {st.session_state.selected_model_name} "
            f"({len(layer_names)} layers)",
            expanded=True
        ):
            # Searchable layer list
            search_q = st.text_input(
                "🔍 Filter layers:", placeholder="e.g. conv, relu, fc …",
                key="layer_search"
            )
            filtered_layers = (
                [l for l in layer_names if search_q.lower() in l.lower()]
                if search_q else layer_names
            )

            layer_display = {
                l: f"{l}  →  [{', '.join(map(str, st.session_state.layer_shapes[l]))}]"
                for l in filtered_layers
            }

            if not filtered_layers:
                st.warning("No layers match your filter.")
                selected_layer = None
            else:
                selected_layer = st.selectbox(
                    "Select a layer to extract features from:",
                    options=filtered_layers,
                    format_func=lambda x: layer_display[x],
                    key="layer_selectbox"
                )

        # ── Extraction configuration (mirrors the extract_features dialog flow) ──
        if selected_layer:
            st.subheader("⚙️ Feature Extraction Configuration")
            col_cfg1, col_cfg2 = st.columns(2)

            with col_cfg1:
                dataset_source = st.radio(
                    "Dataset Source:", ["Local", "Standard"],
                    key="dataset_src", horizontal=True
                )
                transform_choice = st.selectbox(
                    "Transformation:", ["Resize", "Crop", "None"],
                    key="transform_sel"
                )

            with col_cfg2:
                if dataset_source == "Standard":
                    dataset_name = st.selectbox(
                        "Standard Dataset:",
                        ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "STL10"],
                        key="std_dataset"
                    )
                    uploaded_images = None
                else:
                    dataset_name = "Local"
                    uploaded_images = st.file_uploader(
                        "Upload Images (jpg/jpeg/png) — select multiple",
                        type=["jpg", "jpeg", "png"],
                        accept_multiple_files=True,
                        key="img_uploader"
                    )
                    if uploaded_images:
                        st.caption(f"{len(uploaded_images)} file(s) selected")

                    # Optional: label each image via a folder-name convention hint
                    st.caption(
                        "💡 Tip: name files as `<class>_<anything>.jpg` so labels "
                        "can be inferred automatically, or all will be labeled *unknown*."
                    )

            # Check if features already exist (mirrors tkinter overwrite check)
            expected_npz = (
                Path("embeddings")
                / st.session_state.selected_model_name
                / f"{selected_layer} ({transform_choice})"
                / (f"{st.session_state.selected_model_name}_"
                   f"{selected_layer}_{transform_choice}"
                   f"_all_classes_{dataset_name}.npz")
            )

            if expected_npz.exists():
                st.warning(f"⚠️ Feature file already exists: `{expected_npz.name}`")
                overwrite = st.radio(
                    "What would you like to do?",
                    ["Use existing features", "Re-extract (overwrite)"],
                    key="overwrite_radio"
                )
                if overwrite == "Use existing features":
                    if st.button("✅ Use Existing Features", key="use_existing_btn"):
                        st.session_state.npz_file_path = str(expected_npz.resolve())
                        with open("logs/last_npz_path.txt", "w") as f:
                            f.write(st.session_state.npz_file_path)
                        # Cache bytes for download
                        st.session_state.extracted_npz_bytes = expected_npz.read_bytes()
                        st.session_state.extracted_npz_name  = expected_npz.name
                        st.session_state.extraction_done = True
                        st.success("✅ Using existing feature file. Proceed to Step 3.")
                    extract_disabled = True
                else:
                    extract_disabled = False
            else:
                extract_disabled = False

            # ── Extract button ───────────────────────────────────────────────────
            if not extract_disabled:
                can_extract = (
                    (dataset_source == "Standard") or
                    (dataset_source == "Local" and uploaded_images)
                )
                if st.button(
                    "🚀 Extract Features from Selected Layer",
                    type="primary",
                    key="extract_btn",
                    disabled=not can_extract
                ):
                    target_size = (
                        (224, 224) if mode == "Pretrained"
                        else (custom_height, custom_width)
                    )
                    use_rgb_final = True if mode == "Pretrained" else use_rgb

                    feat_list, lbl_list, failed, npz_path = extract_features(
                        layer_name=selected_layer,
                        dataset_source=dataset_source,
                        transform_choice=transform_choice,
                        dataset_name=dataset_name,
                        uploaded_images=uploaded_images,
                        use_rgb=use_rgb_final,
                        target_size=target_size,
                        mode_value=mode
                    )

                    st.session_state.success_count = len(feat_list)
                    st.session_state.failed_images = failed
                    st.session_state.extraction_done = npz_path is not None

                    if feat_list:
                        st.success(
                            f"✅ Extracted features for **{len(feat_list)}** images  "
                            f"from layer **{selected_layer}**."
                        )
                    if failed:
                        with st.expander(f"⚠️ {len(failed)} image(s) failed"):
                            st.write("\n".join(failed))

            # ── Post-extraction download ─────────────────────────────────────────
            if st.session_state.extraction_done and st.session_state.extracted_npz_bytes:
                st.download_button(
                    label="⬇️ Download Feature File (.npz)",
                    data=st.session_state.extracted_npz_bytes,
                    file_name=st.session_state.extracted_npz_name,
                    mime="application/octet-stream",
                    key="dl_npz_btn"
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — EMBEDDING ANALYSIS  (mirrors run_visualisation())
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🔍 Step 3: Embedding Analysis")

col_d, col_a, col_m = st.columns(3)
with col_d:
    dim_option      = st.selectbox("Dimension:",          ["2D", "3D"],              key="dim_sel")
with col_a:
    algo_option     = st.selectbox("Reduction Algorithm:", ["PCA", "TruncatedSVD"],  key="algo_sel")
with col_m:
    distance_option = st.selectbox(
        "Distance Metric:",
        ["euclidean", "cosine", "cityblock", "canberra"],
        key="dist_sel"
    )

run_viz = st.button("📊 Run Embedding Analysis", type="primary", key="run_viz_btn")

if run_viz:
    npz_path = resolve_npz_path()

    if not npz_path:
        st.error(
            "No valid feature file found. "
            "Please extract features in Step 2 first!"
        )
    else:
        # Store params in session state (consumed by visualisation_app)
        st.session_state.viz_dim      = dim_option
        st.session_state.viz_algo     = algo_option
        st.session_state.viz_distance = distance_option

        npz_path_obj = Path(npz_path)
        st.success(f"✅ Feature file ready: `{npz_path_obj.name}`")

        # ── Inline preview using numpy + matplotlib ──────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA, TruncatedSVD
            from sklearn.preprocessing import LabelEncoder

            data = np.load(npz_path)
            features = data["features"]
            labels   = data["labels"]

            le = LabelEncoder()
            encoded = le.fit_transform(labels)
            n_classes = len(le.classes_)

            n_components = 3 if dim_option == "3D" else 2
            if algo_option == "PCA":
                reducer = PCA(n_components=n_components)
            else:
                reducer = TruncatedSVD(n_components=n_components)

            with st.spinner("Running dimensionality reduction …"):
                reduced = reducer.fit_transform(features)

            cmap = plt.cm.get_cmap("tab20", n_classes)
            fig  = plt.figure(figsize=(10, 7))

            if dim_option == "3D":
                ax = fig.add_subplot(111, projection="3d")
                for cls_idx, cls_name in enumerate(le.classes_):
                    mask = encoded == cls_idx
                    ax.scatter(
                        reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                        label=cls_name, color=cmap(cls_idx), alpha=0.7, s=20
                    )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            else:
                ax = fig.add_subplot(111)
                for cls_idx, cls_name in enumerate(le.classes_):
                    mask = encoded == cls_idx
                    ax.scatter(
                        reduced[mask, 0], reduced[mask, 1],
                        label=cls_name, color=cmap(cls_idx), alpha=0.7, s=20
                    )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")

            ax.set_title(
                f"{algo_option} ({dim_option}) — "
                f"{st.session_state.selected_model_name or npz_path_obj.stem}"
            )
            ax.legend(
                loc="upper right",
                fontsize=7,
                ncol=max(1, n_classes // 10),
                markerscale=1.5
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Summary stats ────────────────────────────────────────────────────
            st.markdown("**Dataset summary**")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total samples",   len(features))
            col_s2.metric("Classes",          n_classes)
            col_s3.metric("Feature dim",      features.shape[1])

            if algo_option == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
                evr = reducer.explained_variance_ratio_
                st.info(
                    f"Explained variance — "
                    f"PC1: {evr[0]:.1%}, "
                    f"PC2: {evr[1]:.1%}"
                    + (f", PC3: {evr[2]:.1%}" if len(evr) > 2 else "")
                )

        except ImportError as ie:
            st.warning(
                f"Install scikit-learn and matplotlib for inline preview: `{ie}`\n\n"
                "Run `streamlit run visualisation_app.py` for the full visualiser."
            )
        except Exception as e:
            logging.error(f"Inline viz error: {e}")
            st.error(f"Visualisation error: {e}")

        # ── Instructions for the separate visualisation app ──────────────────────
        with st.expander("ℹ️ Open in the full Visualisation App"):
            st.code(
                f"# Terminal command:\n"
                f"streamlit run visualisation_app.py",
                language="bash"
            )
            st.markdown(
                f"The feature file will be picked up automatically from  \n"
                f"`logs/last_npz_path.txt` → `{npz_path}`"
            )

        # ── Download the npz if not already shown ────────────────────────────────
        if not st.session_state.extracted_npz_bytes:
            with open(npz_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Feature File (.npz)",
                    data=f,
                    file_name=npz_path_obj.name,
                    mime="application/octet-stream",
                    key="dl_npz_viz_btn"
                )

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "CNN Feature Extractor & Embedding Analysis Tool · Powered by PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)