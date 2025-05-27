import streamlit as st
import torch
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch.nn.functional as F
import os
import math
from tensorflow.keras.applications import VGG16
import gdown
import tensorflow.compat.v1 as tf_compat

# --- Constants ---
IMG_SIZE = (224, 224)
PATCH_SIZE = 16
VIT_MODEL_PATH = "./vit_corn_model.pth"
CNN_MODEL_PATH = "./cnn_corn_model.h5"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA_DIR = "./corn_data/train"
NUM_PATCHES = (IMG_SIZE[0] // PATCH_SIZE) ** 2

# Google Drive IDs for model files
VIT_MODEL_GDRIVE_ID = "1_Rm1-AnxrRXaSmCGRYafWVQZGx79rXBD"
CNN_MODEL_GDRIVE_ID = "1XFLVxaT222PHNKkHONS7zzVoVxKOu4ML"

CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- Model Loading ---
@st.cache_resource
def load_vit_model(model_type, num_classes=len(CLASS_NAMES)):
    if model_type != "ViT":
        raise ValueError("This function only loads ViT models")
    if not os.path.exists(VIT_MODEL_PATH):
        st.info(f"ViT model not found locally. Downloading from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={VIT_MODEL_GDRIVE_ID}", VIT_MODEL_PATH, quiet=False)
            st.success("ViT model has downloaded successfully. ViT 模型已成功下载。")
        except Exception as e:
            raise RuntimeError(f"Failed to download ViT model: {str(e)}")
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        raise RuntimeError(f"Failed to load ViT model weights: {str(e)}")
    model.to(DEVICE)
    model.eval()
    st.write(f"ViT model has downloaded successfully. ViT 模型已成功下载。")

    # Initialize fresh attention weights
    attention_weights = [[] for _ in range(len(model.blocks))]
    def get_hook_fn(block_idx):
        def hook_fn(module, input, output):
            try:
                x = input[0]
                B, N, C = x.shape
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                attention_weights[block_idx].append(attn.detach())
            except Exception as e:
                st.warning(f"Error capturing attention weights for block {block_idx}: {str(e)}")
                dummy_attn = torch.ones((B, module.num_heads, N, N), device=x.device) / N
                attention_weights[block_idx].append(dummy_attn)
        return hook_fn

    for i in range(len(model.blocks)):
        model.blocks[i].attn.register_forward_hook(get_hook_fn(i))
    
    return model, attention_weights, "ViT"

def load_cnn_model(model_type, num_classes=len(CLASS_NAMES)):
    if model_type != "CNN":
        raise ValueError("This function only loads CNN models")
    if not os.path.exists(CNN_MODEL_PATH):
        st.info(f"CNN model not found locally. Downloading from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={CNN_MODEL_GDRIVE_ID}", CNN_MODEL_PATH, quiet=False)
            st.success("CNN model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download CNN model: {str(e)}")
    
    try:
        # Reset TensorFlow session and graph
        tf.keras.backend.clear_session()
        tf_compat.reset_default_graph()
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
        st.write(f"CNN model has loaded successfully. CNN 模型已成功加载。")

        # Create VGG16 feature extractor for block5_conv3
        base_vgg16 = VGG16(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        feature_model = tf.keras.Model(
            inputs=base_vgg16.input,
            outputs=base_vgg16.get_layer('block5_conv3').output
        )
        # Transfer weights from the loaded model's vgg16 layer
        loaded_vgg16_weights = model.get_layer('vgg16').get_weights()
        feature_model.set_weights(loaded_vgg16_weights)
        
        # Initialize models with dummy input
        dummy_input = tf.zeros((1, *IMG_SIZE, 3))
        _ = model(dummy_input)
        _ = feature_model(dummy_input)
        
        return model, feature_model, "CNN"  # model for prediction, feature_model for visualization
    except Exception as e:
        raise RuntimeError(f"Failed to load CNN model: {str(e)}")

# --- Image Preprocessing ---
def preprocess_image_vit(image):
    """Preprocesses image for ViT model (PyTorch)."""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor

def preprocess_image_cnn(image):
    """Preprocesses image for CNN model (TensorFlow), matching evaluation script."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype('float32')
    image_array = image_array / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- Visualization Functions ---
def visualize_patches(image):
    """Visualizes how the input image is divided into patches for ViT."""
    image = image.convert('RGB').resize(IMG_SIZE)
    image_np = np.array(image)
    patches = []
    for i in range(0, IMG_SIZE[0], PATCH_SIZE):
        for j in range(0, IMG_SIZE[1], PATCH_SIZE):
            patch = image_np[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patches.append(patch)
    
    # Reduced figure size and grid for compactness
    fig = plt.figure(figsize=(6, 6), facecolor='none')
    ax = fig.add_subplot(111)
    ax.imshow(image_np)  # Show full image with patch grid
    ax.axis('off')
    
    # Draw grid lines to indicate patches
    for i in range(0, IMG_SIZE[0], PATCH_SIZE):
        ax.axhline(i, color='#E0E0E0', linestyle='--', linewidth=0.5)
        ax.axvline(i, color='#E0E0E0', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(pad=0.5)
    return fig

def visualize_attention_map(attention_weights, image_shape=(224, 224), image=None):
    """
    Visualizes attention maps from ViT transformer blocks, overlaid on the input image.
    If no attention weights are captured, uniform maps are used.
    """
    if not any(attention_weights) or all(not block_weights for block_weights in attention_weights):
        attn_maps = [np.ones((14, 14)) / 196 for _ in range(12)] # Default for 12 blocks, 14x14 patches
    else:
        attn_maps = []
        for block_idx, block_weights in enumerate(attention_weights):
            if not block_weights or block_weights[-1] is None:
                attn_maps.append(np.ones((14, 14)) / 196) # Uniform if specific block fails
                continue
            
            try:
                attn = block_weights[-1].mean(dim=1)[0]  # [num_patches+1, num_patches+1]
                if attn.dim() != 2 or attn.shape[0] < 2:
                    attn_maps.append(np.ones((14, 14)) / 196) # Uniform if shape is incorrect
                    continue
                attn = attn[1:, 1:].mean(dim=0)[:196].reshape(14, 14) # Exclude CLS token, reshape
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8) # Normalize to [0, 1]
                attn_maps.append(attn.cpu().numpy())
            except Exception:
                attn_maps.append(np.ones((14, 14)) / 196) # Uniform on error

    fig, axes = plt.subplots(4, 3, figsize=(12, 16)) # Assuming 12 transformer blocks
    axes = axes.flatten()
    image_np = np.array(image.resize(image_shape))
    
    for idx, (attn, ax) in enumerate(zip(attn_maps, axes)):
        # Resize attention map to image size for overlay
        attn_resized = F.interpolate(
            torch.tensor(attn).unsqueeze(0).unsqueeze(0),
            size=image_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        ax.imshow(image_np)
        ax.imshow(attn_resized, cmap='jet', alpha=0.5) # Overlay with transparency
        ax.axis('off')
        ax.set_title(f'Block {idx+1} Attention')
    
    plt.tight_layout()
    return fig

def visualize_feature_maps(feature_model, preprocessed_image_array, num_maps_to_show=16):
    """
    Visualizes feature maps from the VGG16 feature extractor's block5_conv3 layer.
    """
    if feature_model is None:
        st.error("Feature model is not available. Visualization will be skipped.")
        return None

    try:
        # Verify feature model output shape
        output_shape = feature_model.output.shape
        if len(output_shape) != 4:
            st.error("Feature maps are not in the expected 4D format. Visualization will be skipped.")
            return None

        # Generate feature maps
        feature_maps = feature_model.predict(preprocessed_image_array, verbose=0)
        feature_maps = np.squeeze(feature_maps, axis=0)
        total_channels = feature_maps.shape[-1]
        
        num_maps_to_show = min(num_maps_to_show, total_channels)
        display_maps = feature_maps[:, :, :num_maps_to_show]

        cols = 4
        rows = math.ceil(num_maps_to_show / cols)

        fig = plt.figure(figsize=(cols * 3, rows * 3), facecolor='none')
        axes = fig.subplots(rows, cols)
        axes = axes.flatten()

        for i in range(num_maps_to_show):
            ax = axes[i]
            ax.imshow(display_maps[:, :, i], cmap='viridis')
            ax.set_title(f"Map {i+1}", color='#E0E0E0', fontsize=8)
            ax.axis('off')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=0.5)
        return fig

    except Exception as e:
        st.error(f"An error occurred while generating feature maps: {e}. Visualization will be skipped.")
        return None
    
# --- Prediction Functions ---
def predict_vit(image_tensor, model):
    """Performs prediction using the ViT model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100
    return predicted_class, confidence, probabilities

def predict_cnn(image_array, model):
    """Performs prediction using the CNN model."""
    try:
        outputs = model.predict(image_array, verbose=0)[0]
        probabilities = tf.nn.softmax(outputs).numpy()
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100
    except Exception as e:
        st.error(f"Error during CNN prediction: {str(e)}")
        return None, None, None
    return predicted_class, confidence, probabilities

# --- Probability Visualization ---
def plot_probabilities(probabilities, class_names):
    """Visualizes the model's class probability distribution with a transparent background."""
    import numpy as np
    import matplotlib.pyplot as plt

    probabilities = np.array(probabilities)
    paired = list(zip(probabilities, class_names))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    sorted_probs, sorted_names = zip(*paired_sorted)
    sorted_probs = np.array(sorted_probs)
    sorted_names = list(sorted_names)
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(8, 4), facecolor='none')  # Transparent figure background
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')  # Transparent axes background
    
    y_pos = np.arange(len(sorted_names))[::-1]
    bars = ax.barh(y_pos, sorted_probs * 100, color='skyblue', height=0.4)
    
    for bar in bars:
        width = bar.get_width().item() if isinstance(bar.get_width(), np.ndarray) else bar.get_width()
        # Place text inside bar for high probabilities, outside for lower ones
        if width >= 80:  # Threshold for placing text inside
            text_x = width - 10  # Position text 10 units inside the bar
            ha = 'right'  # Right-align text inside bar
            text_color = 'white'  # White text for contrast inside bar
        else:
            text_x = width + 2  # Small offset outside bar
            ha = 'left'  # Left-align text outside bar
            text_color = '#E0E0E0'  # Light gray for visibility on dark background
        ax.text(text_x, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%', 
                va='center', ha=ha, fontsize=10, color=text_color)
    
    bars[0].set_color('orange')  # Highlight the highest probability
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, color='#E0E0E0')  # Light gray for y-axis labels
    ax.set_xlabel('Confidence (%)', color='#E0E0E0')  # Light gray for x-axis label
    ax.set_title('Class Probability Distribution', color='#E0E0E0')  # Light gray for title
    ax.set_xlim(0, 110)  # Keep x-limit to 110 for consistency
    
    # Set axis spines (border lines) to light gray for visibility
    for spine in ax.spines.values():
        spine.set_color('#E0E0E0')
    
    # Set tick colors to light gray
    ax.tick_params(axis='x', colors='#E0E0E0')
    ax.tick_params(axis='y', colors='#E0E0E0')
    
    plt.tight_layout(pad=1.0)  # Add padding to prevent clipping
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Corn Disease Classification and Model Insights")
    st.write("Upload a corn leaf image to classify diseases and visualize how the model processes the input.")

    # Initialize session state
    if 'last_upload' not in st.session_state:
        st.session_state.last_upload = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None

    model_type = st.sidebar.selectbox("Select Model for Analysis", ["ViT", "CNN"])
    uploaded_file = st.sidebar.file_uploader("Upload a corn leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file != st.session_state.last_upload or model_type != st.session_state.model_type:
            st.session_state.last_upload = uploaded_file
            st.session_state.model_type = model_type
            st.cache_data.clear()
            st.cache_resource.clear()
            st.write("Cache cleared for new upload or model switch")

        st.markdown("### 1. Input Image <span style='font-size:0.6em;'>(输入图像, 入力画像, 입력 이미지, صورة الإدخال)</span>", unsafe_allow_html=True)
        st.write("The uploaded corn leaf image is resized to 224x224 pixels for model processing.")
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception:
            st.error("Invalid image file. Please upload a valid JPG or PNG image.")
            return
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Original Uploaded Image', width=200)
        with col2:
            st.image(image.resize(IMG_SIZE), caption=f'Resized to {IMG_SIZE[0]}x{IMG_SIZE[1]}', width=200)

        st.markdown("### 2. Model Loading <span style='font-size:0.6em;'>(模型加载, モデル読み込み, 모델 로딩, تحميل النموذج)</span>", unsafe_allow_html=True)
        st.write(f"Loading the selected {model_type} model.")
        try:
            if model_type == "ViT":
                model, aux_data, loaded_type = load_vit_model(model_type)
            else:
                model, aux_data, loaded_type = load_cnn_model(model_type)
        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            return

        if model_type == "ViT":
            st.markdown("### 3. Vision Transformer Processing <span style='font-size:0.5em;'>(视觉转换器处理, ビジョントランスフォーマ処理, 비전 트랜스포머 처리, معالجة المحول البصري)</span>", unsafe_allow_html=True)
            with st.expander("3.1 Patch Embedding", expanded=False):
                st.write("The ViT divides the image into 16x16 patches, shown as a grid overlay.")
                patch_fig = visualize_patches(image)
                st.pyplot(patch_fig)

            image_tensor = preprocess_image_vit(image)
            predicted_class, confidence, probabilities = predict_vit(image_tensor, model)

            with st.expander("3.2 Transformer Blocks & Attention Mechanism", expanded=False):
                st.write("These layers process patch embeddings, using self-attention to highlight important regions.")
                attn_fig = visualize_attention_map(aux_data, IMG_SIZE, image)
                if attn_fig:
                    st.pyplot(attn_fig)
                else:
                    st.warning("Attention map visualization is not available for this model.")

        elif model_type == "CNN":
            st.markdown("### 3. Convolutional Neural Network Processing <span style='font-size:0.5em;'>(卷积神经网络处理, 畳み込みニューラルネットワーク処理, 합성곱 신경망 처리, معالجة الشبكة العصبية الالتفافية)</span>", unsafe_allow_html=True)
            with st.expander("3.1 Feature Map Extraction", expanded=False):
                st.write("The CNN extracts hierarchical features through convolutional layers.")
                image_array = preprocess_image_cnn(image)
                feature_fig = visualize_feature_maps(aux_data, image_array, num_maps_to_show=16)
                if feature_fig:
                    st.pyplot(feature_fig)
                else:
                    st.warning("Feature map visualization is not available for this model.")

            # Fresh preprocessing and prediction
            image_array = preprocess_image_cnn(image)
            predicted_class, confidence, probabilities = predict_cnn(image_array, model)
            if predicted_class is None:
                return

        st.markdown("""### 4. Classification Result <span style='font-size:0.6em;'>(分类结果, 分類結果, 분류 결과, نتيجة التصنيف)</span>""", unsafe_allow_html=True)
        st.write("The model's final output is a probability distribution across all possible disease classes.")
        if probabilities is not None:
            st.success(f"**Predicted Class**: **{predicted_class}** with **{confidence:.2f}%** confidence.")
            prob_fig = plot_probabilities(probabilities, CLASS_NAMES)
            st.pyplot(prob_fig)
        else:
            st.error("Prediction failed. Please try again.")

if __name__ == "__main__":
    main()
