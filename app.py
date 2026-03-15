import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer AI & XAI", layout="wide", page_icon="🎗️")

# Custom CSS for UI improvement
st.markdown("""
    <style>
    .main { background-color: #f7f9fa; }
    h1 { color: #1f3a93; }
    .stButton>button { background-color: #1f3a93; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #3f51b5; color: white; }
    .disclaimer { font-size: 14px; color: #666; font-style: italic; background-color: #ffe0e0; padding: 10px; border-radius: 5px; border-left: 5px solid red;}
    </style>
""", unsafe_allow_html=True)

st.title("🎗️ Transfer Learning-Based Breast Cancer Detection")
st.markdown("### With Dual Explainability: A Comparative Study of Grad-CAM and SHAP")

st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b> This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or oncologist.
</div>
""", unsafe_allow_html=True)
st.write("---")

# Gracefully Handle Missing TensorFlow (Because user is on Python 3.14 locally)
try:
    import tensorflow as tf
    import cv2
    import shap
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    st.error(f"⚠️ Machine Learning Libraries Not Found: {e}")
    st.warning("You are likely running Python 3.14 locally, which does not support TensorFlow yet. To see the fully working app, you must deploy this to **Streamlit Cloud** (which uses Python 3.10) where it will run beautifully.")

# --- Helper Functions for XAI ---
@st.cache_resource
def load_ml_model():
    model_path = "best_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_gradcam(model, img_array, intensity=0.5, res=250):
    # Find the last convolutional layer dynamically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: # Is a convolutional feature map
            last_conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Overlay with original image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_og = np.uint8(255 * img_array[0])
    superimposed_img = heatmap_colored * intensity + img_og
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def generate_shap(model, img_array):
    # SHAP requires background data. We'll use completely black images as reference.
    background = np.zeros((5, 224, 224, 3))
    
    e = shap.GradientExplainer(model, background)
    shap_values, indexes = e.shap_values(img_array, ranked_outputs=1)
    
    # Plotting SHAP values
    fig = plt.figure()
    shap.image_plot(shap_values, img_array, show=False)
    # Get the image plot from matplotlib
    fig = plt.gcf()
    return fig

# --- Main App ---
st.header("Upload Image (Mammogram or Histopathology)")
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and TF_AVAILABLE:
    image = Image.open(uploaded_file)
    model = load_ml_model()
    
    col_img, col_proc = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col_proc:
        if model is None:
            st.error("❌ 'best_model.h5' not found! Please make sure the model is in the project directory.")
        else:
            st.write("### Analysis Progress")
            if st.button("Run EfficientNetB3 Inference & XAI"):
                # Inference
                with st.spinner("Processing image through EfficientNetB3..."):
                    img_array = preprocess_image(image)
                    preds = model.predict(img_array)
                    confidence = float(np.max(preds) * 100)
                    class_idx = np.argmax(preds)
                    
                    # Based on standard flow: 0=benign, 1=malignant
                    classes = ["Benign", "Malignant"]
                    result_class = classes[class_idx]
                    color = "green" if result_class == "Benign" else "red"

                # Grad-CAM
                with st.spinner("Generating Grad-CAM Heatmap..."):
                    gradcam_img = generate_gradcam(model, img_array)

                # SHAP
                with st.spinner("Running SHAP GradientExplainer..."):
                    try:
                        shap_fig = generate_shap(model, img_array)
                    except Exception as e:
                        shap_fig = None
                        shap_error = e

                st.success("Analysis Complete!")
                
                # --- Results Display ---
                st.markdown("---")
                st.subheader("Classification Result")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"### Result: <span style='color:{color};'>**{result_class}**</span>", unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"### Confidence: **{confidence:.2f}%**")
                
                st.markdown("---")
                st.subheader("Explainability (XAI) Comparison")
                xai_col1, xai_col2 = st.columns(2)
                
                with xai_col1:
                    st.markdown("**Grad-CAM (Spatial Localization)**")
                    st.image(gradcam_img, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
                    st.caption("Heatmap shows exactly what structural features the CNN focused on.")
                    
                with xai_col2:
                    st.markdown("**SHAP (Feature Attribution)**")
                    if shap_fig:
                        st.pyplot(shap_fig, clear_figure=True)
                        st.caption("Red pixels pushed the model towards Malignant; Blue towards Benign.")
                    else:
                        st.error(f"SHAP Error: {shap_error}")
                
                st.markdown("---")
                st.info("📄 PDF Report Generation coming soon in final deployment!")
