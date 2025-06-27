import streamlit as st
import tensorflow as tf
import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import json
import os
import gdown

st.set_page_config(layout="wide", page_title="Visualisasi ResNet50 Sampah")
st.title("🧠 Visualisasi ResNet50 - Klasifikasi Sampah")

# --- Unduh model dari Google Drive jika belum ada ---
def download_model_from_drive():
    file_id = "1IvJ4MMfz9kai02MZDhHvUAW7UkhCYh9w"  # ID model di Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "klasifikasi_sampah_menggunakan_ResNet-50.h5"

    if not os.path.exists(output):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            gdown.download(url, output, quiet=False)
            st.success("✅ Model berhasil diunduh!")

download_model_from_drive()

# --- Load model & class names ---
@st.cache_resource
def load_model_and_classes():
    model_filename = "klasifikasi_sampah_menggunakan_ResNet-50.h5"

    def resnet_preprocess_for_lambda(x):
        return tf.keras.applications.resnet50.preprocess_input(x)

    custom_objects = {
        'resnet50_preprocess_input_layer': resnet_preprocess_for_lambda
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

    try:
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    except:
        class_names = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

    return model, class_names

model, class_names = load_model_and_classes()

# --- Ambil preprocessing & base ResNet ---
try:
    preprocess_layer = model.get_layer("sequential")
    resnet_model = model.get_layer("resnet50")
except ValueError:
    st.error("Model tidak memiliki layer 'sequential' atau 'resnet50'")
    st.stop()

# --- Layer Visualisasi ---
layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
available_layers = [l for l in layer_names if any(layer.name == l for layer in resnet_model.layers)]

selected_layer = st.sidebar.selectbox("🔍 Pilih Layer ResNet50", available_layers)

# --- Upload gambar ---
uploaded_file = st.file_uploader("📤 Upload Gambar Sampah (.jpg, .jpeg)", type=["jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Gagal membaca gambar.")
        st.stop()

    image_display = image.resize((300, 300))
    st.image(image_display, caption="🖼️ Gambar Diupload (ukuran tetap)", width=300)

    resized_image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(resized_image).astype(np.float32), axis=0)

    # --- Prediksi ---
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    pred_idx = np.argmax(prediction)

    threshold = 0.6  # Confidence threshold
    if confidence < threshold:
        pred_class = "❌ Gambar tidak dikenali sebagai sampah, Coba gambar lain"
    else:
        pred_class = class_names[pred_idx]

    st.success(f"✅ Prediksi: **{pred_class}** ({confidence * 100:.2f}%)")

    # --- Traversal aman untuk ambil feature map ---
    input_tensor = tf.keras.Input(shape=(224, 224, 3), name="input_streamlit")
    x = preprocess_layer(input_tensor)
    target_output = None

    for layer in resnet_model.layers:
        try:
            x = layer(x)
        except:
            continue
        if layer.name == selected_layer:
            target_output = x
            break

    if target_output is None:
        st.error(f"❌ Layer '{selected_layer}' tidak ditemukan saat traversal.")
        st.stop()

    feature_model = tf.keras.Model(inputs=input_tensor, outputs=target_output)
    feat_map = feature_model.predict(img_array)[0]

    # --- Visualisasi Feature Map ---
    if feat_map.ndim == 3:
        ch = feat_map.shape[-1]
        st.markdown(f"### 🧱 Feature Map - {selected_layer} (shape: {feat_map.shape})")

        max_channels = st.slider("🔢 Jumlah Channel yang Ditampilkan", 1, ch, min(32, ch))
        grid_cols = st.slider("🧱 Kolom Grid", 2, min(16, max_channels), 6)
        grid_rows = (max_channels + grid_cols - 1) // grid_cols

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, grid_rows * 2))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i in range(max_channels):
            if i < len(axes):
                axes[i].imshow(feat_map[:, :, i], cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"Ch {i+1}")
        for j in range(max_channels, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### 🔥 Heatmap Overlay")
        heatmap = np.mean(feat_map, axis=-1)
        heatmap = np.clip(heatmap / (np.max(heatmap) + 1e-6), 0, 1)
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(resized_image.size).convert("L")
        heatmap_color = plt.cm.jet(np.array(heatmap_img))[:, :, :3]
        overlay = 0.5 * (np.array(resized_image) / 255.0) + 0.5 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig_overlay = plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        st.pyplot(fig_overlay)

        st.markdown("### 🌐 3D Scatter Feature Map")
        ch_idx = st.slider("Channel untuk Scatter", 0, ch - 1, 0, format="Channel %d")
        fmap_3d = feat_map[:, :, ch_idx]
        if fmap_3d.ndim == 2:
            xx, yy = np.meshgrid(range(fmap_3d.shape[1]), range(fmap_3d.shape[0]))
            fig_3d = plt.figure(figsize=(10, 6))
            ax = fig_3d.add_subplot(111, projection='3d')
            sc = ax.scatter(xx.flatten(), yy.flatten(), fmap_3d.flatten(), c=fmap_3d.flatten(), cmap="plasma", s=10)
            ax.set_title(f"3D Feature Map - {selected_layer}, Channel {ch_idx+1}")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Aktivasi")
            fig_3d.colorbar(sc, shrink=0.5, aspect=10)
            st.pyplot(fig_3d)
    else:
        st.warning("Layer ini tidak bisa divisualisasikan (output tidak 3D).")
else:
    st.info("Silakan upload gambar untuk memulai.")
