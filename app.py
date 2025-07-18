import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image

# โหลดโมเดล
MODEL_PATH = "baseline_model.keras"  # ต้องอัปโหลดไฟล์นี้ขึ้นไปใน repo ด้วย
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# UI
st.set_page_config(page_title="OCT Classifier")
st.title("Retinal OCT Image Classification")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ประมวลผลภาพ
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # พยากรณ์
    predictions = model.predict(img_array)[0]
    confidences = {CLASS_NAMES[i]: round(float(conf) * 100, 2) for i, conf in enumerate(predictions)}

    # แสดงผลลัพธ์
    st.subheader("Prediction:")
    for cls, conf in confidences.items():
        st.write(f"**{cls}**: {conf}%")

    # แสดงแถบความมั่นใจ
    st.bar_chart(predictions)
