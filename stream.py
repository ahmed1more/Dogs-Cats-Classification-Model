import streamlit as st
from PIL import Image
# import tensorflow as tf
import numpy as np
import os

# تحميل الموديل
# model = tf.keras.models.load_model("cat_dog_classifier.h5")

# دالة لتحليل الصور
# def predict_image(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     probability = prediction[0][0]
#     return ("Dog", probability) if probability > 0.5 else ("Cat", 1 - probability)

# واجهة Streamlit
st.title("Cat vs Dog Classifier")
st.subheader("Choose an image to predict whether it's a Cat or a Dog.")

# اختيار مجلد الصور
image_folder = "training_set/Dog"  # تغيير إلى مجلد الكلاب عند الحاجة
image_files = os.listdir(image_folder)

# قائمة لاختيار الصور
selected_image = st.selectbox("Choose an image from the folder:", image_files)

# عرض الصورة وتحليلها
# if selected_image:
#     image_path = os.path.join(image_folder, selected_image)
#     st.image(image_path, caption=f"Selected Image: {selected_image}")

#     if st.button("Predict"):
#         label, probability = predict_image(image_path)
#         st.write(f"Prediction: {label}")
#         st.write(f"Confidence: {probability:.2%}")