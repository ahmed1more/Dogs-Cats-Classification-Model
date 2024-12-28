import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
from gtts import gTTS  # Text-to-speech conversion
import tempfile  # For creating temporary files
import numpy as np
from captum.attr import Saliency  # For analyzing influential regions
import pandas as pd  # For creating interactive tables

# Import the CNN_1 class from the model.py file
from arch import Cat_Dog_CNN

# Load the PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Cat_Dog_CNN().to(device)
model.load_state_dict(torch.load("models\depoly-85-91.pth", map_location=device))
model.eval()

# Define the necessary transformations for images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

# Function for image analysis
def predict_image(image):
    img_tensor = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():  # Disable gradient computation for performance improvement
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]  # Get probabilities for each class
        predicted = torch.argmax(probs).item()  # Get the predicted class
        prediction_label = "Dog" if predicted == 1 else "Cat"
        prediction_prob = probs[predicted].item()
    return prediction_label, prediction_prob, probs, img_tensor

# Function to play audio based on text
def play_audio(text):
    tts = gTTS(text=text, lang='en')  # Convert text to speech
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)  # Save audio to a temporary file
        st.audio(fp.name, format="audio/mp3")  # Play the audio

# Function to generate Saliency Map
def generate_saliency_map(model, img_tensor):
    saliency = Saliency(model)
    img_tensor.requires_grad_()  # Enable gradient computation for the image
    grads = saliency.attribute(img_tensor, target=1)  # Extract gradients for the "Dog" class
    grads = grads.squeeze().cpu().numpy()
    saliency_map = np.maximum(grads, 0)  # Keep only positive values
    saliency_map = saliency_map.mean(axis=0)  # Calculate the average across channels
    return saliency_map

# Common function to process and predict images
def process_and_predict(image, image_caption="Image"):
    st.image(image, caption=image_caption, use_container_width=True)

    # Prediction
    label, probability, probabilities, img_tensor = predict_image(image)
    st.write(f"Prediction: **{label}** with probability: **{probability*100:.2f}%**")
    st.write(f"Probabilities: **Cat**: {probabilities[0]*100:.2f}%, **Dog**: {probabilities[1]*100:.2f}%")

    # Play audio based on the result
    play_audio(f"The prediction is {label} with probability {probability*100:.2f} percent.")

# Streamlit Interface
st.title("Cat vs Dog Classifier with Advanced Features")
st.subheader("Choose an image, upload multiple images, or use your camera for predictions.")

# **Select images from a folder:**
folder_choice = st.radio("Select an image folder:", ("cats", "dogs"))

if folder_choice:
    image_folder = "test_set_Sample/Cat" if folder_choice == "cats" else "test_set_Sample/Dog"
    if os.path.exists(image_folder):
        image_files = os.listdir(image_folder)
        if len(image_files) > 0:
            selected_image = st.selectbox("Choose an image from the folder:", image_files)
            image_path = os.path.join(image_folder, selected_image)
            if st.button(f"Predict {selected_image}"):
                image = Image.open(image_path)
                process_and_predict(image, f"Selected Image: {selected_image}")
        else:
            st.error(f"No images found in the '{folder_choice}' folder.")
    else:
        st.error(f"Image folder '{image_folder}' not found. Please check the path.")

# **Upload a batch of images**
uploaded_files = st.file_uploader("Upload images for prediction and comparison", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Image Comparison and Predictions")

    num_cols = 3  # Number of images per row
    rows = [uploaded_files[i:i + num_cols] for i in range(0, len(uploaded_files), num_cols)]

    for row in rows:
        cols = st.columns(len(row))
        for idx, uploaded_file in enumerate(row):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                
                # Prediction for each image
                label, probability, _, _ = predict_image(image)
                st.write(f"Prediction: **{label}**")
                st.write(f"Confidence: **{probability*100:.2f}%**")

                if st.button(f"Play Sound for {uploaded_file.name}", key=f"sound_{uploaded_file.name}"):
                    play_audio(f"The prediction is {label} with a confidence of {probability*100:.2f} percent.")

# **Camera Support:**
camera_image = st.camera_input("Take a picture using your camera")

if camera_image:
    image = Image.open(camera_image)
    st.subheader("Camera Image Prediction:")
    process_and_predict(image, "Captured Image")
