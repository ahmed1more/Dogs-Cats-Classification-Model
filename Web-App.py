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

# Import the Cat_Dog_CNN class from the arch.py file
from arch import Cat_Dog_CNN

# --- Configuration & Model Loading ---
# Setup device (Use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and load the weights into the model
model = Cat_Dog_CNN().to(device)
model.load_state_dict(torch.load("models\depoly-85-91.pth", map_location=device))

model.eval()

# Define the preprocessing transformations (Must match training preprocessing)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# --- Core Functions ---

def predict_image(image, threshold=0.75):
    """
    Analyzes the image and returns the label, confidence, and raw probabilities.
    Includes a threshold to detect if the image is neither a cat nor a dog.
    """
    # Preprocess image and add batch dimension (Batch Size = 1)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable gradients for faster inference
        output = model(img_tensor)
        # Convert raw output (logits) to probabilities
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        # Get the highest probability and its corresponding index
        prediction_prob, predicted = torch.max(probs, dim=0)
        
        # Logic for 'No one': If the highest confidence is below our threshold
        if prediction_prob < threshold:
            prediction_label = "No one (Unknown)"
        else:
            prediction_label = "Dog" if predicted == 1 else "Cat"
            
    return prediction_label, prediction_prob.item(), probs, img_tensor

def play_audio(text):
    """Converts text to speech and plays it in the Streamlit app."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# Updated function to accept threshold
def process_and_predict(image, threshold, image_caption="Image"):
    st.image(image, caption=image_caption, use_container_width=True)

    # Now we pass the threshold from the slider here
    label, probability, probabilities, _ = predict_image(image, threshold=threshold)
    
    if label == "No one (Unknown)":
        st.warning(f"Result: **{label}** (Confidence: {probability*100:.2f}%)")
        play_audio("I am not sure what this is.")
    else:
        st.success(f"Prediction: **{label}** ({probability*100:.2f}%)")
        play_audio(f"The prediction is {label}.")
# --- Streamlit UI Layout ---

st.title("Smart Cat vs Dog Classifier")
st.sidebar.header("Settings")
# Adding a slider so you can adjust the "Unknown" sensitivity live
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.9)

st.subheader("Predict using Folder, Upload, or Camera")

# Section 1: Select from existing local folders
folder_choice = st.radio("Select an image folder:", ("cats", "dogs"))

if folder_choice:
    image_folder = "test_set_Sample/Cat" if folder_choice == "cats" else "test_set_Sample/Dog"
    if os.path.exists(image_folder):
        image_files = os.listdir(image_folder)
        if image_files:
            selected_image = st.selectbox("Choose an image:", image_files)
            image_path = os.path.join(image_folder, selected_image)
            if st.button(f"Analyze {selected_image}"):
                image = Image.open(image_path)
                process_and_predict(image,conf_threshold, f"Selected: {selected_image}")
        else:
            st.error("Folder is empty.")
    else:
        st.error(f"Path '{image_folder}' not found.")

# Section 2: File Uploader (Batch Support)
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    cols = st.columns(3) # Display 3 images per row
    for idx, file in enumerate(uploaded_files):
        with cols[idx % 3]:
            img = Image.open(file)
            label, prob, _, _ = predict_image(img, threshold=conf_threshold)
            st.image(img, caption=file.name)
            st.write(f"**{label}**")

# Initialize the camera state (Put this right before the camera block)
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

st.divider()
st.subheader("Camera Support")

# Logic for Opening/Closing the camera
if not st.session_state.show_camera:
    # If camera is closed, show "Open" button
    if st.button("📷 Open Camera"):
        st.session_state.show_camera = True
        st.rerun() 
else:
    # If camera is open, show "Close" button
    if st.button("❌ Close Camera"):
        st.session_state.show_camera = False
        st.rerun()

# This part only runs if the user clicked "Open Camera"
if st.session_state.show_camera:
    camera_image = st.camera_input("Take a picture using your camera")

    if camera_image:
        image = Image.open(camera_image)
        st.subheader("Camera Image Prediction:")
        # Call the processing function we defined earlier
        process_and_predict(image,conf_threshold, "Captured Image")