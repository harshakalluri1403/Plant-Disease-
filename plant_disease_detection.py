import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
from preprocess import train_generator
import folium
from streamlit_folium import folium_static

# Set image dimensions
img_height, img_width = 256, 256

# Load the trained model
model_file = r'/home/pkalluri/Project83/model_6.h5'
model = load_model(model_file)

# Reverse the class indices dictionary for easy lookup
class_indices = {v: k for k, v in train_generator.class_indices.items()}

# Preprocess input for MobileNetV2
def preprocess_input(image):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_height, img_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict_disease(image_path, model, class_indices):
    image = load_and_preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_indices[predicted_class]
    return predicted_label

# Function to fetch recommended pesticide based on the predicted disease
def get_recommended_pesticide(disease):
    df = pd.read_excel(r'/home/pkalluri/Project83/pesticide.xlsx', engine='openpyxl')
    pesticide = df[df['Disease'] == disease]['Recommended Pesticide'].values
    if len(pesticide) > 0:
        return pesticide[0]
    else:
        return "No recommendation available"

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Custom CSS for styling with background image
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        background-image: url('https://wallpapers.com/images/hd/aesthetic-monstera-cute-plant-xlqdh6hj136ynju6.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .title {
        color: #000000;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        font-size: 36px;
    }
    .subtitle {
        color: #000000;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
    }
    .capture-button {
        background-color: #1f77b4;
        color: white;
        padding: 10px 20px;
        font-size: 18px;
        border: none;
        border-radius: 5px;
    }
    .capture-button:hover {
        background-color: #0f5294;
    }
    .image-container {
        text-align: center;
    }
    .caption {
        font-size: 18px;
        color: #555;
    }
    .options-container {
        margin-top: 30px;
        text-align: center;
    }
    .option-button {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        font-size: 18px;
        margin: 5px;
        border: none;
        border-radius: 5px;
        display: inline-block;
    }
    .option-button:hover {
        background-color: #218838;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    st.markdown("<h1 class='title'>Plant Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Capture an image or upload a file of your leaf.</p>", unsafe_allow_html=True)

    # Option to capture image from webcam
    if st.button("Capture Image", key='capture-button', help='Click to capture an image'):
        frame = capture_image()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.markdown("<div class='image-container'><h2 class='caption'>Captured Image</h2></div>", unsafe_allow_html=True)
        st.image(frame_rgb, use_column_width=True)
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame_rgb)
        predicted_label = predict_disease(image_path, model, class_indices)
        st.markdown(f"<h2 class='caption'>The predicted disease is: {predicted_label}</h2>", unsafe_allow_html=True)
        recommended_pesticide = get_recommended_pesticide(predicted_label)
        st.markdown(f"<h2 class='caption'>Recommended Pesticide: {recommended_pesticide}</h2>", unsafe_allow_html=True)

    # Option to upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        st.markdown("<div class='image-container'><h2 class='caption'>Uploaded Image</h2></div>", unsafe_allow_html=True)
        st.image(image_rgb, use_column_width=True)
        image_path = 'uploaded_image.jpg'
        cv2.imwrite(image_path, image_rgb)
        predicted_label = predict_disease(image_path, model, class_indices)
        st.markdown(f"<h2 class='caption'>The predicted disease is: {predicted_label}</h2>", unsafe_allow_html=True)
        recommended_pesticide = get_recommended_pesticide(predicted_label)
        st.markdown(f"<h2 class='caption'>Recommended Pesticide: {recommended_pesticide}</h2>", unsafe_allow_html=True)

    # Additional options with maps
    st.markdown("<div class='options-container'>", unsafe_allow_html=True)
    
    # Map for finding a plant doctor near me
    if st.button("Find a plant doctor near me", key='doctor-button', help='Find a nearby plant doctor'):
        st.markdown(f"<h2 class='caption'>Nearby Plant Doctors</h2>", unsafe_allow_html=True)
        plant_doctor_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        folium.Marker([19.0760, 72.8777], popup="Plant Doctor 1", tooltip="Plant Doctor 1").add_to(plant_doctor_map)
        folium.Marker([28.7041, 77.1025], popup="Plant Doctor 2", tooltip="Plant Doctor 2").add_to(plant_doctor_map)
        folium_static(plant_doctor_map)

    # Map for finding a pesticide store near me
    if st.button("Pesticide store near me", key='pesticide-button', help='Find a nearby pesticide store'):
        st.markdown(f"<h2 class='caption'>Nearby Pesticide Stores</h2>", unsafe_allow_html=True)
        pesticide_store_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        folium.Marker([13.0827, 80.2707], popup="Pesticide Store 1: 50 kg", tooltip="Pesticide Store 1: 50 kg").add_to(pesticide_store_map)
        folium.Marker([22.5726, 88.3639], popup="Pesticide Store 2: 30 kg", tooltip="Pesticide Store 2: 30 kg").add_to(pesticide_store_map)
        folium_static(pesticide_store_map)

    # Map for finding a plant nursery near me
    if st.button("Plant nursery near me", key='nursery-button', help='Find a nearby plant nursery'):
        st.markdown(f"<h2 class='caption'>Nearby Plant Nurseries</h2>", unsafe_allow_html=True)
        plant_nursery_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        folium.Marker([12.9716, 77.5946], popup="Nursery 1: Flowering Plants", tooltip="Nursery 1: Flowering Plants").add_to(plant_nursery_map)
        folium.Marker([17.3850, 78.4867], popup="Nursery 2: Succulents", tooltip="Nursery 2: Succulents").add_to(plant_nursery_map)
        folium_static(plant_nursery_map)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

