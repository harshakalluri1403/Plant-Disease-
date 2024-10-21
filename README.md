# Plant Disease Detection

## Overview
This project outlines the technical design and implementation of a plant disease detection system. The system features a web-based interface built with Streamlit, enabling real-time image upload capabilities and additional functionalities such as pesticide recommendations and geolocation services.

![Main Web Page for Plant Disease Detection](https://github.com/harshakalluri1403/Plant-Disease-/blob/f062944841173afcb8e4a01698141862432f0fe4/Readme/Screenshot%202024-08-12%20085225.png)  <!-- Replace with your image path -->

## 1. Web Application Framework (Streamlit)
The web application for plant disease detection is developed using Streamlit, a Python-based framework known for its ease of use and rapid prototyping capabilities. Streamlit allows for the creation of interactive, user-friendly web applications with minimal effort. The app provides a streamlined interface for farmers and agricultural experts to upload images of tomato leaves and receive real-time predictions of possible diseases.

### Features of Streamlit in this project:
- **Interactive Widgets**: Used for image uploads and capturing images via webcam.
- **Real-Time Predictions**: Immediate feedback to users upon uploading an image, with the ability to diagnose the disease within seconds.
- **Clean UI/UX**: The use of Streamlit ensures a responsive design, with simple and clear instructions guiding the user through the process of uploading images and receiving recommendations.

The lightweight nature of Streamlit allows the application to be run on various devices, including desktop computers and mobile devices, making it accessible to a wide range of users.

## 2. Real-Time Image Upload and Disease Diagnosis
![Uploading the Image to the Main Page from Local Folder](https://github.com/harshakalluri1403/Plant-Disease-/blob/f062944841173afcb8e4a01698141862432f0fe4/Readme/Screenshot%202024-08-12%20090532.png)  <!-- Replace with your image path -->

One of the core functionalities of the web application is the ability to upload images of tomato leaves and receive real-time disease predictions. This feature allows users to:

- **Upload Images**: Users can upload an image from their device, which is then resized and preprocessed to fit the input requirements of the deep learning models (MobileNetV2, ResNet50, and VGG16).
- **Real-Time Diagnosis**: After uploading, the image is fed through the pre-trained model, which predicts the disease associated with the plant leaf. The prediction process takes only a few seconds, ensuring minimal delay between the image upload and disease diagnosis.
- **Disease Prediction Results**: The predicted disease, along with a confidence score, is displayed to the user. Based on the prediction, the application provides recommendations for treatment, including appropriate pesticides.
- **Webcam Image Capture**: For real-time usage in the field, the app includes a "Capture Image" feature that allows users to take a photo using their webcam. This image is processed in the same manner as an uploaded image.

### Steps of Real-Time Processing:
1. Image upload or capture via webcam.
2. Image preprocessing: resizing, normalization, and format conversion.
3. Feeding the preprocessed image into the trained model.
4. Displaying the predicted disease and appropriate pesticide recommendations.

## 3. Additional Features (Pesticide Recommendations, Geolocation Services)
![Displaying Nearby Plant Doctors](https://github.com/harshakalluri1403/Plant-Disease-/blob/f062944841173afcb8e4a01698141862432f0fe4/Readme/Screenshot%202024-08-12%20090640.png)  <!-- Replace with your image path -->

Beyond disease diagnosis, the application provides several additional features aimed at offering comprehensive support to users, including pesticide recommendations and geolocation services.

- **Pesticide Recommendations**: After predicting the disease, the application suggests suitable pesticides from a pre-defined dataset. The recommendations are based on the predicted disease label and provide users with actionable insights on how to treat the plant.
- **Geolocation Services**:
  - **Plant Doctors**: A map is provided that shows the locations of nearby plant doctors who can offer expert advice on plant health.
  - **Pesticide Stores**: The app allows users to locate stores where the recommended pesticides can be purchased.
  - **Plant Nurseries**: The app provides information on nearby plant nurseries, helping users find and purchase healthy plants.

These features are implemented using Folium, a Python library for creating interactive maps, enabling users to explore their surrounding areas and access necessary resources related to plant health.

## 4. Results and Discussion
![Predicting the Disease and Recommending Suitable Pesticides](https://github.com/harshakalluri1403/Plant-Disease-/blob/f062944841173afcb8e4a01698141862432f0fe4/Readme/Screenshot%202024-08-12%20091010.png)  <!-- Replace with your image path -->

This section discusses the performance of three deep learning models—VGG16, MobileNetV2, and ResNet50—in the task of tomato plant disease detection. It also analyzes the results across different epochs and assesses the overall system performance and user experience.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Run the Streamlit application:
   ``` bash
   streamlit run app.py
   ```
2. Open your web browser and go to http://localhost:8501.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
