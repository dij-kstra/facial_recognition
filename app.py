import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('facial_expression_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app
st.title('Facial Emotion Recognition')
st.write('Upload a cropped Image of a face of single person')

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((48, 48))  # Resize to model's input size
        image = np.array(image) / 255.0  # Normalize pixel values

        # Ensure the image has the correct shape (batch_size, height, width, channels)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Make prediction using the model
        prediction = model.predict(image)
        predicted_label = emotion_labels[np.argmax(prediction)]

        # Display the uploaded image and prediction
        st.image(image[0, :, :, 0], caption='Uploaded Image', use_column_width=True)
        st.write(f"Predicted Emotion: {predicted_label}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
