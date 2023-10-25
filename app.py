import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import model_from_json

# Load the trained model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model_weights.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="✅", initial_sidebar_state="expanded")

st.title('Facial Emotion Recognition')
st.write('Upload an Image. Faces will be detected and their emotions will be predicted')

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a copy of the image to draw rectangles on
        image_with_rectangles = np.copy(img_array)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around each face
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Extract and preprocess the detected face for emotion prediction
            detected_face = gray[y:y + h, x:x + w]
            detected_face = cv2.resize(detected_face, (48, 48))
            detected_face = detected_face / 255.0
            detected_face = np.expand_dims(detected_face, axis=0)
            detected_face = np.expand_dims(detected_face, axis=-1)

            # Make emotion prediction using the model
            emotion_prediction = model.predict(detected_face)
            predicted_label = emotion_labels[np.argmax(emotion_prediction)]

            # Add the predicted emotion label near the rectangle
            cv2.putText(image_with_rectangles, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the color image with rectangles and predictions
        st.image(image_with_rectangles, caption='Uploaded Image with Emotion Predictions', use_column_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")


