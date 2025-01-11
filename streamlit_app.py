import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Function to detect drowsiness using webcam input
def detect_drowsiness():
    st.info("Starting webcam... Press 'Stop Detection' to end.")
    
    # Use the camera input widget in Streamlit
    webcam_image = st.camera_input("Capture image from webcam")

    if webcam_image is not None:
        # Convert the uploaded webcam image to an OpenCV format
        frame = np.array(webcam_image)

        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)

        # Make predictions using the model
        prediction = model.predict(processed_frame)
        drowsiness_score = prediction[0][0]  # Adjust based on your model's output

        # Display prediction
        label = "Drowsy" if drowsiness_score > 0.5 else "Alert"
        color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
        cv2.putText(frame, f"{label} ({drowsiness_score:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the processed frame
        st.image(frame, channels="RGB", caption="Processed Webcam Feed")

    # Allow the user to stop detection
    if st.button("Stop Detection"):
        st.write("Detection Stopped")

# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("Press the button below to start detecting drowsiness using your webcam.")

if st.button("Start Detection"):
    detect_drowsiness()
