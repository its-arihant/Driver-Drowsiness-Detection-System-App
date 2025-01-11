import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from pygame import mixer

# Initialize pygame mixer for alarm sound
try:
    mixer.init()
    sound = mixer.Sound('alarm.wav')
except Exception as e:
    st.warning(f"Failed to initialize sound system: {e}")

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to preprocess the eye frame for the model
def preprocess_eye(eye_frame):
    eye_frame_resized = cv2.resize(eye_frame, (80, 80))  # Resize to model input size
    eye_frame_normalized = eye_frame_resized / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(eye_frame_normalized, axis=0)  # Add batch dimension

# Function to handle detection
def detect_drowsiness():
    st.info("Starting webcam... Press 'Stop Detection' to end.")
    
    # Attempt to access the webcam dynamically
    cap = None
    for i in range(5):  # Try the first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            st.info(f"Webcam found at index {i}")
            break
    else:
        st.error("Failed to access the webcam. Please check your device and permissions.")
        return
    
    Score = 0  # Drowsiness score
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                break
            
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces and eyes
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                # Extract and preprocess eye region
                eye = frame[ey:ey + eh, ex:ex + ew]
                processed_eye = preprocess_eye(eye)
                prediction = model.predict(processed_eye)
                
                # If eyes are closed
                if prediction[0][0] > 0.30:
                    Score += 1
                    label = "Closed"
                    color = (0, 0, 255)
                    if Score > 5:
                        try:
                            sound.play()
                        except:
                            pass
                # If eyes are open
                elif prediction[0][1] > 0.90:
                    Score = max(0, Score - 1)  # Prevent Score from going negative
                    label = "Open"
                    color = (0, 255, 0)
                
                # Display the label
                cv2.putText(frame, f"{label} - Score: {Score}", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display frame in Streamlit
            st.image(frame, channels="BGR", caption="Driver Drowsiness Detection")
            
            # Stop detection if the button is pressed
            if st.button("Stop Detection"):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        st.success("Detection stopped.")

# Streamlit UI
st.title("Driver Drowsiness Detection System")
st.write("Detect drowsiness using your webcam. Press the button below to start.")

if st.button("Start Detection"):
    detect_drowsiness()
