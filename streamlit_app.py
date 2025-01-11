import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from pygame import mixer
from PIL import Image

# Initialize pygame mixer for alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load Haar Cascade Classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to preprocess eye frames for the model
def preprocess_eye(eye_frame):
    eye_resized = cv2.resize(eye_frame, (80, 80))  # Resize to model input size
    eye_normalized = eye_resized / 255.0  # Normalize pixel values
    return np.expand_dims(eye_normalized, axis=0)  # Add batch dimension

# Streamlit app interface
st.title("Driver Drowsiness Detection")
st.write("Press 'Start Detection' to begin monitoring drowsiness using your webcam.")

# Initialize the detection system
if st.button("Start Detection"):
    st.info("Starting webcam... Press 'q' to stop detection.")

    Score = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Please check your device.")
            break

        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces and eyes
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Process each detected eye
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)

            eye_frame = frame[ey:ey + eh, ex:ex + ew]
            processed_eye = preprocess_eye(eye_frame)
            prediction = model.predict(processed_eye)

            if prediction[0][0] > 0.30:  # Eyes closed
                cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                Score += 1
                if Score > 5:
                    try:
                        sound.play()
                    except:
                        pass
            elif prediction[0][1] > 0.90:  # Eyes open
                cv2.putText(frame, "Open", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                Score -= 1
                if Score < 0:
                    Score = 0

        cv2.putText(frame, f"Score: {Score}", (100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", caption="Webcam Feed")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    st.success("Detection stopped.")
