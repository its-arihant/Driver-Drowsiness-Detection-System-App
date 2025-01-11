import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Function to capture video and detect drowsiness
def detect_drowsiness():
    st.info("Starting webcam... Press 'Q' in the video window to stop.")
    cap = cv2.VideoCapture(0)  # Open webcam (device 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make predictions
        prediction = model.predict(processed_frame)
        drowsiness_score = prediction[0][0]  # Adjust based on your model's output

        # Display the prediction
        label = "Drowsy" if drowsiness_score > 0.5 else "Alert"
        color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
        cv2.putText(frame, f"{label} ({drowsiness_score:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Exit on pressing 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("Press the button below to start detecting drowsiness using your webcam.")

if st.button("Start Detection"):
    detect_drowsiness()
