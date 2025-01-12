import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from playsound import playsound  # Use playsound for audio playback

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')

# Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Function to detect drowsiness using webcam input
def detect_drowsiness():
    st.info("Webcam activated. Capture an image for detection.")
    
    # Use the camera input widget in Streamlit
    webcam_image = st.camera_input("Capture image from webcam")

    if webcam_image is not None:
        # Convert the uploaded webcam image to an OpenCV format
        frame = cv2.imdecode(np.frombuffer(webcam_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)

        # Make predictions using the model
        prediction = model.predict(processed_frame)
        drowsiness_score = prediction[0][0]  # Adjust based on your model's output

        # Determine drowsiness state
        label = "Drowsy" if drowsiness_score > 0.5 else "Alert"
        color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

        # Add text to the image
        cv2.putText(frame, f"{label} ({drowsiness_score:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the processed frame with the label
        st.image(frame, channels="BGR", caption="Processed Webcam Feed")

    # Allow the user to stop detection
    if st.button("Stop Detection"):
        st.write("Detection Stopped")


# Additional code for face and eye detection with sound alert
def monitor_drowsiness():
    cap = cv2.VideoCapture(0)
    Score = 0

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces and eyes
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        # Draw background for score display
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(255, 0, 0), thickness=3)

            # Preprocess eye region
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80)) / 255  # Resize and normalize
            eye = np.expand_dims(eye, axis=0)  # Add batch dimension

            # Model prediction
            prediction = model.predict(eye)

            if prediction[0][0] > 0.30:
                cv2.putText(frame, 'Closed', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'Score {Score}', (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                Score += 1
                if Score > 5:
                    try:
                        playsound('alarm.wav')  # Use playsound for audio
                    except Exception as e:
                        print(f"Error playing sound: {e}")

            elif prediction[0][1] > 0.90:
                cv2.putText(frame, 'Open', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'Score {Score}', (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                Score = max(0, Score - 1)

        # Display the frame with annotations
        cv2.imshow('Drowsiness Detection', frame)

        # Stop detection when 'q' is pressed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("Press the button below to start detecting drowsiness using your webcam.")

if st.button("Start Detection"):
    detect_drowsiness()

if st.button("Start Drowsiness Monitoring"):
    monitor_drowsiness()
