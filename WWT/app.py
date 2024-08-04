import streamlit as st
import cv2
import numpy as np
import warnings
import tensorflow as tf
import winsound

warnings.filterwarnings("ignore", category=DeprecationWarning)

model = tf.keras.models.load_model(r"C:\Users\mhais\Downloads\C__Users_mhais_OneDrive_Desktop_driver_drowsiness_driver_final.h5")


def detect_drowsiness(frame, model):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]


def play_alarm():
    frequency = 1000
    duration = 1600
    winsound.Beep(frequency, duration)  # Play the sound


st.title("SafeDrive AI")


option = st.selectbox("Choose Input Method", ("Upload Video", "Use Live Camera"))

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video("uploaded_video.mp4")

        if st.button('Detect'):
            cap = cv2.VideoCapture("uploaded_video.mp4")
            drowsy_detected = False
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 == 0:
                    prediction = detect_drowsiness(frame, model)
                    if prediction > 0.5:
                        drowsy_detected = True
                        play_alarm()  # Call the play_alarm function
                        break

            cap.release()

            if drowsy_detected:
                st.write("Drowsiness Detected")
            else:
                st.write("No Drowsiness Detected")

elif option == "Use Live Camera":
    st.write("Starting live camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        drowsy_detected = False
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam.")
                break

            frame_placeholder.image(frame, channels="BGR")
            prediction = detect_drowsiness(frame, model)
            if prediction > 0.5:
                drowsy_detected = True
                play_alarm()  # Call the play_alarm function
                break

        cap.release()

        if drowsy_detected:
            st.write("Drowsiness Detected")
        else:
            st.write("No Drowsiness Detected")
