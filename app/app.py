import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import sqlite3

# --- Paths ---
BASE_DIR = "C:\\Users\\USER\\OneDrive\\Desktop\\Richard-250000395\\emotion_detection"
MODEL_PATH = os.path.join(BASE_DIR, "model", "face_emotionModel.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")
DB_PATH = os.path.join(BASE_DIR, "database.db")

# --- Load model and Haar Cascade ---
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Streamlit UI ---
st.title("Facial Emotion Detection 🎭")
st.write("Upload a photo to detect your emotion.")

# --- File uploader ---
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img)
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected! Make sure your face is clearly visible.")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi_input = roi.reshape(1,48,48,1)/255.0

            prediction = model.predict(roi_input)
            label = emotion_labels[np.argmax(prediction)]

            # Draw rectangle and label
            cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img_np, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        st.image(img_np, caption="Detected Emotion")

        # --- User info ---
        name = st.text_input("Enter your name")
        email = st.text_input("Enter your email")

        if st.button("Submit"):
            if name == "" or email == "":
                st.warning("Please enter your name and email.")
            else:
                # Save image bytes
                uploaded_img.seek(0)  # Reset pointer
                img_bytes = uploaded_img.read()

                # Save to database
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO users (name, email, emotion, image) VALUES (?, ?, ?, ?)",
                    (name, email, label, img_bytes)
                )
                conn.commit()
                conn.close()

                st.success(f"Data saved! Looks like you are feeling: {label}")