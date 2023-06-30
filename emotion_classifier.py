import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import model_from_json
import streamlit as st
import base64

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
json_file = open('new model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("new model/emodel.h5")
st.header(":blue[Emotions Detection System]")

with st.sidebar:
    add_select = st.selectbox(
        "Select Activity",
        ("Home", "Emotion Detection")
    )
if add_select == "Home":
    st.title = ""

elif add_select == "Emotion Detection":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")
    while start_button_pressed and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended")
            break
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            image_with_rectangle = cv2.rectangle(gray_frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            image_with_text = cv2.putText(gray_frame, emotion_dict[maxindex], (x + 5, y - 20),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (255, 0, 0), 2, cv2.LINE_4)

            frame_placeholder.image(gray_frame, channels="RGB")
    while stop_button_pressed:
        break



