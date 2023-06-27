import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import model_from_json
import streamlit as st
import base64

emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emodel.h5")
st.header(":blue[Emotions Detection System]")

option = st.selectbox(
    'Three Options for your test',
    ('With Image file', 'With Video file', 'With Webcam'))


if option == "With Image file":
    file = st.file_uploader("upload an image", type=["jpg", "png"])
    def import_and_predict(image_data, model):
        size = (48, 48)
        image = ImageOps.fit(image_data, size)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        predict = emotion_model.predict(img_reshape)
        if file is None:
            st.text("Please upload an image.")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            string = emotion_dict[np.argmax(predictions)]
            st.success(string)
            return predict
    import_and_predict(file, model_from_json)
elif option == "With Video file":
    cap = cv2.VideoCapture("C:/Users/STH/Desktop/Emotions/happy.mp4")
    # Find haar cascade to draw bounding box around face

elif option == "With Webcam":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("The video capture has ended")
            break
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            image_with_rectangle = cv2.rectangle(gray_frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            image_with_text = cv2.putText(gray_frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

            frame_placeholder.image(image_with_text, channels="RGB")
        if stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()


