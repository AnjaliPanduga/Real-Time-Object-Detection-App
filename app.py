import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("🚀 Object Detection App")

option = st.sidebar.selectbox(
    "Choose Detection Type",
    ("Face Detection",
     "Face & Eye Detection",
     "Car Detection (Video)",
     "Full Body Detection (Video)")
)

# -------- GET BASE DIRECTORY (VERY IMPORTANT FOR CLOUD) -------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

car_path = os.path.join(BASE_DIR,"haarcascades","haarcascade_car.xml")
body_path = os.path.join(BASE_DIR,"haarcascades","haarcascade_fullbody.xml")

# -------- LOAD CASCADES -------- #

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

car_classifier = cv2.CascadeClassifier(car_path)
body_classifier = cv2.CascadeClassifier(body_path)

# Safety Check
if car_classifier.empty():
    st.error("Car Cascade NOT loaded in Cloud!")

if body_classifier.empty():
    st.error("Body Cascade NOT loaded in Cloud!")

# -------- IMAGE DETECTION FUNCTION -------- #

def process_image(img, detect_eye=False):

    img = cv2.resize(img, (600,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    faces = face_classifier.detectMultiScale(
        gray,1.2,5,minSize=(40,40))

    if len(faces)==0:
        faces = face_classifier.detectMultiScale(
            gray,1.1,4,minSize=(30,30))

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

        if detect_eye:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_classifier.detectMultiScale(
                roi_gray,1.1,6,minSize=(15,15))

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),
                              (ex+ew,ey+eh),(0,255,0),2)

    return img

uploaded_file = st.file_uploader("Upload Image/Video",
                                 type=["jpg", "png", "jpeg", "mp4"])

# -------- IMAGE DETECTION -------- #

if uploaded_file is not None:

    if option == "Face Detection":

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = process_image(img, False)
        st.image(result, channels="BGR")


    elif option == "Face & Eye Detection":

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = process_image(img, True)
        st.image(result, channels="BGR")

# -------- CAR VIDEO -------- #

    elif option == "Car Detection (Video)":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_classifier.detectMultiScale(gray,1.1,3)

            for (x,y,w,h) in cars:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

            stframe.image(frame, channels="BGR")

        cap.release()

# -------- FULL BODY VIDEO -------- #

    elif option == "Full Body Detection (Video)":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(
                gray,1.2,3,minSize=(60,60))

            for (x,y,w,h) in bodies:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            stframe.image(frame, channels="BGR")

        cap.release()
