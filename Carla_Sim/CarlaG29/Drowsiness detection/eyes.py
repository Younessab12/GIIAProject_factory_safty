import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import asyncio
import time

using_phone = False

async def play():
    global using_phone
    wave_object = sa.WaveObject.from_wave_file('alarm.wav')
    while True:
        if using_phone:
            print("#")
            play_object = wave_object.play()
            play_object.wait_done()
            using_phone = False
        else:
            print(".")
play()



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


face_classifier =  cv2.CascadeClassifier(r'haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_classifier = cv2.CascadeClassifier(r'haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_classifier = cv2.CascadeClassifier(r'haar cascade files/haarcascade_righteye_2splits.xml')

labels = ['Closed', 'Open']

model = load_model('models/eyes_classifier.h5')
cap = cv2.VideoCapture(0)

def get_eye_state(face_img, eye):
    x, y, w, h = eye
    eye_img = face_img[y:y+h,x:x+w]
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img / 255
    eye_img =  eye_img.reshape(24, 24, -1)
    eye_img = np.expand_dims(eye_img, axis=0)
    state, = np.argmax(model.predict(eye_img), axis=-1)
    return state


def eyes_detection(gray_img, bgr_img):
    faces = face_classifier.detectMultiScale(gray_img, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    # Get the closest face to the center of the screen
    face = None
    min_distance_from_center_screen = 1e9
    for x, y, w, h in faces:
        if (x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2 < min_distance_from_center_screen:
            min_distance_from_center_screen = (x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2
            face = (x, y, w, h)
    if face is None:
        return None

    fx, fy, fw, fh = face
    cv2.rectangle(bgr_img, (fx, fy), (fx + fw, fy + fh), (100, 100, 100), 1)

    face_img = gray_img[fy:fy+fh, fx:fx+fw]

    left_eyes = left_eye_classifier.detectMultiScale(face_img)
    right_eyes =  right_eye_classifier.detectMultiScale(face_img)

    min_x, max_x = 1e9, -1
    left_eye, right_eye = None, None
    # Right eye should have a minimal x
    for x, y, w, h in right_eyes:
        if x + w / 2 < min_x:
            min_x = x + w / 2
            right_eye = (x, y, w, h)
    # Left eye should have a maximal x
    for x, y, w, h in left_eyes:
        if x + w / 2 > max_x:
            max_x = x + w / 2
            left_eye = (x, y, w, h)

    left_state = 1 # Open by default
    if left_eye is not None:
        lx, ly, lw, lh = left_eye
        cv2.rectangle(bgr_img, (lx + fx, ly + fy), (lx + fx + lw, ly + fy + lh), (0, 0, 255), 1)
        left_state = get_eye_state(face_img, left_eye)
        
        
    right_state = 1 # Open by default
    if right_eye is not None:
        rx, ry, rw, rh = right_eye
        cv2.rectangle(bgr_img, (rx + fx, ry + fy), (rx + fx + rw, ry + fy + rh), (200, 200, 0), 1)
        right_state = get_eye_state(face_img, right_eye)
    
    return left_state, right_state

def dist(x1, x2, y1, y2):
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def phone_detection(rgb_img, bgr_img):
    rgb_img.flags.writeable = False
    results = pose.process(rgb_img)
    rgb_img.flags.writeable = True

    mp_drawing.draw_landmarks(
            bgr_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks != None :
        right_x1 = results.pose_landmarks.landmark[8].x
        right_x2 = results.pose_landmarks.landmark[20].x
        right_y1 = results.pose_landmarks.landmark[8].y
        right_y2 = results.pose_landmarks.landmark[20].y
        left_x1 = results.pose_landmarks.landmark[7].x
        left_x2 = results.pose_landmarks.landmark[19].x
        left_y1 = results.pose_landmarks.landmark[7].y
        left_y2 = results.pose_landmarks.landmark[19].y
        rd = dist(right_x1, right_x2, right_y1, right_y2)
        ld = dist(left_x1, left_x2, left_y1, left_y2)
        if rd < 0.15 or ld < 0.15:
            return True

    return False
with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        success, bgr_img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        height, width = bgr_img.shape[:2] 
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        eyes_state = eyes_detection(gray_img, bgr_img)

        phone_usage = phone_detection(rgb_img, bgr_img)

        using_phone = phone_usage

        cv2.imshow('Drowsiness detection', bgr_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
