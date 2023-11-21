import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def dist_euc(x1, x2, y1, y2):
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def dist_man(x1, x2, y1, y2):
    return np.abs(x1-x2) + np.abs(y1-y2)

grayscale = False
use_euc = True

cap = cv2.VideoCapture(1)
with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks != None :
            right_x1 = results.pose_landmarks.landmark[8].x
            right_x2 = results.pose_landmarks.landmark[20].x
            right_y1 = results.pose_landmarks.landmark[8].y
            right_y2 = results.pose_landmarks.landmark[20].y
            left_x1 = results.pose_landmarks.landmark[7].x
            left_x2 = results.pose_landmarks.landmark[19].x
            left_y1 = results.pose_landmarks.landmark[7].y
            left_y2 = results.pose_landmarks.landmark[19].y
            dist = dist_euc
            if not use_euc:
                dist = dist_man
            print("R = %.3f | L = %.3f" % (dist(right_x1, right_x2, right_y1, right_y2), dist(left_x1, left_x2, left_y1, left_y2)), end='\r')
        cv2.imshow('Phone pose detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()