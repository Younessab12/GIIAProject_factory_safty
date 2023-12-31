import mediapipe as mp
import cv2
import numpy as np
import time
import math
import os
import sys

import facerecognition.face_rec as FaceRec
import face_assessment.face_assessment as FaceAssess
import illegal_action_detection.illegal_action_detection as IllegalAction
import utils.api as API

api = API.API(baseUrl="http://localhost:3000", apiKey="ABCDE123")

FaceAssessment =  FaceAssess.FaceAssessment()

IllegalActionDetection =  IllegalAction.IllegalActionDetection()

FaceRecognition =  FaceRec.Face_detector()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
flag=False
cont=0
dist=np.inf
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        star_time = time.time()
        ret,frame=cap.read()
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        activities = []

        # detect person name
        operatorName = "Unknown"
        names, boxes = FaceRecognition.detectface(image)
        if names:
            operatorName = names[0]


        # detect illegal action
        illegal_actions = IllegalActionDetection.detect(frame, res, objects_to_detect=["phoneusing"])
        if illegal_actions:
            if illegal_actions["smocking"] != None:
                activities.append({
                    "activityName": "smocking",
                    "gravity": 10
                })

            if illegal_actions["phoneusing"] != None:
                activities.append({
                    "activityName": "phone use",
                    "gravity": 10
                })

        # detect face action
        face_assessment = FaceAssessment.detect(frame, res, operatorName)
        if face_assessment and face_assessment[0] == "YAWNING":
            activities.append({
                "activityName": "YAWNING",
                "gravity": 5
            })

        for activity in activities:
            api.reportActivity(
                activity={
                    "gravity": activity["gravity"],
                    "activityName": activity["activityName"],
                    "operatorName": operatorName,
                }
            )

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # cv2.putText(image, str(round(1/(time.time()-star_time),2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        star_time = time.time()

        # cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
      
cv2.destroyAllWindows()
cap.release()


