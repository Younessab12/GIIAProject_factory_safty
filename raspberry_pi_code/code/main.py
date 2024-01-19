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

import tools.update_assets as update_assets
update_assets.update_assets()

apikey = os.environ['API_KEY']
camId = int(os.environ['CAM_ID'])
apiurl = os.environ['API_URL']

print("connecting to db")
api = API.API(baseUrl=apiurl, apiKey=apikey)

FaceAssessment =  FaceAssess.FaceAssessment()

IllegalActionDetection =  IllegalAction.IllegalActionDetection()

FaceRecognition =  FaceRec.Face_detector()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap=cv2.VideoCapture(camId)
flag=False
cont=0
dist=np.inf
def draw_bounding_box(frame,detection_result):
    """
        this fct is used to draw the bounding box of the detected smoking action
        input: frame: np.array
                detection_result: list(containing the bounding boxe, label and confidence of the detected action)
        output: frame: np.array
    """
    boxes,confidences,label = list(detection_result[0]),round(float(detection_result[1]),2),detection_result[2]
    # for box in boxes:
    cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0,255,0), 2)
    cv2.putText(frame,label+' '+str(round(confidences,2)),(int(boxes[0]), int(boxes[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return frame
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
            print(names)
            operatorName = names[0]
            print(operatorName)


        # detect illegal action
        illegal_actions = IllegalActionDetection.detect(frame,res.left_hand_landmarks,res.right_hand_landmarks,res.face_landmarks)
        print(illegal_actions)
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

        if len(activities) == 0 :
            api.ping(operatorName)
        if illegal_actions["smocking"] != None:
            image=draw_bounding_box(image,illegal_actions["smocking"])
        if illegal_actions["phoneusing"]!=None:
            image=draw_bounding_box(image,illegal_actions["phoneusing"])

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.putText(image, str(round(1/(time.time()-star_time),2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        star_time = time.time()

        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
cv2.destroyAllWindows()
cap.release()


