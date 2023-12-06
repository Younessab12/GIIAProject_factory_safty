import mediapipe as mp
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO


class Detector:

    def __init__(self,model_path) -> None:
        self.model = YOLO(model_path)

    def detect(self,frame:np.array):
        """
            this fct is used to detect the smoking action in a frame
            input: frame: np.array
                frame to be processed(when the distance between the mouth and the hand is small enough)
            output: list or None(if no smoking action is detected)
                list : bosxes.xyxy is the bounding boxe of the cigarette in the frame
                       boxes.conf is the confidence of the model in the detection
                      "smoking" is the label of the detected action
        """
        res = self.model(frame,stream=True,show=False)
        for r in res:
            boxes = r.boxes
            if len(boxes) > 0 and float(boxes.conf[0])>=0.5:
                return [boxes.xyxy[0],boxes.conf[0],"smoking"]
        return None
    
    def calculate_distance(self,index_cord,mouth_cord,face_down_cord,face_up_cord):
        """
            this fct is used to calculate the distance between the mouth and the index finger relative to the face
            input: index_cord: mp_holistic.hand_landmarks.landmark[8] (this input represents the index finger landmark)
                   mouth_cord: mp_holistic.face_landmarks.landmark[13] (this input represents the mouth landmark)
                   face_down_cord: mp_holistic.face_landmarks.landmark[152] (this input represents the face down landmark)
                   face_up_cord: mp_holistic.face_landmarks.landmark[10] (this input represents the face up landmark)
            output: distance: float (the distance between the mouth and the index finger relative to the face )
        """
        distance = np.sqrt((index_cord.x - mouth_cord.x)**2 + (index_cord.y - mouth_cord.y)**2+ (index_cord.z - mouth_cord.z)**2)/(np.sqrt((face_down_cord.x - face_up_cord.x)**2 + (face_down_cord.y - face_up_cord.y)**2+ (face_down_cord.z - face_up_cord.z)**2))
        return distance
    
    def draw_bounding_box(self,frame,boxes,label,confidences):
        """
            this fct is used to draw the bounding box of the detected smoking action
            input: frame: np.array
                   boxes: list
                   label: str
                   confidences: float
            output: frame: np.array
        """
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(frame,label+' '+str(round(confidences,2)),(int(box[0]), int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return frame
    