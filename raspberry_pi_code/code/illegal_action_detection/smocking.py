import mediapipe as mp
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO


class SmockingDetector:

    def __init__(self) -> None:
        self.model = YOLO('../assets/models/bestnano.pt')
    
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

    def detect(self,frame:np.array,right_hand_landmarks,left_hand_landmarks,face_landmarks):
        """
            this fct is used to detect the smoking action in a frame
            input: frame: np.array
                frame to be processed(when the distance between the mouth and the hand is small enough)
            output: list or None(if no smoking action is detected)
                list : bosxes.xyxy is the bounding boxe of the cigarette in the frame
                       boxes.conf is the confidence of the model in the detection
                      "smoking" is the label of the detected action
        """
        if right_hand_landmarks and face_landmarks :
            left_dist=self.calculate_distance(right_hand_landmarks.landmark[13],face_landmarks.landmark[13],face_landmarks.landmark[152],face_landmarks.landmark[10])
        if left_hand_landmarks and face_landmarks :
            right_dist=self.calculate_distance(left_hand_landmarks.landmark[13],face_landmarks.landmark[13],face_landmarks.landmark[152],face_landmarks.landmark[10])
        if left_dist<0.8 or right_dist:
            res = self.model(frame,stream=True,show=False)
            for r in res:
                boxes = r.boxes
                if len(boxes) > 0 and float(boxes.conf[0])>=0.5:
                    return [boxes.xyxy[0],boxes.conf[0],"smoking"]
        return None
    
    
    
    def draw_bounding_box(self,frame,detection_result):
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
    def main(self):
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
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
                if res.right_hand_landmarks and res.face_landmarks :
                    dist=self.calculate_distance(res.right_hand_landmarks.landmark[13],res.face_landmarks.landmark[13],res.face_landmarks.landmark[152],res.face_landmarks.landmark[10])
                if res.left_hand_landmarks and res.face_landmarks :
                    dist=self.calculate_distance(res.left_hand_landmarks.landmark[13],res.face_landmarks.landmark[13],res.face_landmarks.landmark[152],res.face_landmarks.landmark[10])
                
                if dist<0.8:
                    flag=True
                    
                if flag and cont<10:
                    cont+=1
                    detection_result=self.detect(frame)
                    if detection_result:
                        image=self.draw_bounding_box(image,list(detection_result))
                        print(list(detection_result))
                    if cont==10:
                        flag=False
                        cont=0
                        dist=np.inf

                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                cv2.putText(image, str(round(1/(time.time()-star_time),2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                star_time = time.time()

                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        cap.release()
