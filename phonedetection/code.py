from ultralytics import YOLO
import cv2
import numpy as np
import time


class PhoneDetector:
    def __init__(self,model_path,cam_id=0):
        self.model = YOLO(model_path)
        self.cam_id = cam_id
    

    def detect(self):
        print("starting detection")
        curr_time = time.time()
        cap = cv2.VideoCapture(self.cam_id)
## phone detection 
        while True:
            ret,frame = cap.read()
            frame = cv2.flip(frame,1)
            res = self.model(frame,stream=True,show=False)
            for r in res:
                boxes = r.boxes
                if len(boxes) > 0:
                    print("phone detected",time.time()-curr_time )
                    cv2.putText(frame, "phone detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "./bestnanophone.pt"  
    detector = PhoneDetector(model_path)
    detector.detect()

