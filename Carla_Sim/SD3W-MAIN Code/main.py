import eel
import cv2
import mediapipe as mp
from S3DW.S3DW_models import S3DWStack
import numpy as np 


data_payload = "S3DW is loading"

eel.init("web")


def my_thread():
    global data_payload


    agent = S3DWStack()
    mp_face_mesh = mp.solutions.face_mesh
    vid = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70,
        refine_landmarks=True,
    ) as face_mesh:
        agent.camera_on = True
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_coords = results.multi_face_landmarks

            if face_coords != None:
                data_payload = agent.show_data(frame, results)
                level, orientation,eyes, inclination =  data_payload.split()
                eel.my_javascript_function(eyes, level, orientation, inclination)
            eel.sleep(0.2)

    vid.release()
    cv2.destroyAllWindows()
    
eel.spawn(my_thread)



eel.start("index.html")