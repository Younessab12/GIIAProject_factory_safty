import cv2
import mediapipe as mp
from S3DW.S3DW_models import S3DWStack



models_info = {"eyes" : "models/eyes55.sav", "head_position" : "models/head_pos.sav", "head_orientation" : "models/orientation.h5"}

agent = S3DWStack()

mp_face_mesh = mp.solutions.face_mesh
url = "videos/10.mp4"
vid = cv2.VideoCapture(0)
draw = True


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
        img_height, img_width = frame.shape[:2]

        if face_coords != None:
            txt = agent.show_data(frame, results)
            print(txt)
            frame = cv2.putText(
                img = frame,
                text = txt,
                org = (30, 20),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1.0,
                color = (125, 246, 55),
                thickness = 3
                )



        cv2.imshow("test", frame)

        if cv2.waitKey(1) == ord("q"):
            agent.thread_on = False
            break
vid.release()
cv2.destroyAllWindows()

