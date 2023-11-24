import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
# import matplotlib.pyplot as plt



# def draw_landmarks_on_image(rgb_image, detection_result):
#   face_landmarks_list = detection_result.landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected faces to visualize.
#   for idx in range(len(face_landmarks_list)):
#     face_landmarks = face_landmarks_list[idx]

#     # Draw the face landmarks.
#     face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     face_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#     ])

#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_contours_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp.solutions.drawing_styles
#           .get_default_face_mesh_iris_connections_style())

#   return annotated_image


# Initialiser la fenêtre de tracé interactif
plt.ion()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

# Initialiser des variables pour stocker les données
timestamps = []
lip_distances = []

start_time = time.time()
bail_duration_threshold = 2.0  # Seuil de durée pour détecter le bâillement

# Variables pour la détection des pics
peaks, _ = find_peaks(lip_distances, distance=100)  # La distance dépend de la fréquence d'échantillonnage

# Variables pour la détection du bâillement
bail_start_time = None

# Initialiser le graphique
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Lip Distance')
ax.set_title('Lip Distance Over Time')

while cap.isOpened():
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Indices des points de repère des lèvres
            left_lip_index = 61  # Modifier selon votre modèle spécifique
            right_lip_index = 91  # Modifier selon votre modèle spécifique

            # Coordonnées des points de repère
            left_lip = face_landmarks.landmark[left_lip_index]
            right_lip = face_landmarks.landmark[right_lip_index]

            # Calcul de la distance entre les deux lèvres
            lip_distance = np.sqrt((right_lip.x - left_lip.x)**2 + (right_lip.y - left_lip.y)**2)

            # Ajouter les données à la liste
            current_time = time.time() - start_time
            timestamps.append(current_time)
            lip_distances.append(lip_distance)

            # Mettre à jour le graphique
            line.set_xdata(timestamps)
            line.set_ydata(lip_distances)
            ax.relim()
            ax.autoscale_view()

            # Dessiner les landmarks sur l'image
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            # draw_landmarks_on_image(frame_rgb,results)




        # Mettre à jour l'affichage de Matplotlib
        fig.canvas.flush_events()

        # Détecter les pics
        new_peaks, _ = find_peaks(lip_distances, distance=100)  # La distance dépend de la fréquence d'échantillonnage
        if len(new_peaks) > len(peaks):
            # Un pic a été détecté, réinitialiser le compteur du bâillement
            bail_start_time = None

        peaks = new_peaks

        # Vérifier la distance actuelle par rapport au seuil
        current_distance = lip_distances[-1]
        if current_distance > 0.03:
            if bail_start_time is None:
                # Début du bâillement
                bail_start_time = time.time()
            else:
                # Le bâillement dure depuis le début
                bail_duration = time.time() - bail_start_time
                if bail_duration > bail_duration_threshold:
                    # Afficher le message sur la caméra
                    cv2.putText(frame, "Bâillement détecté!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            # La distance est en dessous du seuil, réinitialiser le compteur du bâillement
            bail_start_time = None

    # Afficher l'image avec les landmarks et le message
    cv2.putText(frame,str(fps),(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Landmarks Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la fenêtre interactive Matplotlib à la fin
plt.ioff()
plt.show()

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
