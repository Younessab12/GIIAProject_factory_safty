import face_assessment.face_knn as face_knn
import os
import json
import math

class FaceAssessment:
  def __init__(self):
    # for each person load person face assessment data
    self.models = {}

    # loop over ../assets/calib/data
    person_list = []
    for file_name in os.listdir('assets/calib/data'):
      if file_name.endswith('.json'):
        person_list.append(
          {
            "name": file_name.split('.')[0].split('calibration_')[1],
            "link": 'assets/calib/data/' + file_name
          }
        )

    for person in person_list:
      self.models[person["name"]] = face_knn.Face_Assessment(person["link"])

    print(person_list)

  def detect(self, frame, res, operatorName):
    data = self.get_eyes_lips_relative_distance(res.face_landmarks)

    return self.models[operatorName].get_results(operatorName, data["lips"], data["left_eye"], data["right_eye"])

  def assess_face(self, person, lips, left_eye, right_eye):
    return self.models[person].get_results(lips, left_eye, right_eye)

  def calculate_distance(self, point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2+(point1.z - point2.z)**2)

  def get_eyes_lips_relative_distance(self, face_landmarks):
    if not face_landmarks:
        return {
            "lips": -1,
            "left_eye": -1,
            "right_eye": -1
        }

    #relevant points for lips, left eye, and right eye
    upper_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    upper_left_eye_point = face_landmarks.landmark[386]
    bottom_left_eye_point = face_landmarks.landmark[374]
    upper_right_eye_points = face_landmarks.landmark[159]
    bottom_right_eye_points = face_landmarks.landmark[145]
    upper_face = face_landmarks.landmark[10]
    bottom_face = face_landmarks.landmark[152]

    #Relative distances
    lips_distance = self.calculate_distance(upper_lip, bottom_lip) / self.calculate_distance(upper_face, bottom_face)
    left_eye_distance = self.calculate_distance(upper_left_eye_point, bottom_left_eye_point) / self.calculate_distance(upper_face, bottom_face)
    right_eye_distance = self.calculate_distance(upper_right_eye_points, bottom_right_eye_points) / self.calculate_distance(upper_face, bottom_face)

    return {
        "lips": lips_distance,
        "left_eye": left_eye_distance,
        "right_eye": right_eye_distance
    }
