import math
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
# """

#   Use same as yawning
#   Test it on test.ipynb
  
#   input :
#     + face_landmarks : mp result of a single face after face_mesh processing 
#       ( see test.ipynb for how we got it)

#   output :
#     + a dict of this from :
#       {
#         lips: lips_relative_distance,
#         left_eye: left_eye_relative_distance,
#         right_eye: right_eye_relative_distance
#       }
#       if no value of distance was detected return -1

# """

# # def get_eyes_lips_relative_distance(face_landmarks):
# #   lips_relative_distance = -1
# #   left_eye_relative_distance = -1 # distance of openess
# #   right_eye_relative_distance = -1 # distance of openess


# #   # processing


# #   return {
# #     "lips"     : lips_relative_distance,
# #     "left_eye" : left_eye_relative_distance,
# #     "right_eye": right_eye_relative_distance
# #   }

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2+(point1.z - point2.z)**2)


def get_eyes_lips_relative_distance(face_landmarks):
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
    lips_distance = calculate_distance(upper_lip, bottom_lip) / calculate_distance(upper_face, bottom_face)
    left_eye_distance = calculate_distance(upper_left_eye_point, bottom_left_eye_point) / calculate_distance(upper_face, bottom_face)
    right_eye_distance = calculate_distance(upper_right_eye_points, bottom_right_eye_points) / calculate_distance(upper_face, bottom_face)

    return {
        "lips": lips_distance,
        "left_eye": left_eye_distance,
        "right_eye": right_eye_distance
    }
