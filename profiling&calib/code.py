import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

"""
  calibration takes in consideration:
    1- skin color
    2- yawning
    3- talking
    4- general ( static ) 
"""

import time
import cv2

"""
  To calibrate we will record the following:
    + skin color detection
    + mouth movements calibration:
      - yawning
      - talking
      - ...
"""

class Calib:
  def __init__(self):
    self.start_time= time.time()
    self.duration = 0
    self.state = 0
    
    part_skin = ['forehead', 'left_cheek', 'right_cheek', 'right_hand', 'left_hand']
    self.record = { part: [] for part in part_skin }
    self.color = { part: None for part in part_skin }
    

  def process(self, frame, holistic_res):
    self.duration = time.time()- self.start_time
    
    self.display_image_with_text(frame, f'dur : {self.duration}', "calib")

    face_landmarks = holistic_res.face_landmarks
    pose_landmarks = holistic_res.pose_landmarks

    if self.state == 0:
      skin = self.skin_color_detection(frame, face_landmarks, pose_landmarks)

    return self.get_state()

  def skin_color_detection(self, frame, face_landmarks, pose_landmarks):
    """
      Get frame and face_landmarks
      Crop the forehead area and cheeks and get the average color for each
      return the array of the average colors
    """

    skin_color = self.get_skin_color_from_frame(frame, face_landmarks, pose_landmarks)
    for part, part_color  in skin_color.items():
      if part_color is not None and self.color[part] is None:
        self.record[part].append(part_color)

    for part, colors in self.record.items():
      if len(colors) > 50:
        self.color[part] = np.average(colors, axis=0)

    print(self.color)
    

  def get_skin_color_from_frame(self, frame, face_landmarks, pose_landmarks):
    """
      Get frame and face_landmarks
      Crop the forehead area and cheeks and get the average color for each
      return the array of the average colors
    """
    record = {}
    forehead, left_cheek, right_cheek, right_hand, left_hand = None, None, None, None, None

    if face_landmarks:
      forehead = self.crop_forehead(frame, face_landmarks)
      left_cheek = self.crop_left_cheek(frame, face_landmarks)
      right_cheek = self.crop_right_cheek(frame, face_landmarks)
    
    if pose_landmarks:
      right_hand = self.crop_right_hand(frame, pose_landmarks)
      left_hand = self.crop_left_hand(frame, pose_landmarks)

    record['forehead'] = self.get_average_color(forehead)
    record['left_cheek'] = self.get_average_color(left_cheek)
    record['right_cheek'] = self.get_average_color(right_cheek)
    record['right_hand'] = self.get_average_color(right_hand)
    record['left_hand'] = self.get_average_color(left_hand)

    return record

  def calibrate_yawning(self):
    pass

  def calibrate_talking(self):
    pass

  def calibrate_general(self):
    pass

  def crop_forehead(self, frame, face_landmarks):
    """
      Get frame and face_landmarks
      Crop the forehead area and return it
    """
    forehead_landmark = face_landmarks.landmark[151]
    forehead = self.crop_part_from_image(frame, forehead_landmark, 20)
    return forehead
  
  def crop_left_cheek(self, frame, face_landmarks):
    """
      Get frame and face_landmarks
      Crop the left cheek area and return it
    """
    left_cheek_landmark = face_landmarks.landmark[118]
    left_cheek = self.crop_part_from_image(frame, left_cheek_landmark, 20)
    return left_cheek
  
  def crop_right_cheek(self, frame, face_landmarks):
    """
      Get frame and face_landmarks
      Crop the right cheek area and return it
    """
    right_cheek_landmark = face_landmarks.landmark[348]
    right_cheek = self.crop_part_from_image(frame, right_cheek_landmark, 20)
    return right_cheek
  
  def crop_right_hand(self, frame, pose_landmarks):
    """
      Get frame and pose_landmarks
      Crop the right hand area and return it
    """
    right_hand_landmark = pose_landmarks.landmark[15]
    right_hand = self.crop_part_from_image(frame, right_hand_landmark, 20)
    return right_hand
  
  def crop_left_hand(self, frame, pose_landmarks):
    """
      Get frame and pose_landmarks
      Crop the left hand area and return it
    """
    left_hand_landmark = pose_landmarks.landmark[16]
    left_hand = self.crop_part_from_image(frame, left_hand_landmark, 20)
    return left_hand

  def crop_part_from_image(self, frame, point, width):
    """
      Get frame and point
      Crop the area around the point and return it
    """
    part = frame[
      int(point.y * frame.shape[0])-width: int(point.y * frame.shape[0]+width),
      int(point.x * frame.shape[1])-width: int(point.x * frame.shape[1]+width)
    ]
    if len(part) <= 0 or len(part[0]) <= 0: return None
    return part

  def get_average_color(self, frame):
    if frame is None: return None
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

  def get_state(self):
    pass

  def display_image_with_text(self, img, text, title):
    cv2.putText(
      img = img,
      text = text,
      org = (200, 200),
      fontFace = cv2.FONT_HERSHEY_DUPLEX,
      fontScale = 1.0,
      color = (125, 246, 55),
      thickness = 1
    )
    cv2.imshow(title, img)
