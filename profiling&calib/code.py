import time
import cv2
import math
import json
import os
from imutils import paths
import face_recognition
import pickle
import numpy as np

"""
    !!!
        To use calib:

            person_name = "your_name"
            calib = Calib(person_name)

            ## loop over the frames of the input
                calib.process(frame, holistic_res)
            ## end loop

            calib.export_json()
            calib.train_model_face_recognition()

"""


"""
  To calibrate we will record the following:
    + skin color detection
    + mouth movements calibration:
      - yawning
      - talking
      - ...
"""

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class FaceRecord:
  def __init__(self, title):
    self.title = title
    self.lips_record = []
    self.left_eye_record = []
    self.right_eye_record = []

  def add_to_record(self, result):
    self.lips_record.append(result['lips'])
    self.left_eye_record.append(result['left_eye'])
    self.right_eye_record.append(result['right_eye'])

  def get_record(self):
    return {
      "lips": self.lips_record,
      "left_eye": self.left_eye_record,
      "right_eye": self.right_eye_record
    }

SKIN_DETECTION, STANDARD_FACE, YAWNING, TALKING, ENDED = range(5)

class Calib:
  def __init__(self, person_name="default"):
    self.owner = person_name
    self.env_init()
    self.start_time= time.time()
    self.duration = 0
    self.state = 0
    
    part_skin = ['forehead', 'left_cheek', 'right_cheek', 'right_hand', 'left_hand']
    self.record = { part: [] for part in part_skin }
    self.color = { part: None for part in part_skin }

    self.face_record = {
      STANDARD_FACE: FaceRecord(STANDARD_FACE),
      YAWNING: FaceRecord(YAWNING),
      TALKING: FaceRecord(TALKING)
    }

    self.messages = [
      "show face and hands for skin detection",
      "show face in natural position",
      "show face in yawning position",
      "read the following text: 'The quick brown fox jumps over the lazy dog'",
      "ended"
    ]
    
    self.number_frame_required = {
      SKIN_DETECTION: 50,
      STANDARD_FACE: 100,
      YAWNING: 100,
      TALKING: 250,
      ENDED: 0
    }
  
  def process(self, frame, holistic_res):
    self.duration = time.time()- self.start_time

    self.display_image_with_text(frame, f'dur : {self.duration}', "calib")
    self.display_image_with_text(frame, self.messages[self.state], "calib", i=2)

    face_landmarks = holistic_res.face_landmarks
    pose_landmarks = holistic_res.pose_landmarks

    calibrations = [
      self.skin_color_detection,
      self.calibrate_general,
      self.calibrate_yawning,
      self.calibrate_talking,
    ]

    if self.state < len(calibrations):
      state_over = calibrations[self.state](frame, face_landmarks, pose_landmarks)
      if state_over:
        print(f"state {self.state} over")
        self.state += 1
        time.sleep(1)
        print(f"state {self.state} started")
        print(self.messages[self.state])

    return self.get_state() == len(calibrations)

  def train_model_face_recognition(self):
    """
        this fct will train the model and save the encodings in a pickle file
        param:
             path_to_imgs_folder: path to the folder containing the images

    """
    print("[INFO] start processing faces...")
    imagePaths = list(paths.list_images(self.face_recognition_dataset_folder))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(f"{self.folder}/encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

  def get_results(self):
    return {
      'person': self.owner,
      'record_time' : self.start_time,
      "duration": self.duration,
      SKIN_DETECTION: self.color,
      YAWNING: self.face_record[YAWNING].get_record(),
      TALKING: self.face_record[TALKING].get_record(),
      STANDARD_FACE: self.face_record[STANDARD_FACE].get_record()
    }

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
      if len(colors) > self.number_frame_required[SKIN_DETECTION]:
        self.color[part] = np.average(colors, axis=0)

    print(self.color)

    return not any([color is None for color in self.color.values()])
    
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

  def calibrate_yawning(self, frame, face_landmarks, pose_landmarks):
    return self.calibrate_face_action(face_landmarks, YAWNING)

  def calibrate_talking(self, frame, face_landmarks, pose_landmarks):
    return self.calibrate_face_action(face_landmarks, TALKING)

  def calibrate_general(self, frame, face_landmarks, pose_landmarks):
    self.save_image_recogniton(frame)
    return self.calibrate_face_action(face_landmarks, STANDARD_FACE)

  def calibrate_face_action(self, face_landmarks, action):
    if not face_landmarks: return False
    self.face_record[action].add_to_record(self.get_eyes_lips_relative_distance(face_landmarks))
    return len(self.face_record[action].lips_record) > self.number_frame_required[action]

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

  def display_image_with_text(self, img, text, title, i=1):
    cv2.putText(
      img = img,
      text = text,
      org = (10, 10+i*20),
      fontFace = cv2.FONT_HERSHEY_DUPLEX,
      fontScale = 1.0,
      color = (125, 246, 55),
      thickness = 1
    )
    cv2.imshow(title, img)

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

  def save_image_recogniton(self, frame):
    cv2.imwrite(f'{self.face_recognition_folder}/{time.time()}.jpg', frame)

  def env_init(self):
    folder = 'calib_records'
    if not os.path.exists(folder):
      os.makedirs(folder)

    face_recognition_folder = f'{folder}/data/{self.owner}'
    if not os.path.exists(face_recognition_folder):
      os.makedirs(face_recognition_folder)

    self.folder = folder
    self.face_recognition_dataset_folder = f'{folder}/data'
    self.face_recognition_folder = face_recognition_folder
    self.record_file = f'{folder}/calibration_{self.owner}'

  def export_json(self):
    data = self.get_results()
    with open(f'{self.record_file}.json', 'w') as outfile:
      json.dump(data, outfile, cls=NumpyEncoder)

