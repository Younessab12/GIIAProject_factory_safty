"""

  Use same as yawning
  Test it on test.ipynb
  
  input :
    + face_landmarks : mp result of a single face after face_mesh processing 
      ( see test.ipynb for how we got it)

  output :
    + a dict of this from :
      {
        lips: lips_relative_distance,
        left_eye: left_eye_relative_distance,
        right_eye: right_eye_relative_distance
      }
      if no value of distance was detected return -1

"""
def get_eyes_lips_relative_distance(face_landmarks):
  lips_relative_distance = -1
  left_eye_relative_distance = -1 # distance of openess
  right_eye_relative_distance = -1 # distance of openess


  # processing


  return {
    "lips"     : lips_relative_distance,
    "left_eye" : left_eye_relative_distance,
    "right_eye": right_eye_relative_distance
  }