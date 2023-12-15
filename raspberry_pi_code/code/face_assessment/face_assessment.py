import face_knn

class FaceAssessment:
  def __init__(self, person_list):
    # for each person load person face assessment data
    self.models = {}
    self.models["person1"] = face_knn.Face_Assessment("link to person1 face assessment data")
    pass

  def assess_face(self, person, lips, left_eye, right_eye):
    return self.models[person].get_results(lips, left_eye, right_eye)

