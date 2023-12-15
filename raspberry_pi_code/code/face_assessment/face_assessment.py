import face_knn

class FaceAssessment:
  def __init__(self, person_list):
    # for each person load person face assessment data
    self.models = {}
    for person in person_list:
      self.models[person.name] = face_knn.Face_Assessment(person.link)

  def assess_face(self, person, lips, left_eye, right_eye):
    return self.models[person].get_results(lips, left_eye, right_eye)

