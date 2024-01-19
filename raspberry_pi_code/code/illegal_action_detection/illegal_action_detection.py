import illegal_action_detection.phoneusing as phoneusing
import illegal_action_detection.smocking as smocking

class IllegalActionDetection:

    def __init__(self):
        self.phoneusing = phoneusing.PhoneDetector()
        self.smocking = smocking.SmockingDetector()

    def detect(self, frame, left_hand_landmarks,right_hand_landmarks,face_landmarks,objects_to_detect = ['phoneusing', 'smocking']):
        detection = {
            'phoneusing': None,
            'smocking': None
        }
        if 'phoneusing' in objects_to_detect:
            detection['phoneusing'] = self.phoneusing.detect(frame)
        if 'smocking' in objects_to_detect:
            detection['smocking'] = self.smocking.detect(frame,right_hand_landmarks,left_hand_landmarks,face_landmarks)
        return detection