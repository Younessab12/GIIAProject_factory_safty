import illegal_action_detection.phoneusing as phoneusing
import illegal_action_detection.smocking as smocking

class IllegalActionDetection:
    def __init__(self):
        self.phoneusing = phoneusing.PhoneDetector()
        self.smocking = smocking.SmockingDetector()

    def detect(self, frame, mp):
        detection = {}
        # detection['phoneusing'] = self.phoneusing.detect(frame)
        detection['smocking'] = self.smocking.detect(frame)
        return detection