import phoneusing
import smocking

class IllegalActionDetection:
    def __init__(self):
        self.phoneusing = phoneusing.PhoneUsing()
        self.smocking = smocking.Smocking()

    def detect(self, frame):
        detection = {}
        detection['phoneusing'] = self.phoneusing.detect(frame)
        detection['smocking'] = self.smocking.detect(frame)
