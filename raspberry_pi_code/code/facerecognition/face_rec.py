import pickle 
import time
import face_recognition

class Face_detector:
    def __init__(self):
        
        self.encodingsP = './assets/models/encodings.pickle'
        self.data = pickle.loads(open(self.encodingsP, "rb").read())

    def detectface(self,frame):
        boxes = face_recognition.face_locations(frame)
        currentname = "unknown"
        encodings = face_recognition.face_encodings(frame, boxes)
        names=[]
        mp = {}

        for encoding in encodings:
            matches = face_recognition.compare_faces(self.data["encodings"],
                encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                if currentname != name:
                    currentname = name

            names.append(name)
            return names, boxes
        
        return None, None