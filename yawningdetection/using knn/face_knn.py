import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class Face_Assessment:
    STD=0
    YAWNING=1
    TALKING=2
    def __init__(self, file_name):
        # load data from json file
        with open(file_name) as f:
            data = json.load(f)

        std = data['1']
        yawning = data['2']
        talking = data['3']

        # create training data
        X = []
        y = []

        lips = "lips"
        left_eye = "left_eye"
        right_eye = "right_eye"

        std_arr = np.transpose(np.array([std[lips], std[left_eye], std[right_eye]]))
        yawning_arr = np.transpose(np.array([yawning[lips], yawning[left_eye], yawning[right_eye]]))
        talking_arr = np.transpose(np.array([talking[lips], talking[left_eye], talking[right_eye]]))

        STD=0
        YAWNING=1
        TALKING=2

        for i in std_arr:
            X.append(i)
            y.append(STD)
        for i in yawning_arr:
            X.append(i)
            y.append(YAWNING)
        for i in talking_arr:
            X.append(i)
            y.append(TALKING)

        neigh = KNeighborsClassifier(n_neighbors=5)
        self.detector = neigh.fit(X, y)

    def assess(self, lips, left_eye, right_eye):
        arr = np.array([[lips, left_eye, right_eye]])
        return self.detector.predict(arr)
    
    def get_results(self,lips, left_eye, right_eye):
        mp = {
            0: "STD",
            1: "YAWNING",
            2: "TALKING"
        }
        results = []
        assets = self.assess(lips, left_eye, right_eye)
        for i in assets:
            results.append(mp[int(i)])
        return results
