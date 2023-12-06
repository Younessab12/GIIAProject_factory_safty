import numpy as np
import pickle
from tensorflow import keras
from math import pi, sqrt
import cv2 as cv

from S3DW.S3DW_utils import *


class S3DWStack:
    """
    This class represents a stack of models used for vision-based tasks, specifically eye and head position and orientation detection.

    Attributes:
        - frame: A numpy array representing a single frame of video input
        - results: A dictionary of results obtained from running the models on the frame
        - models_info: A dictionary containing information about the models to be loaded
        - models: A dictionary containing the loaded models for each task
        - models_names: A list of names of the loaded models

    Methods:
        - load_models(): Loads the models specified in models_info and returns a dictionary containing the loaded models
        - update_vision(frame, results): Updates the frame and results attributes of the object with the given parameters
        - show_data(frame, results): Uses the loaded models to predict eye and head position and orientation, and returns a formatted string containing the predictions
    """

    def __init__(self):
        self.frame = None
        self.results = None
        self.models_info = models_info
        self.models = self.load_models()
        self.models_names = models_info.keys()

    def load_models(self):
        models = {}
        for key in self.models_info:
            model_path = self.models_info[key]
            if key == "eyes":
                models[key] = EyesModel(model_path)
            if key == "head_position":
                models[key] = HeadPositionModel(model_path)
            if key == "head_orientation":
                models[key] = OrientationModel(model_path)
            if key == "eyescnn":
                models[key] = EyesModelCNN(model_path)
        return models

    def update_vision(self, frame, results):
        self.frame = frame
        self.results = results

    def show_data(self, frame, results):
        self.update_vision(frame, results)
        angle = 90
        message = ""

        for key in self.models:
            model = self.models[key]
            if key == "head_orientation":
                model.frame = frame
                model.get_mesh_coords(results)
                angle = model.get_angle()
            elif key == "eyescnn":
                model.frame = frame
                model.get_mesh_coords(results)

            else:
                model.get_mesh_coords(results)
            ind, txt = model.pred()
            message += txt + " "

        message += "{:.2f}".format(angle)
        return message


class S3DWModel:
    """
    Base class for all the models used in the S3DWStack

    Args:
        model_path (str): Path of the trained model file
        frame (np.ndarray, optional): Frame to use the model on. Defaults to None.
        draw (bool, optional): Whether to draw the predicted results. Defaults to False.

    Attributes:
        model_path (str): Path of the trained model file
        model: Trained model object
        normalized_landmarks (list): List of normalized landmark coordinates
        mesh_coord (dict): Mesh coordinates
        prediction: Model prediction
        c1 (float): Model constant 1
        c2 (float): Model constant 2

    Methods:
        load_model: Loads the trained model
        update_normalized_landmarks: Updates normalized_landmarks attribute
        make_vector: Calculates the vector between two points
        mean_coord: Calculates the mean of the coordinates
        std_dev_coord: Calculates the standard deviation of the coordinates
        cross_prod: Calculates the cross product of two vectors
        distance: Calculates the distance between two points
        points_def: Abstract method that defines the key points for the model
        calc_inputs: Abstract method that calculates the inputs for the model
        update_inputs: Updates the model constants
        pred: Abstract method that returns the predicted label and index
        normalize: Abstract method that normalizes the values between 0 and 1
        drawlandmarks: Abstract method that draws the predicted results

    """

    def __init__(self, model_path, frame=None, draw=False):
        self.model_path = model_path
        self.model = self.load_model()
        self.normalized_landmarks = None
        self.mesh_coord = None
        self.prediction = None
        # initial value

        self.c1 = 1
        self.c2 = 1

    def load_model(self):
        """
        Loads the trained model

        Returns:
            model: Trained model object
        """
        if self.model_path.endswith(".h5"):
            model = keras.models.load_model(self.model_path)
        elif self.model_path.endswith(".sav"):
            model = pickle.load(open(self.model_path, "rb"))
        else:
            raise Exception("Model extension not supported")
        print(f"Model Loaded {self.model_path}")
        return model

    def update_normalized_landmarks(self, results):
        """
        Updates the normalized_landmarks attribute

        Args:
            results: Multi face landmarks results
        """
        self.normalized_landmarks = [
            (int(point.x), int(point.y))
            for point in results.multi_face_landmarks[0].landmark
        ]
        return None

    def make_vector(self, A, B) -> tuple:
        """
        Calculates the vector between two points

        Args:
            A (tuple): First point coordinates
            B (tuple): Second point coordinates

        Returns:
            tuple: Vector between two points
        """
        AB = (B[0] - A[0], B[1] - A[1])
        return AB

    def mean_coord(self, coords, mesh_coord) -> tuple:
        """
        Calculates the mean of the coordinates

        Args:
            coords (list): List of coordinates
            mesh_coord (dict): Mesh coordinates

        Returns:
            tuple: Mean coordinates
        """
        X = np.mean(np.array([mesh_coord[p][0] for p in coords]))
        Y = np.mean(np.array([mesh_coord[p][1] for p in coords]))

        return (int(X), int(Y))

    def std_dev_coord(self, coords, mesh_coord) -> tuple:
        """
        Returns the standard deviation of the coordinates of a set of landmark points.

        Args:
            coords: A list of landmark point indices.
            mesh_coord: A dictionary containing the mesh coordinates of all the landmark points.

        Returns:
            A tuple representing the standard deviation of the coordinates of the set of landmark points.
        """
        X = np.std(np.array([mesh_coord[p][0] for p in coords]))
        Y = np.std(np.array([mesh_coord[p][1] for p in coords]))

        return (X, Y)

    def cross_prod(self, p1, p2) -> float:
        """
        Computes the cross product of two vectors represented by points p1 and p2.

        Args:
            p1: A tuple representing the coordinates of point 1.
            p2: A tuple representing the coordinates of point 2.

        Returns:
            A float representing the cross product of the two vectors.
        """
        return p1[0] * p2[1] - p1[1] * p2[0]

    def distance(self, A, B) -> float:
        """
        Computes the Euclidean distance between two points A and B.

        Args:
            A: A tuple representing the coordinates of point A.
            B: A tuple representing the coordinates of point B.

        Returns:
            A float representing the Euclidean distance between the two points.
        """
        return sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)

    def points_def(self):
        """
        Defines the landmark points required for the model.

        Returns:
            None.
        """
        pass

    def calc_inputs(self):
        """
        Calculates the inputs required for the model.

        Returns:
            None if there is an exception. Otherwise, a tuple representing the input values.
        """
        pass

    def update_inputs(self):
        """
        Updates the input values required for the model.

        Returns:
            None.
        """
        new_input = self.calc_inputs()
        if new_input != None:
            self.c1, self.c2 = new_input

    def pred(self):
        """
        Takes the shape of the input layer and returns the index and label.

        Returns:
        - ind (int): index of predicted class.
        - txt (str): label of predicted class.
        """
        pass

    def normalize(self):
        """Normalizes the values between 0 and 1"""
        pass

    def drawlandmarks(self):
        """
        Draws the detected landmarks on the frame.
        """
        pass


class EyesModelCNN(S3DWModel):
    # Needs the frame
    def __init__(self, model_path, frame=None, draw=False):
        super().__init__(model_path, frame, draw)
        self.model_lst = ["Opened", "Closed"]
        self.points = iris_points

    def get_mesh_coords(self, results):
        img_height, img_width = self.frame.shape[:2]
        mesh_coord = np.array(
            [
                np.multiply([p.x, p.y], [img_width, img_height]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )
        self.mesh_coord = mesh_coord
        return mesh_coord

    def calc_inputs(self):
        left_eye = self.points[0:4]
        right_eye = self.points[4:]
        mesh_coord = self.mesh_coord

        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_coord[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_coord[RIGHT_IRIS])
        l_radius *= 2.5
        r_radius *= 2.5

        try:
            save_1 = self.frame[
                int(r_cy - r_radius) : int(r_cy + r_radius),
                int(r_cx - r_radius) : int(r_cx + r_radius),
            ]
            save_1 = cv.resize(save_1, dsize=(35, 35), interpolation=cv.INTER_CUBIC)

            save_2 = self.frame[
                int(l_cy - l_radius) : int(l_cy + l_radius),
                int(l_cx - l_radius) : int(l_cx + l_radius),
            ]
            save_2 = cv.resize(save_2, dsize=(35, 35), interpolation=cv.INTER_CUBIC)

            return save_1, save_2
        except:
            return None

    def image_preprocessing(self, img):
        # to Gray scale
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # normalisation
        img_np = img_gray.astype("float32") / 255
        # resizin
        input_vector = np.array([img_np]).reshape((35, 35, 1))
        input_vector = np.expand_dims(input_vector, axis=0)

        return input_vector

    def decision(self, img):
        input_vector = self.image_preprocessing(img)
        proba = self.model.predict(input_vector)[0][0]
        return proba

    def pred(self):
        self.update_inputs()
        left = self.decision(self.c1)
        right = self.decision(self.c2)
        index = min(left, right)
        # print(index)
        prediction = 1 if index <= 0.25 else 0
        self.prediction = prediction
        return prediction, self.model_lst[prediction]

    def normalize(self):
        """Already between 0 and 1"""
        ind = self.prediction
        return ind


class EyesModel(S3DWModel):
    def __init__(self, model_path, frame=None, draw=False):
        super().__init__(model_path, frame, draw)
        self.model_lst = ["Opened", "Closed"]
        self.points = eyes_points

    def get_mesh_coords(self, results):
        mesh_coord = np.array(
            [
                np.multiply([p.x, p.y], [1080, 1920]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )
        self.mesh_coord = mesh_coord
        return mesh_coord

    def calc_inputs(self):
        left_eye = self.points[0:4]
        right_eye = self.points[4:]
        mesh_coord = self.mesh_coord
        width_dist_left = self.distance(
            mesh_coord[left_eye[0]], mesh_coord[left_eye[1]]
        )
        height_dist_left = self.distance(
            mesh_coord[left_eye[2]], mesh_coord[left_eye[3]]
        )
        width_dist_right = self.distance(
            mesh_coord[right_eye[0]], mesh_coord[right_eye[1]]
        )
        height_dist_right = self.distance(
            mesh_coord[right_eye[2]], mesh_coord[right_eye[3]]
        )

        try:
            left_ratio = width_dist_left / height_dist_left
            right_ratio = width_dist_right / height_dist_right
            return left_ratio, right_ratio
        except:
            return None

    def pred(self):
        self.update_inputs()
        input = np.array([[self.c1, self.c2]])
        prediction = self.model.predict(input)[0]
        self.prediction = prediction
        return prediction, self.model_lst[prediction]

    def normalize(self):
        """Already between 0 and 1"""
        ind = self.prediction
        return ind


class OrientationModel(S3DWModel):
    def __init__(self, model_path, frame=None, draw=False):
        super().__init__(model_path, frame, draw)
        self.model_lst = ["Center", "Right", "Left"]
        self.points = head_or_points

    def get_mesh_coords(self, results):
        img_height, img_width = self.frame.shape[:2]
        mesh_coord = np.array(
            [
                np.multiply([p.x, p.y], [img_width, img_height]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )
        self.mesh_coord = mesh_coord
        return mesh_coord

    def calc_inputs(self):
        """Points is a 3D List"""

        left_face = self.points[0]
        right_face = self.points[1]
        mesh_coord = self.mesh_coord
        (std_xl, std_yl) = self.std_dev_coord(left_face, mesh_coord)
        (std_xr, std_yr) = self.std_dev_coord(right_face, mesh_coord)

        try:
            left_ratio = std_xl / std_xr
            right_ratio = std_yl / std_yr
            return left_ratio, right_ratio
        except:
            return None

    def pred(self):
        self.update_inputs()
        input = np.array([[self.c1], [self.c2]]).reshape(1, 2, 1)[0:1]
        prediction = np.argmax(self.model.predict(input))
        self.prediction = prediction
        return prediction, self.model_lst[prediction]

    def normalize(self):
        """Left and Right are equivalent"""
        ind = 1 if self.prediction else 0
        return ind

    def get_angle(self) -> float:
        A = self.mesh_coord[152]
        B = self.mesh_coord[10]
        C = (A[0] + 50, A[1])
        AC = self.make_vector(A, C)
        AB = self.make_vector(A, B)
        # angle between 0 and 90°
        angle = (
            np.arcsin(
                self.cross_prod(AB, AC) / ((self.distance(A, C) * self.distance(A, B)))
            )
            * 180
            / pi
        )

        return angle


class HeadPositionModel(S3DWModel):
    def __init__(self, model_path, frame=None, draw=False):
        super().__init__(model_path, frame, draw)
        self.model_lst = ["Front", "Down", "Up"]
        self.points = head_pos_points

    def get_mesh_coords(self, results):
        anchorpoint1 = results.multi_face_landmarks[0].landmark[self.points[0]].z
        anchorpoint2 = results.multi_face_landmarks[0].landmark[self.points[1]].z
        mesh_coord = [anchorpoint1, anchorpoint2]
        self.mesh_coord = mesh_coord
        return mesh_coord

    def calc_inputs(self):
        """Same value as mesh coord"""

        return self.mesh_coord

    def pred(self):
        self.update_inputs()
        input = np.array([[self.c1, self.c2]])
        prediction = self.model.predict(input)[0]
        return prediction, self.model_lst[prediction]

    def normalize(self):
        """Up and down are equivalent"""
        ind = 1 if self.prediction else 0
        return ind
