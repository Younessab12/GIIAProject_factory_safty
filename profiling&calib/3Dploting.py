import matplotlib.pyplot as plt
import numpy as np
import json

file = open('calib_records/calibration_youness aabibi.json')
data = json.load(file)

yawnin_data = data["2"]
yawning_data_lips = yawnin_data["lips"]
yawning_data_left_eye = yawnin_data["left_eye"]
yawning_data_right_eye = yawnin_data["right_eye"]
talkin_data = data["3"]
talking_data_lips = talkin_data["lips"]
talking_data_left_eye = talkin_data["left_eye"]
talkin_data_right_eye = talkin_data["right_eye"]
orginale_data = data["1"]
orginale_data_lips = orginale_data["lips"]
orginale_data_left_eye = orginale_data["left_eye"]
orginale_data_right_eye = orginale_data["right_eye"]

ax = plt.axes(projection='3d')
ax.plot3D( yawning_data_right_eye, yawning_data_left_eye,yawning_data_lips , 'o')
ax.plot3D(talkin_data_right_eye, talking_data_left_eye,talking_data_lips , 'o')
ax.plot3D( orginale_data_right_eye, orginale_data_left_eye,orginale_data_lips, 'o')
plt.show()