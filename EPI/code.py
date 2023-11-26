import cv2
import numpy as np

def skin_color_detection(image):
  ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

  # Define the range for skin color
  # These values can vary; you might need to adjust them
  min_YCrCb = np.array([0, 133, 77], np.uint8)
  max_YCrCb = np.array([255, 173, 127], np.uint8)

  # Find skin region
  skinRegion = cv2.inRange(ycrcb_image, min_YCrCb, max_YCrCb)

  # Do a bit-wise and with the original image
  skin = cv2.bitwise_and(image, image, mask = skinRegion)
  return skin