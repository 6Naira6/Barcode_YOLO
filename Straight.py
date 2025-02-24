import cv2
import numpy as np

# Load the image
image = cv2.imread('img2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

angle = lines[0][0][1] * 180 / np.pi

rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1), (image.shape[1], image.shape[0]))

cv2.imwrite('straightened_image.jpg', rotated)
