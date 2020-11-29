# import the necessary packages
from imutils import object_detection
from FaceRecognition.utils import non_max_suppression_slow
import numpy as np
import cv2

# construct a list containing the images that will be examined
# along with their respective bounding boxes
images = [("/home/salim/Coding/Face_Recognition/Datasets/Detector_DATA/v1_reduced/BW/1/group_6.png", np.array(
    [(415, 80, 450, 115), (285, 105, 320, 140), (160, 120, 195, 155), (285, 130, 320, 165), (165, 135, 200, 170),
     (170, 135, 205, 170),(165, 140, 200, 175), (170, 140, 205, 175), (105, 175, 140, 210), (110, 175, 145, 210),
     (385, 205, 420, 240), (390, 235, 425, 270), (385, 270, 420, 305), (380, 310, 415, 345)]))]

# loop over the images
for (imagePath, boundingBoxes) in images:
    # load the image and clone it
    print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    orig = image.copy()
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # perform non-maximum suppression on the bounding boxes
    #pick = object_detection.non_max_suppression(boundingBoxes, 0.3)
    pick = non_max_suppression_slow(boundingBoxes, 0.01)

    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # display the images
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
