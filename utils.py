import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torchvision
import torch
import os
from PIL import Image
import cv2
import time
import imutils


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=False)


def dataset_show(loader, classes):
    # get some random  images from either the training or the testing loader
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print('GroundTruth :', ' '.join('%5s' % classes[labels[j]] for j in range(20)))

    return images, labels


# Pour tester le détecteur de visages, on adapte un dataset de visages trouvé sur Kaggle pour en faire le dataset v0 (voir spécifs du v0 dans Face_Detector/main.py)
# Modifications : conversion en B/W, resize
def convert_dataset_to_bw(dataset_dir, destination_dir, img_size):
    for idx, img_name in enumerate(os.listdir(dataset_dir)):
        img = Image.open(dataset_dir + "/" + img_name)
        # Convert to BW
        img = img.convert('L')
        # Resize the imgs (Net won't accept imgs of different sizes)
        img = img.resize(img_size)
        # Save the BW image
        img.save(destination_dir + "/group_" + str(idx) + ".png")


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[0] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pyramid_sliding_window(net, image, scale, winW, winH, stepSize):
    # Store the initial image before resize, it will be used for the final printing
    faces_img = image.copy()
    # loop over the image pyramid
    # all_detected_faces : contains for each pyramid level the scaling factor and the detected faces corresponding to
    # pyramid level
    all_detected_faces = []
    for resized in pyramid(image, scale=scale):
        #print("resized size = ", resized.shape)
        detected_faces = []
        curr_scale_factor = image.shape[0] / resized.shape[0]
        #print(curr_scale_factor)
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # We use the 36*36 window to match the net's img input size
            resized_tensor = torch.from_numpy(window)
            # Transform the 500*500 (2d) img to a 4d tensor (the additional 2 dimensions contain no information)
            resized_tensor = resized_tensor[None, None, :, :]  # tensor shape is now [1,1,500,500]
            # Feed the network the input tensor
            output = net(resized_tensor)

            # We only register faces with a prob higher than 0.99 to avoid false positives
            # (softmax dim parameter : dim=0->rows add up to 1, dim=1->rows add up to 1)
            softmax = torch.nn.functional.softmax(output, dim=1)
            if softmax[0][1] >= 0.999:
                detected_faces.append((x, y))

            '''
            # Old version : using softmax's default threshold = 0.5, LOTS of false positives
            classes = ('noface','face')
            _, predicted = torch.max(output, 1)
            # if output[1] > 0.9 (peut aller jusqu'à 0.99 pour éviter les faux positifs, mais risque de faux négatifs)
            #print("output = ",output)
            #print("torch.max(output,1) = ",torch.max(output, 1))
            if predicted == 1 :
                print('detected new face')
                detected_faces.append((x,y))
            #print('\n')
            '''

            '''
            #Draw the sliding window
            clone = resized.copy() #copy=np funct
            img = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 4)
            cv2.imshow("Window", img)
            cv2.waitKey(1)
            time.sleep(0.025)#0.025
            '''

        #Add the detected faces and the corresponding factors to the all_faces variable
        all_detected_faces.append([curr_scale_factor,detected_faces])

    # We use the non_max_supp algorithm to delete overlaping bounding boxes
    # to avoid detecting the same face multiple times
    for j in range(len(all_detected_faces)):
        for i in range(len(all_detected_faces[j][1])): #all_detected_faces[j][1]->detected faces of the i-pyramid-level
            # in this line we both :
            # - change the tuple from a 2d (startX, startY) to a 4d (startX, startY, endX, endY)
            # - multiply each number of the tuple by the current scale factor
            all_detected_faces[j][1][i] = (
                                              all_detected_faces[j][1][i][0] * all_detected_faces[j][0], all_detected_faces[j][1][i][1] * all_detected_faces[j][0]
                                          ) + (
                                            (all_detected_faces[j][1][i][0] + winW)*all_detected_faces[j][0], (all_detected_faces[j][1][i][1] + winH)*all_detected_faces[j][0]
            )

    # Concatenate detected faces into the same array
    final_detected_faces = []
    for j in range(len(all_detected_faces)):
        final_detected_faces += all_detected_faces[j][1]
    final_detected_faces = non_max_suppression_slow(np.array(final_detected_faces), 0.01)

    # Here the sliding window is done for one pyramid scale
    # We draw the detected faces
    for (startX, startY, endX, endY) in final_detected_faces:
        #Converting the coordinates to int to match cv2.rectangle input type
        startX, startY, endX, endY = int(round(startX)),int(round(startY)), int(round(endX)), int(round(endY))
        faces_img = cv2.rectangle(faces_img, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv2.imshow("Window", faces_img)
    cv2.waitKey(0)


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            #print("overlap = ",overlap)
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]
