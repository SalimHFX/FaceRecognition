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
    # loop over the image pyramid
    for resized in pyramid(image, scale=scale):
        print("resized size = ", resized.shape)
        detected_faces = []
        # cpt = 1
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
            _, predicted = torch.max(output, 1)
            # if output[1] > 0.9 (peut aller jusqu'à 0.99 pour éviter les faux positifs, mais risque de faux négatifs)
            # print("output = ",output)
            # print("torch.max(output,1) = ",torch.max(output, 1))
            classes = ('noface', 'face')
            if predicted == 1:
                # print("predicted = ",classes[predicted], cpt)
                # cpt +=1
                detected_faces.append((x, y))

            '''
            #Draw the sliding window
            clone = resized.copy() #copy=np funct
            img = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 4)
            cv2.imshow("Window", img)
            cv2.waitKey(1)
            time.sleep(0.025)#0.025
            '''

        # Here the sliding window is done for one pyramid scale
        # We draw the detected faces
        faces_img = resized.copy()
        for face in detected_faces:
            faces_img = cv2.rectangle(faces_img, face, (face[0] + winW, face[1] + winH), (255, 0, 0), 4)
            cv2.imshow("Window", faces_img)
        cv2.waitKey(0)
