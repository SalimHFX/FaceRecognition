# Notes S.Duffner :
# Pour résumer sur la détection. Pour pouvoir détecter un ou plusieurs visages sur une (grande) image/photo
# il faut appliquer votre classifieur (CNN) à chaque position (x,y) de l'image et décider s'il s'agit d'un visage ou non.
# A priori, si vous avez utilisé la fonction softmax (transforme la sortie du net en probabilité) comme dans les tutoriaux,
# il suffirait de regarder si la probabilité (sortie du neurone de la classe "visage") est > 0.5. Mais vous allez avoir
# beaucoup trop de fausses détections, donc vous pouvez augmenter ce seuil

# Because you CNN has input size 36x36 you can only detect faces of that size.
# How to detect faces of larger size? Just decrease the size of the input image. You can process several sizes,
# for example by reducing by a factor of 1.2 each time. This is called a (image) pyramid.

# Au début on a une image 500*500 par exemple
# Boucler
#   Parcourir chaque position (x,y) de cette image pour décider s'il y a un visage ou non
#   S'il n'y a pas de visage
#       Réduire l'image d'un facteur de 1.2 et recommencer

from FaceRecognition.Face_Classifier.loader import load_single_dataset
import torchvision.transforms as transforms
from FaceRecognition.Face_Classifier.net import Net
import torch
from FaceRecognition.utils import dataset_show, imshow
import matplotlib.pyplot as plt
from PIL import Image
from FaceRecognition.utils import pyramid_sliding_window
import imutils
import numpy as np

# GLOBAL VARIABLES
dataset_dir = '/home/salim/Coding/Face_Recognition/Datasets/Detector_DATA/v1/BW'
transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0,),std=(1,))])
valid_size = 0.2
# Un batch_size élevé permet de paralléliser si accès à un GPU et d'avoir une MAJ
# plus rapide des poids du net. En général la convergence est plus rapide avec
# des batchs petits mais l'exécution est plus lente (epochs plus longues)
# Valeurs = puissances de 2 en général, pour faciliter le travail GPU
batch_size = 1
classes = ('noface','face')
model_path = '/home/salim/Coding/Face_Recognition/FaceRecognition/Face_Classifier/model_v0.pth'


if __name__ == '__main__':
    '''
        Dataset :
            - v1 : each image is a picture containing several people and is of dimensions higher than 36*36
    '''

    # Load the dataset
    data_loader = load_single_dataset(dataset_dir,batch_size,valid_size,transform)

    # Load saved model
    net = Net()
    net.load_state_dict(torch.load(model_path))

    # Apply the face classifier on each position (x,y) of the image
    dataiter = iter(data_loader)
    image, label = dataiter.next() #img.shape = [1,1,500,500]

    # Loop over each 500*500 image
    # For each image
    #   Cut a 32*32 window and give it to the net as an input
    #   Move the window 1 pixel to the right

    # Original tensor size is [32,1,500,500] (if batch = 32)
    # image.permute(2, 3, 1, 0) gives [500,500,32,1] -> blue/green img
    # ou plus simplement on vire les dimensions qui ne nous intéressent pas
    #image = Image.open('/home/salim/Coding/Face_Recognition/Datasets/Detector_DATA/v1/BW/1/group_5.png')
    image = image[0,0,:,:]
    pyramid_sliding_window(net,np.array(image), 1.2, 36, 36, 5)




    # Decide if the image contains a face or not (softmax ?)

    # Implement some visualization method to color the detected face on the picture

    # Little workaround to stop the matplotlib show() from blocking the console
    #plt.show()
