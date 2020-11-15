# Notes S.Duffner :
# Pour résumer sur la détection. Pour pouvoir détecter un ou plusieurs visages sur une (grande) image/photo
# il faut appliquer votre classifieur (CNN) à chaque position (x,y) de l'image et décider s'il s'agit d'un visage ou non.
# A priori, si vous avez utilisé la fonction softmax comme dans les tutoriaux, il suffirait de regarder si la probabilité
# (sortie du neurone de la classe "visage") est > 0.5. Mais vous allez avoir beaucoup trop de fausses détections, donc
# vous pouvez augmenter ce seuil

# Because you CNN has input size 36x36 you can only detect faces of that size.
# How to detect faces of larger size? Just decrease the size of the input image. You can process several sizes,
# for example by reducing by a factor of 1.2 each time. This is called a (image) pyramid.

from Face_Classifier.loader import load_single_dataset
import torchvision.transforms as transforms
from Face_Classifier.net import Net
import torch
from utils import dataset_show
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
dataset_dir = '/home/salim/PycharmProjects/Face_Recognition/Detector_DATA/v1/BW'
transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0,),std=(1,))])
valid_size = 0.2
# Un batch_size élevé permet de paralléliser si accès à un GPU et d'avoir une MAJ
# plus rapide des poids du net. En général la convergence est plus rapide avec
# des batchs petits mais l'exécution est plus lente (epochs plus longues)
# Valeurs = puissances de 2 en général, pour faciliter le travail GPU
batch_size = 32
classes = ('noface','face')
model_path = '/home/salim/PycharmProjects/Face_Recognition/Face_Classifier/model_v0.pth'


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
    images, labels = dataset_show(data_loader,classes)

    outputs = net(images) #ça plante ici -> size mismatch
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(20)))

    # Decide if the image contains a face or not (softmax ?)

    # Implement some visualization method to color the detected face on the picture

    # Little workaround to stop the matplotlib show() from blocking the console
    plt.show()
