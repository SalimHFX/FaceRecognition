import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torchvision
import os
from PIL import Image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=False)

def dataset_show(loader,classes):
    # get some random  images from either the training or the testing loader
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print('GroundTruth :',' '.join('%5s' % classes[labels[j]] for j in range(20)))

    return images,labels

# Pour tester le détecteur de visages, on adapte un dataset de visages trouvé sur Kaggle pour en faire le dataset v0 (voir spécifs du v0 dans Face_Detector/main.py)
# Modifications : conversion en B/W, resize
def convert_dataset_to_bw(dataset_dir, destination_dir,img_size):
    for idx,img_name in enumerate(os.listdir(dataset_dir)):
        img = Image.open(dataset_dir+"/"+img_name)
        #Convert to BW
        img = img.convert('L')
        #Resize the imgs (Net won't accept imgs of different sizes)
        img = img.resize(img_size)
        #Save the BW image
        img.save(destination_dir+"/group_"+str(idx)+".png")


