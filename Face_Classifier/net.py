import torch.nn as nn
import torch.nn.functional as F

# Conseils d'ordre général sur les CNN
# - Les couches de Conv permettent de trouver les features locales. Les couches FC permettent de classifier les features
# - En général 1 à 3 couches FC. Ça n'ajoute rien de rajouter de la profondeur (i.e nb de couches) dans les couches FC. Rajouter plutôôt
#   de la largeur (nb neurones).
# - Beware : Trop de largeurs (neurones) -> Overfitting

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer : 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Pool Layer, 2x2 window
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer : 6 input image channels, 16 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 3 fully-connected layers - affine operation : y = Wx + b
        # value of linear params ??
        # 16 is defined by the nb of out_channels (i.e nb of filter kernels in the previous conv layer)
        # 5*5 is the spatial size defined by the conv and pooling operations performed on the input data
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        # Output of the last fc layer should be equal to the nb of classes
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max-pooling with the pool layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5) # ?
        # créer un vect 1D à partir d'une matrice 2D en concaténant les lignes
        x = x.view(-1,self.num_flat_features(x))
        # Two ReLu layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Final fc layer
        x = self.fc3(x)
        return x

    #nb total d'éléments
    def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
        num_features*= s
      return num_features
