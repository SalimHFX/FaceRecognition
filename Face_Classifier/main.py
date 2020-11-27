import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
from loader import load_datasets
from net import Net
from train_network import train_net
from utils import dataset_show
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
train_dir = 'C:/Users/andre/Documents/Flo/data/start_deep/train_images'
test_dir = 'C:/Users/andre/Documents/Flo/data/start_deep/test_images'
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])
valid_size = 0.2
# Un batch_size élevé permet de paralléliser si accès à un GPU et d'avoir une MAJ
# plus rapide des poids du net. En général la convergence est plus rapide avec
# des batchs petits mais l'exécution est plus lente (epochs plus longues)
# Valeurs = puissances de 2 en général, pour faciliter le travail GPU
batch_size = 32
classes = ('noface', 'face')
model_path = 'C:/Users/andre/Documents/Flo/FaceRecognition/Face_Classifier/model_v0.pth'

if __name__ == '__main__':
    # Load the dataset
    # Input images are 36x36 pixels
    # Train dataset : 91 720 imgs (64 770 faces / 26 950 nofaces)
    # Test dataset : 7628 imgs (6831 faces / 797 nofaces)
    train_data, test_data, train_loader, valid_loader, test_loader = load_datasets(train_dir, test_dir, batch_size,
                                                                                   valid_size, transform)
    # Show some of its images
    # dataset_show(train_loader,classes)

    # Define the CNN
    net = Net()
    # Define a Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()  # en python crossEntropyLoss utilise la fonction softmax, par défaut seuil à
    # 0.5 si on utilise le max
    # lr = pas
    # Trop faible -> met des heures/jours à converger
    # Trop grand -> pbs numériques (NaN, infini)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # Switch device to GPU before training the net (doesn't work atm, cuda needs to be enabled)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    # Train the network
    net, train_losses, val_losses = train_net(train_loader, valid_loader, net, device, optimizer, criterion, model_path)
    # load the saved model, that will be the best one obtained in training
    net.load_state_dict(torch.load(model_path))

    # Test the net on the test data
    # images, labels = dataset_show(test_loader, classes)
    # images, labels = images.to(device), labels.to(device)

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(20)))

    # Accuracy on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (len(test_data), 100 * correct / total))

    # Accuracy on the train set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d train images: %d %%' % (len(train_data), 100 * correct / total))

    # Little workaround to stop the matplotlib show() from blocking the console
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.show()
