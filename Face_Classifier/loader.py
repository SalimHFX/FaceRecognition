import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# train/test datasets -> imgs must be of the SAME Size
def load_datasets(train_dir,test_dir, batch_size, valid_size, transform):
    # In the folder "test_images" you can/should combine "google_faces" and "yale_faces" and rename it to "1"  (faces) and the other one "google_images02_36x36" to "0" (non-faces)
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_data, test_data, train_loader, valid_loader, test_loader


# Load a dataset of imgs of different sizes with torch
def load_single_dataset(dataset_dir,batch_size,valid_size,transform):
    data = torchvision.datasets.ImageFolder(dataset_dir, transform=transform)

    num = len(data)
    indices_train = list(range(num))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num))
    new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

    sampler = SubsetRandomSampler(new_idx)

    loader = torch.utils.data.DataLoader(data,batch_size=batch_size, sampler=sampler, num_workers=1)

    return loader
