import os

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from PIL import Image
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms



class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model
        
    def forward(self, x):
        return self.model(x)


def filter(img, threshold, binary, skeleton):
    img = np.array(img)
    img[img < threshold] = 0
    if binary or skeleton:
        img[img > 0] = 255
        if skeleton:
            skeleton_img = skeletonize(img[:,:,0], method='lee')
            img[:,:,0] = skeleton_img
            img[:,:,1] = skeleton_img
            img[:,:,2] = skeleton_img
    img = Image.fromarray(img)
    return img


def train(data_dir, image_type, threshold=0, binary=False, skeleton=False, num_classes=2,
          batch_size=64, num_epochs=10, lr=0.001, image_size=(224,224), random=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if random:
        binary = False
        skeleton = False
    if skeleton:
        binary = False
    if binary:
        skeleton = False

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Image Type: {image_type}')
    print(f'Threshold: {threshold}')
    print(f'Binarize: {binary}')
    print(f'Skeletonize: {skeleton}')
    print(f'Number of Classes: {num_classes}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Initial Learning Rate: {lr}')
    print(f'Device: {device}')

    train_transforms = transforms.Compose([transforms.Lambda(lambda img: filter(img, threshold,
                                                                         binary, skeleton)),
                                           transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Lambda(lambda img: filter(img, threshold,
                                                                               binary, skeleton)),
                                          transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    if not random:
        train_folder = os.path.join(data_dir, image_type, 'train')
    else:
        train_folder = os.path.join(data_dir, image_type, 'train_random')
    val_folder = os.path.join(data_dir, image_type, 'val')
    test_folder = os.path.join(data_dir, image_type, 'test')
        
    train_dataset = datasets.ImageFolder(train_folder, train_transforms)
    val_dataset = datasets.ImageFolder(val_folder, test_transforms)
    test_dataset = datasets.ImageFolder(test_folder, test_transforms)

    # labels = test_dataset.samples
    # print(labels)
    labels = np.array(train_dataset.samples)[:,1]
    labels = labels.astype(int)
    black_weight = 1 / len(labels[labels == 0])
    white_weight = 1 / len(labels[labels == 1])
    sample_weights = np.array([black_weight, white_weight])
    weights = sample_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    print(f'Number of black eyes: {len(labels[labels == 0])}')
    print(f'Number of white eyes: {len(labels[labels == 1])}')
    print()

    if random:
        pt_name = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_random.pt'
    elif binary:
        pt_name = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_binarized.pt'
    elif skeleton:
        pt_name = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_skeletonized.pt'
    else:
        pt_name = f'./out/checkpoints/model_{image_type}_filtered_{threshold}.pt'

    checkpoint = Checkpoint(f_params=pt_name, monitor='valid_loss_best')

    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)

    net = NeuralNetClassifier(PretrainedModel, 
                              criterion=nn.CrossEntropyLoss,
                              lr=lr,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=num_classes,
                              optimizer=optim.SGD,
                              optimizer__momentum=0.9,
                              # iterator_train__shuffle=True,
                              iterator_train__num_workers=16,
                              iterator_train__sampler=sampler,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=16,
                              train_split=predefined_split(val_dataset),
                              callbacks=[checkpoint, train_acc],
                              device=device)

    net.fit(train_dataset, y=None)

    test_probs = net.predict_proba(test_dataset)

    if random:
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_random.csv'
    elif binary:
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_binarized.csv'
    elif skeleton:
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_skeletonized.csv'
    else:
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}.csv'

    pd.DataFrame(test_probs).to_csv(csv_name)


if __name__ == '__main__':
    if not os.path.isdir(os.path.join('out', 'probabilities')):
        os.makedirs(os.path.join('out', 'probabilities'))
    if not os.path.isdir(os.path.join('out', 'checkpoints')):
        os.makedirs(os.path.join('out', 'checkpoints'))

    data_dir = os.path.join('out', 'datasets')
    train(data_dir, 'retcam')
    train(data_dir, 'retcam', random=True)
    train(data_dir, 'segmentations', random=True)
    thresholds = [0, 50, 100, 150, 200, 210, 220, 230, 240, 250, 256]
    for threshold in thresholds:
        train(data_dir, 'segmentations', threshold=threshold)
        train(data_dir, 'segmentations', binary=True, threshold=threshold)
        train(data_dir, 'segmentations', skeleton=True, threshold=threshold)
    # threshold = 75
    # train(data_dir, 'segmentations', threshold=threshold)
    # train(data_dir, 'segmentations', binary=True, threshold=threshold)
    # train(data_dir, 'segmentations', skeleton=True, threshold=threshold)
