import os

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from PIL import Image
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping
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


def filter(img, threshold, binary, skeleton, upper, upper_threshold=150):
    img = np.array(img)
    img[img < threshold] = 0
    if upper:
        img[img > upper_threshold] = 0
    if binary or skeleton:
        img[img > 0] = 255
        if skeleton:
            skeleton_img = skeletonize(img[:,:,0], method='lee')
            img[:,:,0] = skeleton_img
            img[:,:,1] = skeleton_img
            img[:,:,2] = skeleton_img
    img = Image.fromarray(img)
    return img


def train(data_dir, image_type, threshold=0, upper=False, upper_threshold=150, binary=False,
          skeleton=False, num_classes=2, batch_size=64, num_epochs=10, lr=0.001, random=False,
          image_size=(224,224)):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if device == 'cuda:0':
        torch.cuda.empty_cache()
    if random:
        binary = False
        skeleton = False
        f_params = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_random.pt'
        f_history = f'./out/histories/model_{image_type}_filtered_{threshold}_random.json'
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_random.csv'
    elif skeleton:
        binary = False
        f_params = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_skeletonized.pt'
        f_history = f'./out/histories/model_{image_type}_filtered_{threshold}_skeletonized.json'
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_skeletonized.csv'
    elif binary:
        skeleton = False
        f_params = f'./out/checkpoints/model_{image_type}_filtered_{threshold}_binarized.pt'
        f_history = f'./out/histories/model_{image_type}_filtered_{threshold}_binarized.json'
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}_binarized.csv'
    else:
        f_params = f'./out/checkpoints/model_{image_type}_filtered_{threshold}.pt'
        f_history = f'./out/histories/model_{image_type}_filtered_{threshold}.json'
        csv_name = f'./out/probabilities/{image_type}_filtered_{threshold}.csv'

    train_transforms = transforms.Compose([transforms.Lambda(lambda img: filter(img, threshold,
                                                                                binary, skeleton,
                                                                                upper,
                                                                                upper_threshold)),
                                           transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(25),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Lambda(lambda img: filter(img, threshold,
                                                                               binary, skeleton,
                                                                               upper,
                                                                               upper_threshold)),
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

    labels = np.array(train_dataset.samples)[:,1]
    labels = labels.astype(int)
    black_weight = 1 / len(labels[labels == 0])
    white_weight = 1 / len(labels[labels == 1])
    sample_weights = np.array([black_weight, white_weight])
    weights = sample_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Image Type: {image_type}')
    print(f'Threshold: {threshold}')
    print(f'Binarize: {binary}')
    print(f'Skeletonize: {skeleton}')
    print(f'Number of Classes: {num_classes}')
    print(f'Number of black eyes: {len(labels[labels == 0])}')
    print(f'Number of white eyes: {len(labels[labels == 1])}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Initial Learning Rate: {lr}')
    print(f'Device: {device}')
    print()

    checkpoint = Checkpoint(monitor='valid_loss_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)

    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)

    early_stopping = EarlyStopping()

    callbacks = [checkpoint, train_acc, early_stopping]

    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss,
                              lr=lr,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=num_classes,
                              optimizer=optim.SGD,
                              optimizer__momentum=0.9,
                              iterator_train__num_workers=16,
                              iterator_train__sampler=sampler,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=16,
                              train_split=predefined_split(val_dataset),
                              callbacks=callbacks,
                              device=device)

    net.fit(train_dataset, y=None)

    img_locs = [loc for loc, _ in test_dataset.samples]
    test_probs = net.predict_proba(test_dataset)
    test_probs = [prob[0] for prob in test_probs]
    data = {'img_loc' : img_locs, 'probability' : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)



if __name__ == '__main__':
    if not os.path.isdir(os.path.join('out', 'probabilities')):
        os.makedirs(os.path.join('out', 'probabilities'))
    if not os.path.isdir(os.path.join('out', 'checkpoints')):
        os.makedirs(os.path.join('out', 'checkpoints'))
    if not os.path.isdir(os.path.join('out', 'histories')):
        os.makedirs(os.path.join('out', 'histories'))

    data_dir = os.path.join('out', 'datasets')

    # threshold = 75
    # train(data_dir, 'segmentations', threshold=threshold, upper=True)
    # train(data_dir, 'segmentations', binary=True, threshold=threshold, upper=True)
    # train(data_dir, 'segmentations', skeleton=True, threshold=threshold, upper=True)
    #
    # threshold = 0
    # upper_threshold = 10
    # train(data_dir, 'segmentations', threshold=threshold,
    #       upper=True, upper_threshold=upper_threshold)
    # train(data_dir, 'segmentations', binary=True, threshold=threshold,
    #       upper=True, upper_threshold=upper_threshold)
    # train(data_dir, 'segmentations', skeleton=True, threshold=threshold,
    #       upper=True, upper_threshold=upper_threshold)

    # train(data_dir, 'retcam')
    # train(data_dir, 'retcam', random=True)
    train(data_dir, 'segmentations', random=True)
    #
    # thresholds = [0, 50, 100, 150, 200, 210, 220, 230, 240, 250, 257]
    # for threshold in thresholds:
    #     train(data_dir, 'segmentations', threshold=threshold)
    #     train(data_dir, 'segmentations', binary=True, threshold=threshold)
    #     train(data_dir, 'segmentations', skeleton=True, threshold=threshold)
