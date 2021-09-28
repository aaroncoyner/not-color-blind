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

from matplotlib import pyplot as plt


class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


def filter(img, binary, skeleton):
    img = np.array(img)
    img[img == np.amin(img)] = 0
    img[img > 0] = 255
    if skeleton:
        skeleton_img = skeletonize(img[:,:,0], method='lee')
        img[:,:,0] = skeleton_img
        img[:,:,1] = skeleton_img
        img[:,:,2] = skeleton_img
    img = Image.fromarray(img)
    return img


def test(data_dir, image_type, binary=False, skeleton=False, num_classes=2, batch_size=64,
         image_size=(224,224)):

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if device == 'cuda:0':
        torch.cuda.empty_cache()

    if skeleton:
        binary = False
        csv_name = f'./out/probabilities/{image_type}_skeletonized_test.csv'
    elif binary:
        skeleton = False
        csv_name = f'./out/probabilities/{image_type}_binarized_test.csv'
    else:
        csv_name = f'./out/probabilities/{image_type}_test.csv'

    test_transforms = transforms.Compose([transforms.Lambda(lambda img: filter(img, binary, skeleton)),
                                          transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    test_folder = os.path.join(data_dir, image_type, 'test')
    test_dataset = datasets.ImageFolder(test_folder, test_transforms)

    print()
    print(f'Data Directory: {data_dir}')
    print(f'Image Type: {image_type}')
    print(f'Binarize: {binary}')
    print(f'Skeletonize: {skeleton}')
    print(f'Number of Classes: {num_classes}')
    print(f'Batch Size: {batch_size}')
    print(f'Device: {device}')
    print()

    net = NeuralNetClassifier(PretrainedModel,
                    criterion=torch.nn.CrossEntropyLoss,
                    module__output_features=2)
    net.initialize()
    net.load_params(f_params='./out/checkpoints/model_segmentations_filtered_0_binarized.pt')

    img_locs = [loc for loc, _ in test_dataset.samples]
    test_probs = net.predict_proba(test_dataset)
    test_probs = [prob[0] for prob in test_probs]
    data = {'img_loc' : img_locs, 'probability' : test_probs}
    pd.DataFrame(data=data).to_csv(csv_name, index=False)



if __name__ == '__main__':
    data_dir = os.path.join('out', 'datasets')
    test(data_dir, 'segmentations', binary=True)
