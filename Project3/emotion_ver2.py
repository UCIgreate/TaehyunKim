from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from skimage.feature import hog
from skimage import data, exposure
import cv2 as cv
import dlib
import face_recognition
import scipy.misc
from sklearn.metrics import f1_score



import load_choirdat
import onlinehd

transform_t = transforms.Compose([ transforms.ToTensor(), ])


# loads simple mnist dataset
def load():

    train_dataset = ImageFolder(root='./mod_train', transform=transform_t)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_dataset = ImageFolder(root='./test', transform=transform_t)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128,
                                               shuffle=True)

    pre_inputs = []
    pre_targets = []
    pre_inputs_test = []
    pre_targets_test = []

    ppc = (8, 8)
    cpb = (3, 3)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        for i in enumerate(targets):
            image = np.transpose(inputs[i[0]].numpy(), (1,2,0))

            hog_v = hog(image, orientations=8, pixels_per_cell=ppc,
                                cells_per_block=cpb, feature_vector=True)

            # if i[1] == 0:
            #     pre_inputs.append(hog_v)
            #     pre_targets.append(i[1])
            # elif i[1] == 1:
            #     pre_inputs.append(hog_v)
            #     pre_targets.append(i[1])
            # elif i[1] == 2:
            #     pre_inputs.append(hog_v)
            #     pre_targets.append(i[1])
            # elif i[1] == 3:
            #     pre_inputs.append(hog_v)
            #     pre_targets.append(i[1])

            pre_inputs.append(hog_v)
            pre_targets.append(i[1])


    for batch_idx, (inputs, targets) in enumerate(test_loader):
        for i in enumerate(targets):
            image = np.transpose(inputs[i[0]].numpy(), (1,2,0))

            hog_v = hog(image, orientations=8, pixels_per_cell=ppc,
                     cells_per_block=cpb, feature_vector=True)

            # if i[1] == 0:
            #     pre_inputs_test.append(hog_v)
            #     pre_targets_test.append(i[1])
            # elif i[1] == 1:
            #     pre_inputs_test.append(hog_v)
            #     pre_targets_test.append(i[1])
            # elif i[1] == 2:
            #     pre_inputs_test.append(hog_v)
            #     pre_targets_test.append(i[1])
            # elif i[1] == 3:
            #     pre_inputs_test.append(hog_v)
            #     pre_targets_test.append(i[1])

            pre_inputs_test.append(hog_v)
            pre_targets_test.append(i[1])

    return pre_inputs, pre_targets, pre_inputs_test, pre_targets_test

# simple OnlineHD training
def main():
    print('Loading...')

    x, y, x_test, y_test = load()

    y = np.array(y)
    y_test = np.array(y_test)

    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    x = torch.from_numpy(x).float()
    x_test = torch.from_numpy(x_test).float()
    y = torch.from_numpy(y).long()
    y_test = torch.from_numpy(y_test).long()

    features = x.size(1)
    classes = y.unique().size(0)
    print(x.shape, x_test.shape)
    print(classes, features)

    model = onlinehd.OnlineHD(classes, features, dim=10000)
    # if torch.cuda.is_available():
    #     x = x.cuda()
    #     y = y.cuda()
    #     x_test = x_test.cuda()
    #     y_test = y_test.cuda()
    #     model = model.to('cuda')
    #     print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=1, lr=0.01, epochs=20)
    t = time() - t

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    f1_acc = f1_score(y, yhat, average='weighted')
    f1_acc_test = f1_score(y_test, yhat_test, average='weighted')
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{f1_acc = :6f}')
    print(f'{f1_acc_test = :6f}')
    print(f'{t = :6f}')

if __name__ == '__main__':
    main()
