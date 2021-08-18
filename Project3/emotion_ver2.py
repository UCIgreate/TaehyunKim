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


import load_choirdat
import onlinehd

transform_t = transforms.Compose([ transforms.ToTensor(), ])


# loads simple mnist dataset
def load():

    train_dataset = ImageFolder(root='./mod_train', transform=transform_t)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_dataset = ImageFolder(root='./mod_test', transform=transform_t)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128,
                                               shuffle=True)

    pre_inputs = []
    pre_targets = []
    pre_inputs_test = []
    pre_targets_test = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        for i in enumerate(targets):
            image = np.transpose(inputs[i[0]].numpy(), (1,2,0))
            # image_rescale = cv.resize(image, (32, 64))
            image_rescale = image

            fd = hog(image_rescale, orientations=8, pixels_per_cell=(8, 8),
                     cells_per_block=(3, 3), feature_vector=True)

            # Rescale histogram for better display
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            # x, y
            # print(fd.shape)

            # with weight
            if i[1] == 0:
                for k in range(2):
                    pre_inputs.append(fd)
                    pre_targets.append(i[1])
            elif i[1] == 1:
                for k in range(15):
                    pre_inputs.append(fd)
                    pre_targets.append(i[1])
            elif i[1] == 2:
                for k in range(2):
                    pre_inputs.append(fd)
                    pre_targets.append(i[1])
            elif i[1] == 3:
                pre_inputs.append(fd)
                pre_targets.append(i[1])
            elif i[1] == 4:
                pre_inputs.append(fd)
                pre_targets.append(i[1])
            elif i[1] == 5:
                for k in range(2):
                    pre_inputs.append(fd)
                    pre_targets.append(i[1])
            elif i[1] == 6:
                for k in range(2):
                    pre_inputs.append(fd)
                    pre_targets.append(i[1])

            # with no weight
            # pre_inputs.append(fd)
            # pre_targets.append(i[1])

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            # ax1.axis('off')
            # ax1.imshow(image_rescale, cmap=plt.cm.gray)
            # ax1.set_title('Input image')
            # ax2.axis('off')
            # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            # ax2.set_title('Histogram of Oriented Gradients')
            # plt.show()
            # print(hog_image_rescaled.shape)
            # print(i[1])



    for batch_idx, (inputs, targets) in enumerate(test_loader):
        for i in enumerate(targets):
            image = np.transpose(inputs[i[0]].numpy(), (1,2,0))
            # image_rescale = cv.resize(image, (32, 64))
            image_rescale = image

            fd = hog(image_rescale, orientations=8, pixels_per_cell=(8, 8),
                     cells_per_block=(3, 3), feature_vector=True)
            # fd = image

            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            # x_test, y_test
            pre_inputs_test.append(fd)
            pre_targets_test.append(i[1])

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            #
            # ax1.axis('off')
            # ax1.imshow(image_rescale, cmap=plt.cm.gray)
            # ax1.set_title('Input image')
            # ax2.axis('off')
            # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            # ax2.set_title('Histogram of Oriented Gradients')
            # plt.show()


    return pre_inputs, pre_targets, pre_inputs_test, pre_targets_test

# simple OnlineHD training
def main():
    print('Loading...')

    x, y, x_test, y_test = load()

    # x = np.array(x)
    # y = np.array(y)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)
    # x = np.array(x, dtype=np.float32)
    # x_test = np.array(x_test, dtype=np.float32)

    y = np.array(y, dtype=np.int8)
    y_test = np.array(y_test, dtype=np.int8)

    # normalize
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    print(x.shape, y.shape)
    print(x_test.shape, y_test.shape)

    classes = x.size(1)

    model = onlinehd.OnlineHD(7, classes, dim=4000)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=1, lr=0.035, epochs=20)
    t = time() - t

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

if __name__ == '__main__':
    main()