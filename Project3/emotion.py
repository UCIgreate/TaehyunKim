from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from skimage.feature import hog
from skimage import data, exposure
import cv2 as cv


import load_choirdat
import onlinehd

transform_train = transforms.Compose([ transforms.ToTensor(), ])


# loads simple mnist dataset
def load():

    train_dataset = ImageFolder(root='./train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_dataset = ImageFolder(root='./test', transform=transform_train)
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
            image_rescale = cv.resize(image, (192, 192), interpolation=cv.INTER_LINEAR)
            # image_rescale = image

            fd, hog_image = hog(image_rescale, orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(4, 4), visualize=True)

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            #
            # ax1.axis('off')
            # ax1.imshow(image_rescale, cmap=plt.cm.gray)
            # ax1.set_title('Input image')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            pre_inputs.append(hog_image_rescaled)
            pre_targets.append(i[1])

            # print(np.shape(pre_train))
            #
            # ax2.axis('off')
            # ax2.imshow(pre_inputs[i[0]], cmap=plt.cm.gray)
            # ax2.set_title('Histogram of Oriented Gradients')
            # plt.show()

            # just for test

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        for i in enumerate(targets):
            image = np.transpose(inputs[i[0]].numpy(), (1,2,0))
            image_rescale = cv.resize(image, (192, 192), interpolation=cv.INTER_LINEAR)

            fd, hog_image = hog(image_rescale, orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(4, 4), visualize=True)

            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            pre_inputs_test.append(hog_image_rescaled)
            pre_targets_test.append(i[1])

    return pre_inputs, pre_targets, pre_inputs_test, pre_targets_test

# simple OnlineHD training
def main():
    print('Loading...')

    x, y, x_test, y_test = load()
    x = np.array(x)
    x_test = np.array(x_test)
    y = np.array(y)
    y_test = np.array(y_test)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    x = x.flatten(1)
    x_test = x_test.flatten(1)

    # 28709
    model = onlinehd.OnlineHD(7, 36864)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=1.0, lr=0.035, epochs=20)
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