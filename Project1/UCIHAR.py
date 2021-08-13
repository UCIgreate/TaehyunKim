from time import time

import pandas as pd
import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import onlinehd

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# loads simple mnist dataset
def load():

    # fetches data
    with open('./X_train.txt', 'r') as text_file:
        x = text_file.readlines()
    x =' '.join(x).split()
    x = list_chunk(x, 561)
    x = np.array(x)
    x = x.astype(float)

    with open('./X_test.txt', 'r') as text_file:
        x_test = text_file.readlines()
    x_test =' '.join(x_test).split()
    x_test = list_chunk(x_test, 561)
    x_test = np.array(x_test)
    x_test = x_test.astype(float)

    with open('./y_train.txt', 'r') as text_file:
        y = text_file.readlines()
    y =' '.join(y).split()
    y = np.array(y, dtype =np.int8)

    with open('./y_test.txt', 'r') as text_file:
        y_test = text_file.readlines()
    y_test =' '.join(y_test).split()
    y_test = np.array(y_test, dtype =np.int8)

    # split and normalize
    # x, x_test, y, y_test = sklearn.model_selection.train_test_split(x, y)
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    return x, x_test, y-1, y_test-1

# simple OnlineHD training
def main():
    print('Loading...')
    x, x_test, y, y_test = load()

    dimension_hypers = [5000, 7500, 10000]
    bootstraps_hypers = [0.25, 0.5]
    learning_rate_hypers = [0.2, 0.3, 0.4, 0.5]
    epochs_hypers = [20, 40, 60]

    result_ = []

    for dims in dimension_hypers:
        for bs in bootstraps_hypers:
            for lrs in learning_rate_hypers:
                for epoch in epochs_hypers:



                    classes = y.unique().size(0)
                    features = x.size(1)
                    model = onlinehd.OnlineHD(classes, features, dim=dims)

                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()
                        x_test = x_test.cuda()
                        y_test = y_test.cuda()
                        model = model.to('cuda')
                        print('Using GPU!')

                    print('Training...')
                    t = time()
                    model = model.fit(x, y, bootstrap=bs, lr=lrs, epochs=epoch)
                    t = time() - t

                    print('Validating...')
                    yhat = model(x)
                    yhat_test = model(x_test)
                    acc = (y == yhat).float().mean()
                    acc_test = (y_test == yhat_test).float().mean()
                    acc = acc.float()
                    print(f'{acc = :6f}')
                    print(f'{acc_test = :6f}')
                    print(f'{t = :6f}')
                    print()
                    result_.append([dims, bs, lrs, epoch, t, acc.item(), acc_test.item()])

    result = pd.DataFrame(result_)
    result.columns = ['Dimension', 'Bootstraps', 'Learning Rate', 'Epochs', 'Time', 'Acc', 'Acc Test']
    result.to_csv('result.csv')

if __name__ == '__main__':
    main()
