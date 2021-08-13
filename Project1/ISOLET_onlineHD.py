from time import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

import load_choirdat
import onlinehd

def main():
    print('Loading...')
    (x, y), train_features, train_classes = load_choirdat.load_choirdat('./isolet_train.choir_dat')
    (x_test, y_test), test_features, test_classes = load_choirdat.load_choirdat('./isolet_test.choir_dat')

    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(np.array(y)).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(np.array(y_test)).long()

    model = onlinehd.OnlineHD(train_classes, train_features)

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
