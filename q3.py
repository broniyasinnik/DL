# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:38:21 2019

@authors: Razi, Omer Landau
"""

# -*- coding: utf-8 -*-
"""
Deep Learning course
Ex1_Q3

"""

import os

import numpy as np
from pandas.io.parsers import read_csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pylab
from pylab import *
from sklearn.utils import shuffle
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

FTRAIN = r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/training.csv'
FTEST = r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/test.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


# defining fully connected model for 3.3


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(9216, 100)
        self.fc2 = torch.nn.Linear(100, 30)

    def forward(self, input):
        lin1 = F.relu(self.fc1(input))
        lin2 = self.fc2(lin1)
        return lin2


def train_fc_model(train_loader, test_loader):

    net = Net().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    losses_train = []
    losses_test = []
    epochs =100

    for i in range(epochs):
        test_running_loss, running_loss = 0., 0.
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = net.forward(X_train)
            loss_train = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()
        else:
            with torch.no_grad():
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    y_test_pred = net(X_test)
                    loss_test = criterion(y_test_pred, y_test)
                    test_running_loss += loss_test.item()
            epoch_loss = running_loss / len(train_loader)  # average loss at epoch
            test_epoch_loss = test_running_loss / len(test_loader)
            losses_test.append(test_epoch_loss)
            losses_train.append(epoch_loss)
            print(
                'epoch: {} , training loss: {:.6f} ,  validation loss: {:.6f}'.format((i + 1), epoch_loss, test_epoch_loss))

    loss_final = "Training Loss in epoch {} is: {:.6f} , Test Loss is: {:.6f}".format(epochs, loss_train, loss_test)

    print(loss_final)

    # Plotting loss for training and test
    pylab.plot(range(epochs), losses_train, label='train')
    pylab.plot(range(epochs), losses_test, label='test')

    pylab.legend(loc='upper left')
    pylab.xlabel('epochs')
    pylab.ylabel('Loss')


# defining conv model for 3.4


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 2, 1)
        self.linear1 = nn.Linear(128 * 11 * 11, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 30)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(15488)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

def train_conv_model(train_loader, test_loader):

    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    epochs = 50

    losses_train = []
    losses_test = []

    for i in range(epochs):
        test_running_loss, running_loss = 0., 0.
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model.forward(X_train)
            loss_train = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()
        else:
            with torch.no_grad():
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    y_test_pred = model(X_test)
                    loss_test = criterion(y_test_pred, y_test)
                    test_running_loss += loss_test.item()
            epoch_loss = running_loss / len(train_loader)  # average loss at epoch
            test_epoch_loss = test_running_loss / len(test_loader)
            losses_test.append(test_epoch_loss)
            losses_train.append(epoch_loss)
            print(
                'epoch: {} , training loss: {:.6f} ,  validation loss: {:.6f}'.format((i + 1), epoch_loss, test_epoch_loss))

    loss_final = "Training Loss in epoch {} is: {:.6f} , Test Loss is: {:.6f}".format(epochs, loss_train, loss_test)

    print(loss_final)

    # Plotting loss for training and test
    pylab.plot(range(epochs), losses_train, label='train')
    pylab.plot(range(epochs), losses_test, label='test')

    pylab.legend(loc='upper left')
    pylab.xlabel('epochs')
    pylab.ylabel('Loss')

def main():

    # Data Preprocessing

    X, y = load()

    print(X.shape)

    X_conv = X.reshape(X.shape[0], 1, 96, 96)

    x_tensor = torch.autograd.Variable(torch.from_numpy(X).float())
    x_conv_tensor = torch.autograd.Variable(torch.from_numpy(X_conv).float())
    y_tensor = torch.autograd.Variable(torch.from_numpy(y).float())

    dataset = TensorDataset(x_tensor, y_tensor)
    dataset_conv = TensorDataset(x_conv_tensor, y_tensor)

    # splitting data into training and test/validation

    train_dataset, test_dataset = random_split(dataset, [int(x_tensor.shape[0] * 0.8), int(x_tensor.shape[0] * 0.2)])
    train_conv_dataset, test_conv_dataset = random_split(dataset_conv, [int(x_conv_tensor.shape[0] * 0.8), int(x_tensor.shape[0] * 0.2)])

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    train_conv_loader = DataLoader(dataset=train_conv_dataset, batch_size=1, shuffle=True)
    test_conv_loader = DataLoader(dataset=test_conv_dataset, batch_size=1, shuffle=True)

    train_fc_model(train_loader, test_loader) # q 3.3
    train_conv_model(train_conv_loader, test_conv_loader) # q 3.4


if __name__ == '__main__':
    main()