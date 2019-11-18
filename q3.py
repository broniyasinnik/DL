import torch
import pandas as pd
import numpy as np

train = pd.read_csv(r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/training.csv', header=None)
test = pd.read_csv(r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/test.csv', header=None)

train_img = [np.fromstring(i, dtype=float, sep=' ') for i in train[30].values[1:]]
train_img_scaled = torch.tensor(train_img)/255  # scaling the pictures to [0,1]

train_labels = torch.tensor([i.astype(np.float) for i in train.loc[:, range(30)].values[1:]])
train_labels[torch.isnan(train_labels)] = 0

train_labels = train_labels/96

t_l_mean = train_labels.mean()

train_labels_scaled = train_labels - t_l_mean  # scaling the labels to [-1,1] by dividing by max and subtracting mean



