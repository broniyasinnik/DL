import torch
import pandas as pd
import numpy as np

train = pd.read_csv(r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/training.csv')
test = pd.read_csv(r'/Users/omerlandau/Desktop/DL-Wolf/homeworks/EX1/Q3/test.csv')
train_img = [np.fromstring(i, dtype=float, sep=' ') for i in train["Image"].values]
train_img_scaled = torch.tensor(train_img)/255
print(train_img/255)
