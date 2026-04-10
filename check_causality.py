from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

X_train = np.load('processed_data/features_train.npy')
y_train = np.load('processed_data/labels_train.npy')

X_test = np.load('processed_data/features_test.npy')
y_test = np.load('processed_data/labels_test.npy')

_, y_train = torch.topk(torch.tensor(y_train), k=1, dim=2)
y_train = y_train.squeeze(2)
x = X_train[1,:,3]
y = y_train[1]


x = np.diff(x)[1:]
y = np.diff(y)[1:]
df_diff = np.array([x, y]).transpose()

model = VAR(df_diff)
lag_order = model.select_order(maxlags=10)
print(lag_order.summary())
# CONCLUSION TO DRAW FROM THIS:
# All the lag order thingies are very close together, there is really no clear singular lag, which makes sense. So, Granger
# causality cannot really be easily calculated.

tests = grangercausalitytests(np.array([x, y]).transpose(), [6])


