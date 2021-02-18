# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:53:27 2021

@author: laura
"""

from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)

import random
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
#tf.random.set_seed(1234)
#%% Load data
data = loadmat('data4keras_new.mat')
training2 = data['training2']
test = data['test']
del data

#%% Normalize using MinMaxScaler

# 6: LAI, 7: FVC, 8:FAPAR, 9:CWC, 10: CCC
var = 6
feats = np.arange(6)


# Scale training set
ytr = training2[:,var]
mmsy = MinMaxScaler(feature_range=(0,1))
ytrn = mmsy.fit_transform(ytr[:,None])

xtr = training2[:,feats]
mmsx = MinMaxScaler(feature_range=(0,1))
xtrn = mmsx.fit_transform(xtr)

# Scale test set
yts = test[:,var]
ytsn = mmsy.transform(yts[:,None])

xts = test[:,feats]
xtsn = mmsx.transform(xts)

#%% Define NN
neurons = 4
dropout = 0
seed = 1
activation = 'tanh'




#%% Train model
epochs = 1000
batch_size = 64
validation_split = 0.2
T=100
boot = []
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=150)
for t in range (T):
    xi, yi = resample(xtrn, ytrn, replace=True, random_state=123)
    model = Sequential([
            Input((xtrn.shape[1],)),
            Dense(neurons, activation=activation),
            #Dropout(dropout,seed=None),
            Dense(1)
        ])

#model.add(Dropout(dropout, seed= seed))
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss='mse')
    model.fit(xi, yi, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split,shuffle=False,callbacks=[es])
    boot.append(model)

#callbacks=[es]
#%% Show train history
# for k,v in history.history.items():
#     plt.plot(history.epoch, v, label=k)
# plt.grid()
# plt.legend()


#%% Evaluate and predict
erre=[]
errems=[]
for i in range (len(boot)):
    a=boot[i]
    pred = a.predict(xtsn)
    pred_final = mmsy.inverse_transform(pred)
    r = np.sqrt(r2_score(yts, pred_final))
    rmse = np.sqrt(mean_squared_error(yts, pred_final))
    erre.append(r)
    errems.append(rmse)
stop    

plt.plot(erre)
plt.ylim(0.90, 0.905)
plt.plot(errems)
#%% Show results
print('MSE  on test:', mean_squared_error(ytrn_final, preds_final))
print('ME   on test:', np.mean(ytrn_final - preds_final))
print('R2   on test:', r2_score(ytrn_final, preds_final))
print('RMSE on test:', np.sqrt(mean_squared_error(ytrn_final, preds_final)))
print('R    on test:', np.sqrt(r2_score(ytrn_final, preds_final)))
stop