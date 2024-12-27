import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import keras
import copy
import tensorflow as tf
import seaborn as sn
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Flatten, LSTM, Masking
from keras.models import Model
from keras.layers import Input
from sklearn import metrics
from model_helper import *

with open('training_seq_n_12_rmrp0', 'rb') as f:
    training_set = pickle.load(f)

with open('dev_seq_n_12_rmrp0', 'rb') as f:
    dev_set = pickle.load(f)

X_train = training_set['X']
Y_train = training_set['Y']
X_dev = dev_set['X']
Y_dev = dev_set['Y']

np.random.seed(0)
tf.random.set_seed(0)
inputs = Input(shape = (12, 22))
mask = Masking(mask_value = 0.).compute_mask(inputs)
lstm0 = LSTM(20, activation='tanh', input_shape=(12, 22), kernel_initializer='glorot_normal', return_sequences = 'True')(
    inputs, mask = mask)
dense1 = Dense(100, activation='relu', kernel_initializer='glorot_normal')(lstm0)
dense2 = Dense(80, activation='relu', kernel_initializer='glorot_normal')(dense1)
dense3 = Dense(75, activation='relu', kernel_initializer='glorot_normal')(dense2)
dense4 = Dense(50, activation='relu', kernel_initializer='glorot_normal')(dense3)
dense5 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(dense4)
dense6 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(dense5)
flat = Flatten()(dense6)
softmax2 = Dense(10, activation='softmax', name = 'softmax2')(flat)
lstm1 = LSTM(20, activation='tanh', kernel_initializer='glorot_normal', return_sequences = True)(dense6)
lstm2 = LSTM(20, activation='tanh', kernel_initializer='glorot_normal')(lstm1)
dense7 = Dense(15, activation='relu', kernel_initializer='glorot_normal')(lstm2)
dense8 = Dense(15, activation='relu', kernel_initializer='glorot_normal')(dense7)
softmax3 = Dense(10, activation='softmax', name = 'softmax2')(dense8)

def custom_loss(layer):
    def loss(y_true,y_pred):
        loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss2 = K.sparse_categorical_crossentropy(y_true, layer)
        return K.mean(loss1 + loss2, axis=-1)
    return loss

GradeNet = Model(inputs=[inputs], outputs=[softmax3])
GradeNet.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

history_GradeNet_all = []

# Change weights
for i in range(10):
    history_GradeNet = GradeNet.fit(X_train, Y_train, epochs=10, batch_size=256, validation_data = (X_dev, Y_dev), 
                                class_weight = {0:1, 1:1, 2:2, 3: 4, 4: 1, 5: 4, 6: 8, 7: 8, 8: 8, 9: 8})
    history_GradeNet_all.append(history_GradeNet)

F1_train = metrics.f1_score(Y_train, GradeNet.predict(X_train).argmax(axis=1), average = 'macro')
print(F1_train)

GradeNet.save_weights("GradeNet.h5")