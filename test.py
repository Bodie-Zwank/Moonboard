import pickle
import numpy as np
from flask import Flask, request, jsonify, url_for, render_template
from http.server import BaseHTTPRequestHandler, HTTPServer
from preprocessing_helper import *
import os
import heapq
import pandas as pd
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any other level as per the explanation above
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Flatten, LSTM, Masking
from keras.models import Model
from keras.layers import Input
from model_helper import *
from DeepRouteSetHelper import *
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


hyperparameter = [1, 1]
operationList = ["RH", "LH"]

cwd = os.getcwd()
parent_wd = cwd.replace('preprocessing', '')
left_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016LeftHand.csv'
right_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016RightHand.csv'
feature_path = parent_wd + "/raw_data/HoldFeature2016.xlsx"
url_data_path = parent_wd + '/raw_data/moonGen_scrape_2016_cp'

LeftHandfeatures = pd.read_csv(left_hold_feature_path, dtype=str)
RightHandfeatures = pd.read_csv(right_hold_feature_path, dtype=str)
# convert features from pd dataframe to dictionary of left and right hand
RightHandfeature_dict = {}
LeftHandfeature_dict = {}
feature_dict = pd.read_excel(feature_path)
cell_data = feature_dict.iloc[0, 0]
for index in RightHandfeatures.index:
    LeftHandfeature_item = LeftHandfeatures.loc[index]
    LeftHandfeature_dict[(int(LeftHandfeature_item['X_coord']), int(LeftHandfeature_item['Y_coord']))] = np.array(
        list(LeftHandfeature_item['Difficulties'])).astype(int)
    RightHandfeature_item = RightHandfeatures.loc[index]
    RightHandfeature_dict[(int(RightHandfeature_item['X_coord']), int(RightHandfeature_item['Y_coord']))] = np.array(
        list(RightHandfeature_item['Difficulties'])).astype(int)

climb = ["H5", "I6", "F7", "J10", "F13", "H14", "E16", "I18"]

def convert_climb(climb):
    converted = []
    for hold in climb:
        row = int(hold[1:]) - 1
        col = ord(hold[0]) - 65
        converted.append([col, row])
    return converted

def classify_and_reorganize_data(raw_data):
    n_start = 2
    n_end = 1
    n_mid = len(raw_data) - 3
    n_hold = n_start + n_mid + n_end
    x_vectors = np.zeros((10, n_hold))
    for i, (x, y) in enumerate(raw_data):
        x_vectors[0:6, i] = RightHandfeature_dict[(x, y)] # 6 hand features
        x_vectors[6:8, i] = [x, y] #(x, y)
    x_vectors[8:, 0:n_start] = np.array([[1], [0]])
    x_vectors[8:, n_start+n_mid:] = np.array([[0], [1]])
    return x_vectors

def custom_loss(layer):
    def loss(y_true,y_pred):
        loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss2 = K.sparse_categorical_crossentropy(y_true, layer)
        return K.mean(loss1 + loss2, axis=-1)
    return loss

#print(f'x train: {X_train[0]}')
#print('moonboard test: ', moonboardTest['X'][0])

def find_climb_beta(climb):
    climb = convert_climb(climb)

    moonboardTest = classify_and_reorganize_data(climb)
    testbeta = beta(moonboardTest.T)
    status = [beta(moonboardTest.T), beta(moonboardTest.T)]
    status[0].addStartHolds(0)
    status[1].addStartHolds(1)
    totalRun = status[0].totalNumOfHold - 1
    for i in range(totalRun):  # how many new move you wan to add
        status = addNewBeta(status)
        finalScore = [] 
        for i in status:   
            finalScore.append(i.overallSuccessRate())
        largestIndex = heapq.nlargest(4, range(len(finalScore)), key=finalScore.__getitem__)
        if (status[largestIndex[0]].isFinished and status[largestIndex[1]].isFinished) == True:
            break
            
    finalScore = [] 
    for i in status:   
        finalScore.append(i.overallSuccessRate())   
    largestIndex = heapq.nlargest(4, range(len(finalScore)), key=finalScore.__getitem__)

    #print ("After Beamer search, the most possible hand sequence and the successRate:")
    max = 0
    index = 0
    for i in largestIndex:
        if status[i].overallSuccessRate() > max:
            max = status[i].overallSuccessRate()
            index = i
    #return status[index].handOperator
    return status[index]

def grade_climb_with_beta(climb):
    beamerBeta = climb

    numOfMoves = len(beamerBeta.handSequence) 
    routeHandSequence = beamerBeta.handSequence  
    routeOpSequence = beamerBeta.handOperator 
    handStringList = []

    for orderOfHand in range(numOfMoves): 
        targetCoordinate = beamerBeta.getXYFromOrder(routeHandSequence[orderOfHand])
        newHandStr = coordinateToString(targetCoordinate) + "-" + routeOpSequence[orderOfHand]
        handStringList.append(newHandStr)
    print(handStringList)
    all_climb_data = moveGeneratorFromStrList(handStringList, string_mode = False)
    x_vectors = np.zeros((22, len(all_climb_data)))
    for orderOfMove, moveInfoDict in enumerate(all_climb_data):   
        x_vectors[0:2, orderOfMove] = moveInfoDict['TargetHoldString'] 
        x_vectors[2, orderOfMove] = moveInfoDict['TargetHoldHand'] # only express once
        x_vectors[3, orderOfMove] = moveInfoDict['TargetHoldScore']
        x_vectors[4:6, orderOfMove] = moveInfoDict['RemainingHoldString']
        x_vectors[6, orderOfMove] = moveInfoDict['RemainingHoldScore']
        x_vectors[7:9, orderOfMove] = moveInfoDict['dxdyRtoT']
        x_vectors[9:11, orderOfMove] = moveInfoDict['MovingHoldString']
        x_vectors[11, orderOfMove] = moveInfoDict['MovingHoldScore']
        x_vectors[12:14, orderOfMove] = moveInfoDict['dxdyMtoT']
        x_vectors[14:21, orderOfMove] = moveInfoDict['FootPlacement']
        x_vectors[21, orderOfMove] = moveInfoDict['MoveSuccessRate']

    #print(all_climb_data)
    moonboardTest = convert_generated_data_into_test_set(x_vectors)
    #print(moonboardTest)

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
    dense7 = Dense(15, activation='relu', kernel_initializer='glorot_normal', name = 'dense7')(lstm2)
    dense8 = Dense(15, activation='relu', kernel_initializer='glorot_normal', name = 'dense8')(dense7)
    softmax3 = Dense(10, activation='softmax', name = 'softmax2')(dense8)
    
    GradeNet = Model(inputs=[inputs], outputs=[softmax3])
    GradeNet.compile(optimizer='adam', 
                    loss=custom_loss(softmax2),
                    metrics=['sparse_categorical_accuracy'])

    with open("training_seq_n_12_rmrp0", 'rb') as f:
        training_set = pickle.load(f)

    GradeNet.load_weights("GradeNet.h5")
    
    grade = GradeNet.predict(moonboardTest['X']).argmax(axis = 1)
    #print(grade[0])
    return (convert_num_to_V_grade(grade[0]))