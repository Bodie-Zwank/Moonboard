import pickle
from grade_conversion import num_to_font
from network_evaluation import predict_nn
from get_data import one_hot
import numpy as np
from flask import Flask, request, jsonify, url_for, render_template
from http.server import BaseHTTPRequestHandler, HTTPServer

def grade_climb():
    network = pickle.load(open('network.pkl' ,'rb'))
    climb = input("Enter comma separated coordinates of climb: ")
    climb = np.reshape(one_hot(climb.split(",")), (1, 198, 1))
    
    print(num_to_font[predict_nn(network, climb[0])])

grade_climb()