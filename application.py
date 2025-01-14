import pickle
from helpers.small.grade_conversion import num_to_font
from helpers.small.network_evaluation import predict_nn
from helpers.small.get_data import one_hot
import numpy as np
from flask import Flask, request, jsonify, url_for, render_template
from http.server import BaseHTTPRequestHandler, HTTPServer
from grader import *


application = Flask(__name__)
app = application

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/find-beta', methods=['POST'])
def get_beta():
    data = request.json
    coordinates = data['coordinates']
    print(coordinates)
    # Assuming 'coordinates' is a list of coordinate strings and needs further processing
    beta = find_climb_beta(coordinates)
    numOfMoves = len(beta.handSequence) 
    routeHandSequence = beta.handSequence  
    routeOpSequence = beta.handOperator 
    handStringList = []

    for orderOfHand in range(numOfMoves): 
        targetCoordinate = beta.getXYFromOrder(routeHandSequence[orderOfHand])
        newHandStr = coordinateToString(targetCoordinate) + "-" + routeOpSequence[orderOfHand]
        handStringList.append(newHandStr)
    return jsonify({'grade': handStringList})

@app.route('/grade-climb', methods=['POST'])
def grade_climb():
    data = request.json  # Access JSON data sent with POST
    beta = find_climb_beta(data['coordinates'])
    # Assuming 'coordinates' is a list of coordinate strings and needs further processing
    grade = grade_climb_with_beta(beta)
    print(grade)
    return jsonify({'grade': grade})  # Respond with JSON

def calculate_grade(climb):
    network = pickle.load(open('network.pkl' ,'rb'))
    #climb = input("Enter comma separated coordinates of climb: ")
    climb = np.reshape(one_hot(climb), (1, 198, 1))
    return num_to_font[predict_nn(network, climb[0])]
    # evaluating average distance from correct grade (ex. 6C is one grade above 6B+ so it has a "distance" of 1)
    #print(f"Average distance from correct grade: {evaluate_nn(x_temp, raw_grades_temp, trained_network)}")

if __name__ == '__main__':
    app.run(debug=True)