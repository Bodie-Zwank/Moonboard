from grade_conversion import font_to_num
import json
import numpy as np

def one_hot(holds):
    vector = [0 for _ in range(18 * 11)]
    for hold in holds:
        row = int(hold[1:]) - 1
        column = ord(hold[0]) - 65
        index = 11 * row + column
        vector[index] = 1
    return vector

def open_file():
    with open("climbs.json", "r") as f:
        raw_text = f.read()
    return json.loads(raw_text)

# organizing data
# all climbs get put in numpy array of size 198x25907
def parse_file(climbs, benchmark):
    climb_data = []
    benchmark_data = []
    grades = []
    benchmark_grades = []
    raw_grades = []
    benchmark_raw_grades = []
    for index, climb in enumerate(climbs):
        climb = climbs[climb]
        holds = [move["Description"] for move in climb["Moves"]]
        vector = one_hot(holds)
        climb_data.append(vector)
        grade = font_to_num[climb["Grade"]]
        raw_grades.append(grade)
        grade_vector = []
        for i in range(16):
            if i == grade:
                grade_vector.append(1)
            else:
                grade_vector.append(0)
        if climb["IsBenchmark"]:
            benchmark_data.append(vector)
            benchmark_grades.append(grade_vector)
            benchmark_raw_grades.append(font_to_num[climb["Grade"]])
        grades.append(grade_vector)
    x_data = np.reshape(climb_data, (len(climb_data), 198, 1))
    y_data = np.reshape(grades, (len(grades), 16, 1))
    x_benchmark_data = np.reshape(benchmark_data, (len(benchmark_data), 198, 1))
    y_benchmark_data = np.reshape(benchmark_grades, (len(benchmark_grades), 16, 1))
    if benchmark:
        return x_benchmark_data, y_benchmark_data, benchmark_raw_grades
    else:
        return x_data, y_data, raw_grades

# changing each climb from one big vector (ex. [0, 0, 1, 0, ... 1, 0]) to an array of vectors (one for each hold)
# this is for RNN (gives hold sequence)
def to_sequence(climb):
    sequence = []
    for index, hold in enumerate(climb):
        if hold == 1:
            hold_vector = [0 for _ in range(198)]
            hold_vector[index] = 1
            sequence.append(hold_vector)
    return np.array(sequence)
        

def train_and_test(data):
    m, n = data.shape
    data_train = data[0:20000].T
    y_train = data_train[0].T
    x_train = data_train[1:n].T
    data_test = data[20000:m].T
    y_test = data_test[0].T
    x_test = data_test[1:n].T
    return x_train, y_train, x_test, y_test