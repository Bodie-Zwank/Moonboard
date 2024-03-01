import pickle
from model_helper import *

with open("MediumProblemOfDeepRouteSet_v1", "rb") as f:
    file = pickle.load(f)

print(file['MediumDeepRouteSet_v1_id63'].shape())
test_set = convert_generated_data_into_test_set(file)
print(test_set["X"][0])