#from wine_data import read_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

# set seed
seed = 42

with open('data.pkl', "rb") as f: 
    classes, features = pickle.load(f)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size = 0.5, random_state = seed)


# inialize model 
criterion = "gini"
n_estimators = 20

rf = RandomForestClassifier(n_estimators = n_estimators, random_state= seed)

# fit data
rf.fit(x_train, y_train)

# predict
predictions = rf.predict(x_test)
accuracy = np.mean(predictions == y_test)

print(accuracy)