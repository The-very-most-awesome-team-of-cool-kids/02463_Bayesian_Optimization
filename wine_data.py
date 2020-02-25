# Wine dataset

import pandas as pd 
import pickle

col_Names=["Cultivar", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
"Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Colour Intensity", "Hue", "OD280/OD315 of Diluted Wines",
"Proline"]

def read_data():
    data = pd.read_csv("wine.data",names=col_Names)
    return data

data = read_data()

classes = data.iloc[:, 0]
features = data.iloc[:, 1:]


with open('data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([classes, features], f)