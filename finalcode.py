import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics, tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import warnings
warnings.filterwarnings('ignore')

PATH = 'D:\comp_studies\Research project\Part 1 data\Crop_recommendation.csv'
df = pd.read_csv(PATH)

# Label encoding the target variable
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])


# Separating features and target label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

# Test Train data split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Function to save model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to load model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Random forest implementation
rf_filename = 'random_forest_model.pkl'
try:
    RF = load_model(rf_filename)
    # print("Random Forest model loaded from file.")
except FileNotFoundError:
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain, Ytrain)
    save_model(RF, rf_filename)
    # print("Random Forest model trained and saved to file.")

predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
print("Random Forest Classifier's Accuracy is: ", x*100,"%")

# Implementing GaussianNB Classifier
nb_filename = 'naive_bayes_model.pkl'
try:
    NaiveBayes = load_model(nb_filename)
    # print("Naive Bayes model loaded from file.")
except FileNotFoundError:
    NaiveBayes = GaussianNB()
    NaiveBayes.fit(Xtrain, Ytrain)
    save_model(NaiveBayes, nb_filename)
    # print("Naive Bayes model trained and saved to file.")

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
print("Naive Bayes's Accuracy is: ", x*100,"%")

# Implementing XGBoost Classifier
xgb_filename = 'xgboost_model.pkl'
try:
    XGB = load_model(xgb_filename)
    # print("XGBoost model loaded from file.")
except FileNotFoundError:
    from xgboost import XGBClassifier
    XGB = XGBClassifier()
    XGB.fit(Xtrain, Ytrain)
    save_model(XGB, xgb_filename)
    # print("XGBoost model trained and saved to file.")

predicted_values = XGB.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
print("XGBoost's Accuracy is: ", x*100," %")

# Predicting the crop
l = []
n = int(input("Enter the value of nitrogen content: "))
p = int(input("Enter the value of phosphorus content: "))
k = int(input("Enter the value of potassium content: "))
temperature = float(input("Enter the value of temperature: "))
humidity = float(input("Enter the value of humidity: "))
ph = float(input("Enter the value of pH: "))
rain = float(input("Enter the value of Rainfall: "))

data = np.array([[n, p, k, temperature, humidity, ph, rain]])

# Adding predictions to set
prediction = XGB.predict(data)
l.append(prediction[0])
print("The crop that should be planted is: ", le.inverse_transform([l[0]])[0])