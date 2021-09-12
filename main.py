import csv
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
factors = df[["age", "gender", "cp", "chol", "thalach"]]
heartattack = df["target"]

factorstrain, factorstest, heartattacktrain, heartattacktest = train_test_split(factors, heartattack, test_size = 0.25, random_state = 0)
sc_x = StandardScaler()
factorstrain = sc_x.fit_transform(factorstrain)
factorstest = sc_x.transform(factorstest)

classifier = LogisticRegression(random_state = 0)
classifier.fit(factorstrain, heartattacktrain)

LogisticRegression(C = 1.0, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = "auto", n_jobs = None, penalty = "l2", random_state = 0, solver = "lbfgs", tol = 0.0001, verbose = 0, warm_start = False)
heartattackprediction = classifier.predict(factorstest)
predictedvalues = []
actualvalues = []
labels = ["Yes", "No"]

for i in heartattackprediction:
    if i == 0:
        predictedvalues.append("no")
    else:
        predictedvalues.append("yes")

for i in heartattacktest.ravel():
    if i == 0:
        actualvalues.append("no")
    else:
        actualvalues.append("yes")

cm = confusion_matrix(actualvalues, predictedvalues, labels)
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)