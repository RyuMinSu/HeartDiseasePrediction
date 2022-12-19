import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE


def plotcorr(df, nrow, ncol, target, rsize, csize, colsList=None):
    fig = plt.figure(figsize=(rsize, csize))
    for idx, col in enumerate(colsList):
        plt.subplot(nrow, ncol, idx+1)
        sns.histplot(data=df, x=col, y=target)
    plt.tight_layout()
    plt.show()

def Pprob(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    trainPred = model.predict(x_train)
    testPred = model.predict(x_test)
    return trainPred, testPred

def mlPerform(model, x_train, x_test, y_train, y_test):
    trainPred, testPred = Pprob(model, x_train, x_test, y_train)
    trainRoc = roc_auc_score(y_train, trainPred)
    testRoc = roc_auc_score(y_test, testPred)
    trainClr = classification_report(y_train, trainPred)
    testClr = classification_report(y_test, testPred)
    print(f"{model.__class__.__name__}: train roc score({trainRoc})")
    print(trainClr)
    print(f"{model.__class__.__name__}: test roc score({testRoc})")
    print(testClr)
    return trainPred, testPred, testRoc

def plotRoc(model, x_train, x_test, y_train, y_test):
    trainPred, testPred, testRoc = mlPerform(model, x_train, x_test, y_train, y_test)
    tpr, fpr, thr = roc_curve(y_test, testPred)
    fig = plt.figure(figsize=(10, 10))
    plt.title(f"{model.__class__.__name__} Roc Auc Score")
    plt.plot(tpr, fpr, label=f"roc: {testRoc:.3f}")
    plt.legend()
    plt.show()

def plotLc(model, x_train, x_test, y_train, y_test, test_size, cv):
    plotRoc(model, x_train, x_test, y_train, y_test)
    trainSizes, trainScore, testScore = learning_curve(model, x_train, y_train, train_sizes=test_size, cv=cv)
    trainMean = np.mean(trainScore, axis=1)
    testMean = np.mean(testScore, axis=1)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(trainSizes, trainMean, label="train")
    plt.plot(trainSizes, testMean, label="cross val")
    plt.title(f"{model.__class__.__name__} Learning Curve")
    plt.show()

def tsne(X):
    tsne = TSNE(n_components=2)
    decArr = tsne.fit_transform(X)
    return decArr

def plotTsne(X, y_test, rs, cs):
    decArr = tsne(X)
    fig = plt.figure(figsize=(rs, cs))
    plt.scatter(decArr[:,0], decArr[:,1], c=(y_test==0), label="non heart disease")
    plt.scatter(decArr[:,0], decArr[:,1], c=(y_test==1), label="heart disease")
    plt.legend()
    plt.title("TSNE", size=20)
    plt.show()