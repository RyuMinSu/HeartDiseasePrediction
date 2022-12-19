#%%
import numpy as np
import pandas as pd
from heartDBMD import *
from heartMLMD import *
from imblearn.over_sampling  import SMOTE
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import warnings; warnings.filterwarnings("ignore")


rsql = "select * from heart"
df = readDb("*", "*", "*", "*", rsql)
print(f"original df shape: {df.shape}")

df.columns = df.columns.str.lower()#컬럼명 변경
df.rename(columns={"heartdiseaseorattack": "target"}, inplace=True)

print("target distribution:\n", df["target"].value_counts() / len(df))#target distribution

X = df.drop(["target"], axis=1)
y = df["target"]


## oversampling
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
resDf = pd.concat([X_res, y_res], axis=1)
print("\nresample df shape", resDf.shape)
print("resampled target distribution:\n,", resDf["target"].value_counts() / len(resDf))

skf = StratifiedKFold(n_splits=40)
for tridx, teidx in skf.split(X, y):
    x_train, x_test = X.iloc[tridx], X.iloc[teidx]
    y_train, y_test = y.iloc[tridx], y.iloc[teidx]

print("\nstratfiedKFold x_test shape", x_test.shape)
print("stratfiedKFold y_test shape", y_test.shape)

plotTsne(x_test, y_test, 10, 10)


##UnderSampling
df = df.sample(frac=1)

desDf = df[df["target"]==1]
nondesDf = df[df["target"]==0][:len(desDf)]

newDf = pd.concat([desDf, nondesDf], axis=0)
print("newDf shape:", newDf.shape)
print("newDf target distribution:\n", newDf["target"].value_counts() / len(newDf))

newX = newDf.drop(["target"], axis=1)
newy = newDf["target"]

plotTsne(newX, newy, 10, 10)


##scaled undersampling
mms = MinMaxScaler()

scale = ["bmi"]
newX[scale] = mms.fit_transform(newX[scale])

plotTsne(newX, newy, 10, 10)
