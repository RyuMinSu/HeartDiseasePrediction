import numpy as np
import pandas as pd

from dataprep.eda import create_report

from heartMLMD import *
from heartDBMD import *

dfpath = ""
user = ""
pw = ""
host = ""
dbName = ""
tbName = ""
rsql = ""

# creatTb(dfpath, user, pw, host, dbName, tbName)
heart = readDb(host, user, pw, dbName, rsql)

# report = create_report(heart)
# report.save("heartdisease.html")

print(f"heart shape: {heart.shape}")
print(heart.info())

heart.columns = heart.columns.str.lower() #컬럼명 변경
heart.rename(columns={"heartdiseaseorattack": "target"}, inplace=True)

sns.countplot(data=heart, x='target')
print("target distribution\n", heart["target"].value_counts() / len(heart) * 100)

cols = list(heart)
cols.remove("bmi")
heart[cols] = heart[cols].astype("int").astype("category")
print(heart.info())

plotcorr(heart, 7, 3, "target", 10, 20, cols)
plotcorr(heart, 1, 1, "target", 10, 8, ["bmi"])

heart = heart.astype("float")
corr = heart.corr()
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask, 1)] = True
plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="Blues", cbar_kws={"shrink": .5})
plt.show();

df = heart.copy()

##결측처리(완료)
##레이블링 및 더미처리(별도 필요 없음)
##파생변수 생성(불필요)
##스케일링
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
scale = ["bmi"]
df[scale] = mms.fit_transform(df[scale])
df[scale].head()

##데이터분리
from sklearn.model_selection import train_test_split

X = df.drop(["target"], axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

##데이터학습 및 평가
lrclf = LogisticRegression(random_state=0)
rfclf = RandomForestClassifier(random_state=0)
xgbclf = xgboost.XGBClassifier()

models = [lrclf, rfclf, xgbclf]
test_size = np.linspace(.1, 1.0, 5)
cv = 3

plotLc(models[0], x_train, x_test, y_train, y_test, test_size, cv)
plotLc(models[1], x_train, x_test, y_train, y_test, test_size, cv)
plotLc(models[2], x_train, x_test, y_train, y_test, test_size, cv)

##tsne
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

X = heart.drop(["target"], axis=1)
y = heart["target"]

skf = StratifiedKFold(n_splits=40, random_state=None, shuffle=False)
for idx, (tridx, teidx) in enumerate(skf.split(X, y)):
    if idx == 0:
        x_train, x_test = X.iloc[tridx], X.iloc[teidx]
        y_train, y_test = y.iloc[tridx], y.iloc[teidx]
    else:
        pass

tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(x_test)
tdf

plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==1))

