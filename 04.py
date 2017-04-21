#coding: Shift_JIS

import pandas as pd;
import numpy as np;

from io import StringIO;
from sklearn.preprocessing import Imputer;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import OneHotEncoder;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.preprocessing import StandardScaler;
from sklearn.linear_model import LogisticRegression;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.base import clone;
from sklearn.metrics import accuracy_score;
from itertools import combinations;
import matplotlib.pyplot as plt;


from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version;
if Version(sklearn_version) < "0.18":
    from sklearn.grid_search import train_test_split;
else:
    from sklearn.model_selection import train_test_split


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,''';
'''
csv_data = unicode(csv_data);
df = pd.read_csv(StringIO(csv_data));


print df;

print "\n";
print 50 * "=";
print "\n";
# 列ごとにNANの数を数える
#print df.isnull().sum();

# NANのある行を削除
# print df.dropna();

# NAN のある列を削除
#print df.dropna(axis=1);

# NANのある行がthresh未満を削除
# print df.dropna(thresh=4);

# 全ての列がNANである行だけを削除
# print df.dropna(how="all");

# 特定の列にNANが含まれている行だけを削除
# print df.dropna(subset=["C"]);

# axis=0だと列平均　1にすると行平均になる

imr = Imputer(missing_values = "NaN" , strategy="mean" , axis=0);
imr = imr.fit(df);
imputed_data = imr.transform(df.values);
print imputed_data;
'''

'''
df = pd.DataFrame([["green" , "M" , 10.1 , "class1"],
                   ["red" , "L" , 13.5 , "class2"] ,
                   ["blue" , "XL" , 15.3 , "class1"]]);

# 列名を設定
df.columns = ["color" , "size" , "price" , "classlabel"];

print df;
print "\n";
print 50 * "=";
print "\n";

size_mapping = { "XL" : 3 , "L" : 2 , "M" : 1 };
df["size"] = df["size"].map(size_mapping);

class_mapping = {label : idx for idx , label in enumerate(np.unique(df["classlabel"]))};
df["classlabel"] = df["classlabel"].map(class_mapping);

#print df;

# colorにラベル付け
X = df[["color" , "size" , "price"]].values;
color_le = LabelEncoder();
X[:,0] = color_le.fit_transform(X[:,0]);

# OnehotEncoderを使う事で、数字を割り当てることでおきる大小関係を学習の中から除外することができる

ohe = OneHotEncoder(categorical_features=[0]);
X_onehot = ohe.fit_transform(X).toarray();
print X_onehot;
'''

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data" , header=None);
df_wine.columns = ["Class label" , "Alcohol" , "Malloc acid" , "Ash" , "Alcalinity of ash" , "Magnesium" , "Total phenois"
                   , "Flavanoids" , "Nonflavanoid phenols" , "Proanthocyanins" , "Color intensity" , "Hue" , "OD280/OD315 of diluted wines" , "Proline"];

print df_wine.head();
print 50 * "=";

# 特徴とクラスラベルを別々に抽出
X , y = df_wine.iloc[: , 1:].values , df_wine.iloc[:,0].values;

# トレーニングデータとテストデータを分割。全体の30%をテストデータ
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=0);

# 特徴量を正規化する（０～１）にそろえる
mms = MinMaxScaler();
X_train_norm = mms.fit_transform(X_train);
X_test_norm = mms.transform(X_test);

# 標準化　分散１平均0
stdsc = StandardScaler();
X_train_std = stdsc.fit_transform(X_train);
X_test_std = stdsc.transform(X_test);
'''
#L1正則化ロジスティック回帰
# Cが小さいほど正則化の効果が高くなる
lr = LogisticRegression(penalty="l1" , C=0.1);
lr.fit(X_train_std , y_train);

print ("Training accuracy " , lr.score(X_train_std , y_train));
print ("Test accuracy " , lr.score(X_test_std , y_test));



fig = plt.figure();
ax = plt.subplot(111);

colors = ["blue" , "green" , "red" , "cyan" , "magenta" , "yellow" , "black" , 
          "pink" , "lightgreen" , "lightblue" , "gray" , "indigo" , "orange" ];

weights , params = [] , [];
for c in np.arange(-4 , 6):
    lr = LogisticRegression(penalty="l1" , C=10**c , random_state=0);
    lr.fit(X_train_std , y_train);
    weights.append(lr.coef_[1]);
    params.append(10**c);


weights = np.array(weights);

for column , color in zip(range(weights.shape[1]) , colors):
    plt.plot(params , weights[: , column] , 
             label = df_wine.columns[column + 1] ,
             color = color);

plt.xscale("log");
plt.show();

'''


# sequential backward selection
class SBS(object):
    
    def __init__(self , estimator , k_features , scoring=accuracy_score , test_size=0.25 , random_state=1):
        self.scoring = scoring;
        self.estimator = estimator;
        self.k_features = k_features;
        self.test_size = test_size;
        self.random_state = random_state;

    def fit(self , X , y):

        X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = self.test_size ,
                                                               random_state = self.random_state);

        # m行n列なのでnが返ってくる。つまり特徴量の次元
        dim = X_train.shape[1];
        self.indices_ = tuple(range(dim));
        self.subsets_ = [self.indices_];

        score = self._calc_score(X_train , y_train , X_test , y_test , self.indices_);
        self.scores_ = [score];

        while dim > self.k_features:
            scores = [];
            subsets = [];

            # indices_からrの組み合わせのいてれーた
            for p in combinations(self.indices_ , r=dim-1):
                score = self._calc_score(X_train , y_train , X_test , y_test , p);
                scores.append(score);
                subsets.append(p);

            best = np.argmax(scores);
            self.indices_ = subsets[best];
            self.subsets_.append(self.indices_);
            dim -= 1;

            self.scores_.append(scores[best]);

        self.k_score_ = self.scores_[-1];

    def transform(self , X):
        return X[: , self.indices_];

    def _calc_score(self , X_train , y_train , X_test , y_test , indices):
        self.estimator.fit(X_train[: , indices] , y_train);
        y_pred = self.estimator.predict(X_test[: , indices]);
        score = self.scoring(y_test , y_pred);
        return score;

'''
knn = KNeighborsClassifier(n_neighbors=2);

sbs = SBS(knn , k_features=1);
sbs.fit(X_train_std , y_train);
#k_feat = [len(k) for k in sbs.subsets_];
#plt.plot(k_feat , sbs.scores_ , marker="o");
#plt.show();

knn.fit(X_train_std , y_train);
print ("Train accuracy : " , knn.score(X_train_std , y_train));
# トレーニング ＞テストの正解率なので少し過学習
print ("Test accuracy : " , knn.score(X_test_std , y_test));

# 一番性能が良くて一番次元が少ないところを抽出
k5 = list(sbs.subsets_[8]);
knn.fit(X_train_std[: , k5] , y_train);

print ("Train accuracy : " , knn.score(X_train_std[:,k5] , y_train));
# トレーニング ＞テストの正解率なので少し過学習
print ("Test accuracy : " , knn.score(X_test_std[:,k5] , y_test));
'''

# ランダムフォレストで特徴量の重要度を計算する
'''
feat_labels = df_wine.columns[1:];
forest = RandomForestClassifier(n_estimators=100 , random_state=1 , n_jobs=-1);
forest.fit(X_train , y_train);
importances = forest.feature_importances_;
indices = np.argsort(importances)[::-1];
'''
