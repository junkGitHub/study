#coding: Shift_JIS


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from matplotlib.ticker import FormatStrFormatter

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
    from sklearn.lda import LDA
else:
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None);

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline'];


X , y = df_wine.iloc[: , 1:].values , df_wine.iloc[: , 0].values;

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=0);
sc = StandardScaler();


# �W����
X_train_std = sc.fit_transform(X_train);
X_test_std = sc.transform(X_test);

# �����U�s��̌v�Z 
cov_mat = np.cov(X_train_std.T);
# http://mathtrain.jp/varcovmatrix
print np.cov(np.array([[40,80] , [80,90] , [90,100]]).T);