#coding: Shift_JIS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):

    def __init__(self , eta=0.01 , n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter;
    
    def fit(self , x , y):
        # 100 行列なのでshapeは(100 ,2)のtuplu
        self.w_ = np.zeros(1 + x.shape[1]);
        self.errors_ = [];

        for _ in range(self.n_iter):
            errors = 0.0;

            for xi , target in zip(x , y):
                update = self.eta * (target - self.predict(xi));
                self.w_[1:] += update * xi;
                self.w_[0] += update;
                errors += int(update != 0.0);

            self.errors_.append(errors);

        return self;


    def net_input(self , x):
        return np.dot(x , self.w_[1:]) + self.w_[0];

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0 , 1 , -1);

class AdalineGD(object):

    def __init__(self , eta = 0.01 , n_iter=50):
        self.eta = eta;
        self.n_iter = n_iter;

    def fit(self , X , y):
        
        self.w_ = np.zeros(1 + X.shape[1]);
        self.cost_ = [];

        for i in range(self.n_iter):
            output = self.net_input(X);
            errors = (y - output);
            self.w_[1:] += self.eta * X.T.dot(errors);
            self.w_[0] += self.eta * errors.sum();
            cost = (errors ** 2).sum() / 2.0;
            self.cost_.append(cost);

        return self;


    def net_input(self , X):
        return np.dot(X , self.w_[1:]) + self.w_[0];

    def activation(self , X):
        return self.net_input(X);

    def predict(self , X):
        return np.where(self.activation(X) >= 0.0 , 1 , -1);

# 確率的勾配降下法
class AdalineSGD(object):

    def __init__(self , eta=0.01 , n_iter=10 , shuffle=True , random_state=None):
        self.eta = eta;
        self.n_iter = n_iter;
        self.w_initialized = False;
        self.shuffle = shuffle;

        if random_state:
            np.random.seed(random_state);

    def fit(self , X , y):
        self._initialize_weights(X.shape[1]);
        self.cost_ = [];

        for i in range(self.n_iter):
            if self.shuffle:
                X , y = self._shuffle(X , y);
            cost = [];
            for xi , target in zip(X , y):
                cost.append(self._update_weights(xi , target));

            avg_cost = sum(cost) / len(y);
            self.cost_.append(avg_cost);

        return self;
    '''
    def partial_fit(self , X , y):
        if not self.w_initialized:
            self._initialize_weights(
    '''

    def _shuffle(self , X , y):
        r = np.random.permutation(len(y));
        return X[r] , y[r];

    def _initialize_weights(self , m):
        slef.w_ = np.zeros(1 + m);
        self.w_initialized = True;

    def _update_weights(self , xi , target):
        output = self.net_input(xi);
        error = target - output;
        self.w_[1:] += self.eta * xi.dot(error);
        self.w_[0] += self.eta * error;
        cost = 0.5 * error ** 2;
        return cost;

    def net_input(self , X):
        return np.dot(X , self.w_[1:]) + self.w_[0];

    def activation(self , X):
        return self.net_input(X);

    def predict(self , X):
        return np.where(self.activation(X) >= 0.0 , 1 , -1);


def plot_decision_regions(X , y , classifier , resolution=0.02):

    markers = ('s' , 'x' , 'o' , '^' , 'v');
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]);

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None);

# 100行目までの目的変数の抽出 5列目のベクトル
y = df.iloc[0:100 ,4].values;

# whereはインデックスを返すが引数が与えられていればそれを返す
y = np.where(y == "Iris-setosa" , -1 , 1);

x = df.iloc[0:100 , [0 , 2]].values;


'''
# setosa品質のプロット
plt.scatter(x[:50 , 0] , x[:50 , 1] , color="red" , marker="o" , label="setosa");

# versicolor
plt.scatter(x[50:100 ,0] , x[50:100 , 1] , color ="blue" , marker = "x" , label = "versicolor");

# 軸の名前
plt.xlabel("sepal length [cm]");
plt.ylabel("petal length [cm]");

#凡例
plt.legend(loc = "upper left");

# 表示
plt.show();
'''
'''
ppn = Perceptron(eta=0.1 , n_iter=10);
ppn.fit(x , y);
#plt.plot(range(1 , len(ppn.errors_) + 1) , ppn.errors_ , marker="o");
plot_decision_regions(x , y , classifier=ppn );
plt.show();
'''
'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10 , eta=0.01).fit(x , y);
ax[0].plot(range(1 , len(ada1.cost_) + 1) , np.log10(ada1.cost_),marker="o");

ada2 = AdalineGD(n_iter=10 , eta=0.0001).fit(x , y);
ax[1].plot(range(1 , len(ada2.cost_) + 1) , np.log10(ada2.cost_),marker="o");

plt.show();
'''
X_std = np.copy(x)
X_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
X_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.show();

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

# plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()