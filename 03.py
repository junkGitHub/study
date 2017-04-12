#coding: Shift_JIS
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split
    
print(50 * '=')

iris = datasets.load_iris();
# ���ׂĂ̗v�f��2�C3��ڂ𒊏o
X = iris.data[:,[2,3]];
y = iris.target;

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=0);
sc = StandardScaler();
sc.fit(X_train);

X_train_std = sc.transform(X_train);
X_test_std = sc.transform(X_test);


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples test
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

'''
ppn = Perceptron(n_iter=40 , eta0 = 0.1 , random_state=0);
ppn.fit(X_train_std , y_train);

y_pred = ppn.predict(X_test_std);

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


#print('Section: Learning the weights of the logistic cost function')
# ���W�X�e�B�b�N��A���g�����R�X�g�v�Z

# ���ʂ�1�̎��̃R�X�g�v�Z
def cost_1(z):
    return - np.log(sigmoid(z));

#���ʂ�0�̎��̃R�X�g�v�Z
def cost_0(z):
    return -np.log(1 - sigmoid(z));

'''
# -10 ����10��0.1���݂̃��X�g
z = np.arange(-10 , 10 , 0.1);
phi_z = sigmoid(z);

# ���ʂ�1�̎��̃R�X�g�̃v���b�g
c1 = [cost_1(x) for x in z];
plt.plot(phi_z , c1);

# ���ʂ�0�̎��̃R�X�g�̃v���b�g
c0 = [cost_0(x) for x in z];
plt.plot(phi_z , c0 , linestyle="--");

# �\��
plt.show();
'''

# ���W�X�e�B�b�N��A���g�����w�K
# C�͋t�������W���B�傫���قǍŖސ���̒l���d�������B
# ����������Ɖߊw�K��h�����߂̃E�F�C�g�ւ̃y�i���e�B���傫���Ȃ�B
'''
lr = LogisticRegression(C=1000.0 , random_state=0);
lr.fit(X_train_std , y_train);

plot_decision_regions(X_combined_std , y_combined , classifier=lr , test_idx=range(105 , 150));
plt.show();

#predict_proba���g�����T���v���̏����֌W�̊m���\��
print ("proba ",lr.predict_proba(X_test_std[0,:].reshape(1, -1)));
'''

# C��ω����������̃v���b�g
# C������������ƃE�F�C�g���������Ȃ�
'''
weights , params = [] , [];
for c in np.arange(-5 , 5):
    lr = LogisticRegression(C = 10**c , random_state=0);
    lr.fit(X_train_std , y_train);
    weights.append(lr.coef_[1]);
    params.append(10 ** c);

weights = np.array(weights);
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')

plt.xscale("log");
plt.show();
'''
'''
svm = SVC(kernel="linear" , C=1.0 , random_state=0);
svm.fit(X_train_std , y_train);


plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
'''

# �T�|�[�g�x�N�^�}�V�����g��������`���

#�f�[�^�쐬
np.random.seed(0);
X_xor = np.random.randn(200 , 2);
y_xor = np.logical_xor(X_xor[:,0] > 0 , X_xor[:,1] > 0);
y_xor = np.where(y_xor , 1 , -1);

#plot
'''
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()
'''
'''
svm = SVC(kernel="rbf" , random_state = 0 , gamma=0.1 , C = 10);
svm.fit(X_xor , y_xor);

plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.show()
'''
'''
# iris�f�[�^�Z�b�g��rbf�������ĕ��ނ���
# gamma��傫������ƌ����ɂȂ邪���G�ȋ��E���ɂȂ�
#svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm = SVC(kernel="rbf" , random_state=0 , gamma=100 , C=1);
svm.fit(X_train_std , y_train);
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

'''
