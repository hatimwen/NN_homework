"""
:Author:
  `Wen Hongtao <github.com/hatimwen>`_

:Organization:
  School of Information and Communication Engineering, Dalian University of Technology, China

:Version: 2021.03.22

Requirements
------------
* `Numpy 1.18 or higher <http://www.numpy.org>`_
"""

import numpy as np


def get_normal(n=3, M=10):
    X = np.ones([n, M])
    for k in range(M):
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        X[:, k] = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return X


def data_label(X):
    M = X.shape[1]
    label = np.ones(M)
    for k in range(M):
        label[k] = 1 if X[0, k] >= 0 else 0
    return label


def step_function(X):
    y = X >= 0
    return np.array(y, dtype=np.int)


def get_if_convergence(y, label):
    return True if all(y == label) else False


def compute_loss(y, lable):
    return np.sum(abs((lable - y)))/y.shape


def perception(X, label, iter, lr):
    M = X.shape[1]
    X = np.vstack((np.ones(M), X)) # X(k) = [1(k), x1(k), x2(k), ..., xn(k)]
    W = np.zeros(X.shape[0])    # W(k) = [w0(k), w1(k), ..., wn(k)]
    # print(W)
    convergence = False
    for i in range(iter):
        if convergence:
            i_con = i
            return i_con, W
        for k in range(M):
            y = step_function(np.matmul(W.T, X))
            convergence = get_if_convergence(y, label)
            W += lr * (label[k] - y[k]) * X[:, k]
        loss = compute_loss(y, label)
        print('iter={0} loss={1}'.format(i, loss))


def test(X, label, W):
    M = X.shape[1]
    X = np.vstack((np.ones(M), X)) # X(k) = [1(k), x1(k), x2(k), ..., xn(k)]
    y = step_function(np.matmul(W.T, X))
    r = np.array(y == label, dtype=np.int).sum()/M
    return r


if __name__ == '__main__':
    for M in [10, 20, 30]:
        # M = 30  # 10, 20, 30
        M_test = 30
        n = 3
        X = np.ones([n, M])
        X = get_normal(n=n, M=M)
        # print(X)
        d = data_label(X)
        # print(d)
        i_con, W = perception(X=X, label=d, iter=100, lr=0.1)
        print('训练收敛迭代次数：{0}'.format(i_con))
        print('训练收敛权值：')
        print(W)
        X_test = np.ones([n, M_test])
        X_test = get_normal(n=n, M=M_test)
        d_test = data_label(X_test)
        r = test(X=X_test, label=d_test, W=W)
        print('测试正确分类率：{:.2%}'.format(r))

