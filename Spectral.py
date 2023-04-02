import numpy as np
import matplotlib.pyplot as plt
import copy
from KMeans import K_Means


class Spectral(object):
    def __init__(self, n_clusters=2, tolerance=0.001, max_iter=100):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.k_means = K_Means(n_clusters=self.k_)

    def build_adjacency(self, data, var=0.007):
        W = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i, data.shape[0]):  # 全连接
                W[i, j] = np.exp(-(np.linalg.norm(data[i, :] - data[j, :]) / (2 * var)))
                if i == j:
                    W[i][j] = 0
        W = W + W.T
        return W

    def build_degree(self, W):
        D = np.zeros((W.shape))
        for i in range(D.shape[0]):
            D[i][i] = np.sum(W[i, :])
        return D

    def fit(self, data):
        # 1. 建立Adjacency Matrix W
        W = self.build_adjacency(data)
        # print("W:", W, W.shape)

        # 2. 计算Degree Matrix D
        D = self.build_degree(W)
        # print("D:", D, D.shape)

        # 3. 计算Laplacian Matrix L (unnormalized)
        L = D - W

        # 4. 计算L矩阵的特征值和特征向量
        u, s, v = np.linalg.svd(L)
        # print("u:",u,u.shape)
        # print("s:",s,s.shape)
        sort = s.argsort()[::-1]
        s = s[sort]
        u = u[:, sort]
        plt.figure(figsize=(10, 8))
        plt.scatter(np.arange(20), s[(len(s)-20):], s=50)
        plt.show()

        # 5. 取特征值最小的k个特征向量组成矩阵
        U = u[:,(u.shape[1]-self.k_):]
        # U = u[:,0:self.k_]
        # print("U:", U, U.shape)

        # 6. 对U进行KMeans聚类
        self.k_means.fit(U)
        # print("B:",self.k_means.B,self.k_means.B.shape)

        return self

    def predict(self, p_datas):
        result = self.k_means.predict(p_datas)
        return result


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # # 显示数据
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.savefig('./img/KMeans/GT.png')
    # plt.show()
    # plt.close('all')
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    x = generate_X(true_Mu, true_Var)
    # print(x.shape)

    spectral = Spectral(n_clusters=3)
    spectral.fit(x)
    #
    cat = spectral.predict(x)
    # print(cat)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])

    for k in range(3):
        plt.scatter(x[np.flatnonzero(spectral.k_means.B[:, k].reshape(spectral.k_means.B.shape[0], )).tolist(), 0],
                    x[np.flatnonzero(spectral.k_means.B[:, k].reshape(spectral.k_means.B.shape[0], )).tolist(), 1], s=5)

    plt.show()