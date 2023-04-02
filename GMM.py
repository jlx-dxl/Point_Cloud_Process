# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


# plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, tolerance=0.001, max_iter=100):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    # 屏蔽开始
    def init_centers(self, data):
        # 通过随机选点的方式初始化k个聚类中心
        centers_index = np.random.choice(data.shape[0], size=self.k_, replace=False)
        centers = data[centers_index, :]
        return centers

    # 更新后验概率
    def update_post_p(self, data):
        W = np.zeros((data.shape[0], self.pi.shape[1]))
        for k in range(self.k_):
            # print(k)
            W[:, k] = self.pi[:, k] * multivariate_normal.pdf(data, self.mu[k, :], np.diag(self.var[k, :]))
        self.post_p = W / W.sum(axis=1).reshape(-1, 1)

    # 更新pi
    def update_pi(self):
        self.pi = self.N / np.sum(self.N)

    # 更新Mu
    def update_mu(self, data):
        for k in range(self.k_):
            self.mu[k, :] = np.sum((self.post_p[:, k].reshape(-1, 1) * data), axis=0, keepdims=True) / self.N[:, k]

    # 更新Var
    def update_var(self, data):
        for k in range(self.k_):
            self.var[k, :] = np.sum(
                (self.post_p[:, k].reshape(-1, 1) * (data - self.mu[k, :]) * (data - self.mu[k, :])), axis=0,
                keepdims=True) / self.N[:, k]

    def update_B(self, data):
        self.result = np.argmax(self.post_p, axis=1, keepdims=True)
        self.B = np.zeros((data.shape[0], self.k_))
        for i in range(self.result.shape[0]):
            self.B[i, self.result[i]] = 1

    def plot_clusters(self, data, Mu_true, Var_true, iter):
        colors = ['b', 'g', 'r']
        plt.axis([-10, 15, -5, 15])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], s=50, c='k', alpha= 1)
        ax = plt.gca()

        for k in range(self.k_):
            plt.scatter(data[np.flatnonzero(self.B[:, k].reshape(self.B.shape[0], )).tolist(), 0],
                        data[np.flatnonzero(self.B[:, k].reshape(self.B.shape[0], )).tolist(), 1], s=5, c=colors[k], alpha=0.3)

            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[k], 'ls': ':', 'alpha': 1}
            ellipse = Ellipse(self.mu[k], 3 * self.var[k][0], 3 * self.var[k][1], **plot_args)
            ax.add_patch(ellipse)

            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'k', 'alpha': 1}
            ellipse = Ellipse(Mu_true[k], 3 * Var_true[k][0], 3 * Var_true[k][1], **plot_args)
            ax.add_patch(ellipse)

        plt.savefig('./img/GMM/GMM-{}.png'.format(iter + 1))
        plt.pause(0.1)
        plt.clf()

    # 屏蔽结束

    def fit(self, data, Mu_true=None, Var_true = None):
        # 作业3
        # 屏蔽开始

        # 1. 初始化
        self.mu = self.init_centers(data)
        # print("init mu", self.mu, self.mu.shape)
        self.var = np.ones((self.k_, data.shape[1]))
        # print("init var", self.var, self.var.shape)
        self.pi = np.ones((1, self.k_)) / self.k_
        # print("init pi", self.pi, self.pi.shape)

        # plt.figure(figsize=(10, 8))
        # plt.ion()

        for iter in range(self.max_iter_):
            # print(iter)

            # 2. E Step:

            # 更新后验概率
            self.update_post_p(data)
            # print("post_p:", self.post_p, self.post_p.shape)

            self.update_B(data)

            # self.plot_clusters(data, Mu_true, Var_true, iter)

            # 3. M Step

            self.N = np.floor(np.sum(self.post_p, axis=0, keepdims=True))
            N = np.sum(self.N)
            # print("Nk:", self.N, "totally:", N)

            # 更新均值
            mu_last_one = copy.copy(self.mu)
            self.update_mu(data)
            # print("mu:", self.mu, self.mu.shape)
            # print("distance of mu:", np.linalg.norm(mu_last_one - self.mu))

            # 更新方差
            var_last_one = copy.copy(self.var)
            self.update_var(data)
            # print("var:", self.var, self.var.shape)
            # print("distance of var:", np.linalg.norm(var_last_one - self.var))

            # 更新权重
            self.update_pi()
            # print("pi:", self.pi, self.pi.shape)

            if np.linalg.norm(mu_last_one - self.mu) < self.tolerance_ and np.linalg.norm(
                    var_last_one - self.var) < self.tolerance_:
                break

        # 屏蔽结束
        return self

    def predict(self, data):
        # 屏蔽开始
        # 屏蔽结束
        return np.squeeze(self.result.T,axis=0)


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
    # 显示数据
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.savefig('./img/GMM/GT.png')
    # plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X, Mu_true=true_Mu, Var_true=true_Var)
    cat = gmm.predict(X)
    print(cat)
