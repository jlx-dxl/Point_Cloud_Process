# 文件功能： 实现 K-Means 算法

import numpy as np
import matplotlib.pyplot as plt
import copy


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def init_centers(self, data):
        # 通过随机选点的方式初始化k个聚类中心
        centers_index = np.random.choice(data.shape[0], size=self.k_, replace=False)
        centers = data[centers_index, :]
        return centers

    def calculate_distances(self, data, centers, D):
        for i in range(data.shape[0]):
            for j in range(self.k_):
                D[i][j] = np.linalg.norm(data[i, :] - centers[j, :])

        return D

    def calculate_belongings(self, data, D):
        B = np.zeros((data.shape[0], self.k_))
        for i in range(data.shape[0]):
            k_index = np.argmin(D[i, :])
            B[i][k_index] = 1
        return B

    def update_centers(self, data, centers, B):
        for i in range(centers.shape[0]):
            data_belongs_to_k = B[:, i].reshape(B.shape[0], 1) * data  # 以B为mask筛选数据矩阵
            # print("data_belongs_to_k:", i, data_belongs_to_k)
            centers[i, :] = (np.sum(data_belongs_to_k, axis=0) / len(
                np.flatnonzero(B[:, i].reshape(B.shape[0], 1))))  # 在列方向上（即每个类中）求筛选后的数据的平均值（和除以非零元素个数）
            # print("centers[i, :]:", i, centers[i, :])
        return centers

    def fit(self, data):
        # 作业1
        # 屏蔽开始

        # 1. 初始化
        self.centers = self.init_centers(data)
        # print("centers:", centers)
        self.D = np.zeros((data.shape[0], self.k_))
        self.B = np.zeros((data.shape[0], self.k_))

        plt.figure(figsize=(10, 8))
        plt.ion()

        for iter in range(self.max_iter_):
            # print(iter)

            # 2. E Step:

            # 维护一个k个聚类中心到n个点的距离矩阵
            self.D = self.calculate_distances(data, self.centers, self.D)
            # print("D:", self.D, self.D.shape)

            # 根据D矩阵维护一个B矩阵，表示n个点和k个类之间belonging关系
            self.B = self.calculate_belongings(data, self.D)
            # print("B:", self.B, self.B.shape)

            # 3. M Step:

            # 根据B矩阵更新centers
            centers_last_one = copy.copy(self.centers)
            # print("centers_last_one", centers_last_one)
            # print("centers:", self.centers)
            # print(np.linalg.norm(centers_last_one - self.centers))

            self.centers = self.update_centers(data, self.centers, self.B)
            # print("centers_last_one", centers_last_one)
            # print("centers:", self.centers)
            # print(np.linalg.norm(centers_last_one - self.centers))

            if np.linalg.norm(centers_last_one - self.centers) < self.tolerance_:
                break

            plt.axis([-10, 15, -5, 15])
            plt.scatter(self.centers[:, 0], self.centers[:, 1], s=50)

            for k in range(self.k_):
                plt.scatter(data[np.flatnonzero(self.B[:, k].reshape(self.B.shape[0], )).tolist(), 0],
                            data[np.flatnonzero(self.B[:, k].reshape(self.B.shape[0], )).tolist(), 1], s=5)
            plt.pause(1)
            plt.savefig('./img/KMeans/KMeans-{}.png'.format(iter + 1))
            plt.clf()
            #
            # 屏蔽结束
        return self

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        result = np.nonzero(self.B)[1]
        # 屏蔽结束
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
    # x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    x = generate_X(true_Mu, true_Var)
    # print(x.shape)

    k_means = K_Means(n_clusters=3)
    k_means.fit(x)
    # print(k_means.centers)
    #
    cat = k_means.predict(x)
    print(cat)
