##########################################
# @@作者：贾林轩
# @@运行后不断按esc就能看到origin,with_normals的点云效果
###########################################

from pyntcloud import PyntCloud
import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    data = data.T
    print(data.shape)
    N = data.shape[0]
    print("N=",N)
    H = (1/N) * np.dot(data, data.T)   # 协方差矩阵
    print("H:",H,H.shape)
    u, s, v = np.linalg.svd(H)   # 对协方差矩阵做SVD分解，特征向量矩阵即为主成分矩阵
    print("before sort: ","U:", u, u.shape,"S:", s, s.shape)
    eigenvalues = s
    eigenvectors = u
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        print("after sort: ","U:", eigenvectors,"S:", eigenvalues)
    return eigenvalues, eigenvectors


def main():
    # 加载原始点云
    with open('./modelnet40_normal_resampled/'
              'modelnet40_shape_names.txt') as f:
        a = f.readlines()
    for i in a:
        point_cloud_pynt = PyntCloud.from_file(
            './modelnet40_normal_resampled/'
            '{}/{}_0001.txt'.format(i.strip(), i.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

        # 从点云中获取点，只对点进行处理
        points = point_cloud_pynt.points
        points = points.iloc[:,0:3]
        print('total points:', points.shape)

        # 用PCA分析点云主方向
        w, v = PCA(points)
        point_cloud_vector = v[:, 0] # 点云主方向对应的向量
        print('the main orientation of this pointcloud is: ', point_cloud_vector)
        z = v[:,0:2]   # 利用课件中讲的降维原理，将三维投影到二维即降维到二维，取前两个主成分组成z矩阵进行encode
        print(z,z.shape)
        A = np.dot(z.T, points.T)   # encode得到的A（2xN）即为N个点的二维坐标信息
        plt.scatter(A[0,:],A[1,:])
        plt.show()
        print(A.shape)

        # 循环计算每个点的法向量
        # pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        # normals = []
        # 作业2
        # 屏蔽开始
        point_cloud_o3d.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = normals = np.array(point_cloud_o3d.normals)
        # 验证法向量模长为1(模长会有一定的偏差，不完全为1)
        normals_length = np.sum(normals ** 2, axis=1)
        flag = np.equal(np.ones(normals_length.shape, dtype=float), normals_length).all()
        print('all equal 1:', flag)
        # 屏蔽结束
        normals = np.array(normals, dtype=np.float64)
        # TODO: 此处把法向量存放在了normals中
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([point_cloud_o3d],point_show_normal=True)


if __name__ == '__main__':
    main()
