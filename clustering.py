# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from itertools import cycle
import open3d as o3d
import random
from pandas import DataFrame
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     ground: 地面点云
#     segmengted_cloud: 删除地面点之后的点云

def ground_segmentation_using_ransac(data, tau=0.6, e=0.7, s=3, p=0.99):
    # 作业1
    # 屏蔽开始
    iters = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
    # print(iters)
    for i in range(int(iters)):
        # 1. 随机sample

        sample_index = random.sample(range(data.shape[0]), 3)
        selected_points = data[sample_index, :]
        # print("selected_points:",selected_points,selected_points.shape)

        # 2. 求解模型（平面以其法向量表示）

        vector1_2 = (selected_points[0, :] - selected_points[1, :])
        vector1_3 = (selected_points[0, :] - selected_points[2, :])
        N = np.cross(vector1_2, vector1_3)  # 向量叉乘求解平面法向量
        # print("N:", N)

        # 3. 计算每个点到该平面的距离
        vectors = data - selected_points[0, :]  # 将所有数据点和取三个点中随机的一个相连，形成向量
        distance = abs(vectors.dot(N)) / np.linalg.norm(N)  # 求距离

        # 4. 根据距离阈值tau区分内点和外点
        idx_ground = (distance <= tau)
        num_inlier = np.sum(idx_ground == True)

        if (num_inlier / data.shape[0]) > (1 - e):
            break

    # print("iters = %f" % i)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[np.logical_not(idx_ground)]

    # 屏蔽结束

    print('ground data points num:', ground_cloud.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return ground_cloud, segmented_cloud


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering_using_dbscan(data, raidus=0.4, min_num=10):
    # 作业2
    # 屏蔽开始
    # print("data:", data.shape)
    DBSCAN_cluster = DBSCAN(eps=raidus, min_samples=min_num).fit(data)
    clusters_index = DBSCAN_cluster.labels_
    # 屏蔽结束
    return clusters_index


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    sort = cluster_index.argsort()
    cluster_index_sorted = cluster_index[sort]
    # print("cluster_index_sorted:", cluster_index_sorted)

    value = -1
    cache = []  # 定义一个cache用于暂存h_index相同的这些点的坐标
    clustered_points_index = []
    clustered_points_o3d = []

    for i in sort:
        # print("i:",i)

        # 如果目前点的index和value（先前点的index）相同，将其装入cache
        if value == cluster_index[i]:
            cache.append(i)

        # 如果目前点的index和value（先前点的index）不同，说明当前voxel中的points都遍历完了，可以根据策略的不同选取其中一个，并释放cache
        elif value != cluster_index[i]:
            value = cluster_index[i]  # 记录这个point的index即为下一个voxel的index
            cachenp = np.asarray(cache)
            # print("cache:",cachenp,cachenp.shape)
            clustered_points_index.append(cachenp)
            cache = []

    # clustered_points_index = np.asarray(clustered_points_index)
    # print("clustered_points_index:",clustered_points_index)

    color = cycle([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]])

    for idx, c in zip(clustered_points_index, color):
        points = data[idx, :]
        points_o3d = point_cloud_to_instance(points)
        if np.mean(cluster_index[idx]) == -1:
            points_o3d.paint_uniform_color([0.5, 0.5, 0.5])
        else:
            points_o3d.paint_uniform_color(c)
        clustered_points_o3d.append(points_o3d)

    print("total cluster numbers:", max(cluster_index[idx]) + 1)
    return clustered_points_o3d


def point_cloud_to_instance(data):
    origin_points_df = DataFrame(data, columns=['x', 'y', 'z'])
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    return point_cloud_o3d


def main():
    root_dir = './data'  # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[0:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)  # 读取数据点
        # print("origin_points:", origin_points,origin_points.shape)
        point_cloud_o3d = point_cloud_to_instance(origin_points)
        # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

        # 提取地面
        ground_cloud, segmented_cloud = ground_segmentation_using_ransac(data=origin_points)

        ground_cloud_o3d = point_cloud_to_instance(ground_cloud)
        ground_cloud_o3d.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([ground_cloud_o3d])  # 显示提取的地面

        # 聚类
        clusters_index = clustering_using_dbscan(segmented_cloud)
        # print("clusters_index:", clusters_index)

        clustered_points_o3d = plot_clusters(segmented_cloud, clusters_index)

        # o3d.visualization.draw_geometries(clustered_points_o3d)

        clustered_points_o3d.append(ground_cloud_o3d)
        # o3d.visualization.draw_geometries(clustered_points_o3d)   # 显示最终结果


if __name__ == '__main__':
    main()
