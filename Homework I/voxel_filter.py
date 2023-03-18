##########################################
# @@作者：贾林轩
# @@运行后不断按esc就能看到origin,centroid,random的点云效果
###########################################

import open3d as o3d 
import os
import time
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, Type):
    filtered_points = []
    # 作业3
    # 屏蔽开始

    point_cloud = point_cloud.to_numpy()
    print("point_cloud:",point_cloud,point_cloud.shape)

    # 上下限整除cell size=各维度有多少格子
    D = (np.max(point_cloud, axis=0, keepdims=True) - np.min(point_cloud, axis=0, keepdims=True)) // leaf_size
    print("D:",D,D.shape)

    # 每个点的h_x,h_y,h_z坐标
    h = (point_cloud - np.min(point_cloud, axis=0, keepdims=True)) // leaf_size
    print("h:",h,h.shape)

    # 每个点的索引
    h_index = np.zeros((point_cloud.shape[0],1))
    for i in range(point_cloud.shape[0]):
        h_index[i] = h[i][0] + h[i][1] * D[0][0] + h[i][2] * D[0][0] * D[0][1]
    print("h_index:",h_index,h_index.shape)

    # 排序，返回排序的索引值
    sorted_index = np.argsort(h_index, axis=0)
    print("sorted_index:", sorted_index, sorted_index.shape)

    # 初始化，value初始化成0.5是因为h_index都是整数，这样第一次进入循环时必进入value != h_index[sorted_index[i]]的分支
    value = 0.5
    cache = []   # 定义一个cache用于暂存h_index相同的这些点的坐标

    for i in range(sorted_index.shape[0]):
        # print("i:",i)

        # 如果目前点的index和value（先前点的index）相同，将其装入cache
        if value == h_index[sorted_index[i]]:
            cache.append(point_cloud[sorted_index[i],:])
            # print("value not changed:",value, value.shape)
            # print("cache appended:",cache,len(cache))

        # 如果目前点的index和value（先前点的index）不同，说明当前voxel中的points都遍历完了，可以根据策略的不同选取其中一个，并释放cache
        elif value != h_index[sorted_index[i]]:
            value = h_index[sorted_index[i]]   # 记录这个point的index即为下一个voxel的index
            # print("value changed:",value,value.shape)

            if Type == "Random":
                if len(cache)>0: # 这个判断条件是为了避免第一次进入时将空矩阵装入结果
                    cache = np.asarray(cache)
                    # print("array cache:",cache, cache.shape)
                    selected = cache[np.random.randint(0, len(cache), size=1),:]   # 随机从cache中选择一个点
                    # print("selected point:",selected, selected.shape)
                    selected = np.squeeze(selected, axis=0)
                    # print("selected point:", selected, selected.shape)
                    filtered_points.append(selected)   # 将选择的点装入结果
                cache = []

            elif Type == "Centroid":
                if len(cache)>0:
                    cache = np.asarray(cache)
                    # print("array cache:", cache, cache.shape)
                    cache = np.squeeze(cache, axis=1)   # 取坐标均值即为中心点
                    # print("array cache:", cache, cache.shape)
                    selected = np.mean(cache, axis=0, keepdims=True)
                    # print("selected point:", selected, selected.shape)
                    filtered_points.append(selected)   # 将选择的点装入结果
                cache = []
        # time.sleep(1)

    # 屏蔽结束
    # 把点云格式改成array，并对外返回
    filtered_points = np.asarray(filtered_points, dtype=np.float64)
    filtered_points = np.squeeze(filtered_points,axis=1)
    print("filtered_points:",filtered_points,filtered_points.shape)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    data_dir='./modelnet40_normal_resampled/'

    with open(data_dir+'modelnet40_shape_names.txt') as f:
        a = f.readlines()
    for i in a:
        point_cloud_pynt = PyntCloud.from_file(
            data_dir+'{}/{}_0001.txt'.format(i.strip(), i.strip()), sep=",",
            names=["x", "y", "z", "nx", "ny", "nz"])
        # 转成open3d能识别的格式
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        o3d.visualization.draw_geometries([point_cloud_o3d],"Origin") # 显示原始点云

        # 调用voxel滤波函数，实现滤波
        points = point_cloud_pynt.points
        points = points.iloc[:, 0:3]

        # 均值法
        filtered_cloud = voxel_filter(points, 0.03, Type="Centroid")
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_o3d],"Centroid")

        # 随机法
        filtered_cloud = voxel_filter(points, 0.03, Type="Random")
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_o3d],"Random")

if __name__ == '__main__':
    main()
