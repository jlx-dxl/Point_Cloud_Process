# kdtree的具体实现，包括构建和查找
###################
# @author:jlx0424
# 可以直接运行
###################

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet


# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    # 把Node的属性axis,value,point_indices排列好打印出来
    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:  # 如果是leaf
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


# 功能：构建树之前需要对value进行排序，同时对应的的key的顺序也要跟着改变
# 输入：
#     key：在原始点云中的index
#     value:某个维度的坐标值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1  # ？
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


# 分割轴的轮换（dim=3：0,1,2,0,1,2...）
def axis_round_robin(axis, dim):
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据(db.shape=(m,3)，其中m为点的个数)
#     point_indices：排序后的index
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树（用根节点即可代表整棵树）
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # 作业1
        # 屏蔽开始
        point_indices_sorted, value_sorted = sort_key_by_vale(point_indices, db[point_indices, axis])  # 将所有点按照axis方向
        # 找到分界点的索引
        division_point_index = int(point_indices_sorted.shape[0] / 2)  # 在0.5m个点处分界
        # 左边
        left_points_index = point_indices_sorted[0:division_point_index]  # 左边点集的索引集
        left_point_value = db[division_point_index - 1, axis]  # 分界线左边一个点的value
        # 右边
        right_points_index = point_indices_sorted[division_point_index:]  # 右边点集的索引集
        right_point_value = db[division_point_index, axis]  # 分界线右边一个点的value
        # 分界线的value=分界线左右两点value的平均值
        root.value = 0.5 * (left_point_value + right_point_value)
        # 递归建立左右子节点
        root.left = kdtree_recursive_build(root.left, db, left_points_index, axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right, db, right_points_index, axis_round_robin(axis, dim=db.shape[1]),
                                            leaf_size)
        # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        # print(root)
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]  # 取出leaf里的所有points的坐标
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points,
                              axis=1)  # 分别计算query point和leaf里的所有points的距离，结果存diff向量中
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])  # 每个点是否真的加入了解集在函数内部判断
        return True

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)  # 如果query point在某片区域里，则这篇区域要搜索
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set,
                              query)  # 如果query point在某片区域里，但是和对侧区域的距离小于worst-distance，则对侧区域也要搜索
    else:
        kdtree_knn_search(root.right, db, result_set, query)  # 如果query point在某片区域里，则这篇区域要搜索
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set,
                              query)  # 如果query point在某片区域里，但是和对侧区域的距离小于worst-distance，则对侧区域也要搜索
    # 屏蔽结束

    return True


# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)  # 如果query point在某片区域里，则这篇区域要搜索
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set,
                                 query)  # 如果query point在某片区域里，但是和对侧区域的距离小于worst-distance，则对侧区域也要搜索
    else:
        kdtree_radius_search(root.right, db, result_set, query)  # 如果query point在某片区域里，则这篇区域要搜索
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set,
                                 query)  # 如果query point在某片区域里，但是和对侧区域的距离小于worst-distance，则对侧区域也要搜索
    # 屏蔽结束

    return False


def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 64
    k = 8

    np.random.seed(1)
    db_np = np.random.rand(db_size, dim)
    # print("dp_np:", db_np)
    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    print("k-NN search:")
    begin_t = time.time()
    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)

    # print(result_set)

    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    print("Radius search:")
    begin_t = time.time()
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius=0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))


if __name__ == '__main__':
    main()
