# octree的具体实现，包括构建和查找
###################
# @author:jlx0424
# 可以直接运行
###################

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet


# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children  # 8个子Octants
        self.center = center  # cube中心的坐标
        self.extent = extent  # cube边长的一半（中心到面的距离）
        self.point_indices = point_indices  # cube中所包含的点的索引
        self.is_leaf = is_leaf  # 是否是leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "  # 只打印children中的非空成员
        output += 'point_indices: ' + str(self.point_indices)
        return output


# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        # print(root)
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    # 如果在一个cube中没有点，将该Octant置为None
    if len(point_indices) == 0:
        return None

    # 如果一个结点为None，则把其子结点的root都设为None（其他值还是会进行更新）
    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # 如果cube中的点的个数小于leaf_size或cube的边长小于某值，这个octant就是leaf了
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        # 作业4
        # 屏蔽开始
        root.is_leaf = False
        children_point_index = [[] for i in range(8)]

        # 根据8种情况将父结点种的点放入相应的子节点中：粗暴分类法
        # for point_index in point_indices:
        #     point = db[point_index, :]
        #     if point[0] < center[0] and point[1] < center[1] and point[2] < center[2]:
        #         children_point_index[0].append(point_index)
        #     elif point[0] >= center[0] and point[1] < center[1] and point[2] < center[2]:
        #         children_point_index[1].append(point_index)
        #     elif point[0] < center[0] and point[1] >= center[1] and point[2] < center[2]:
        #         children_point_index[2].append(point_index)
        #     elif point[0] >= center[0] and point[1] >= center[1] and point[2] < center[2]:
        #         children_point_index[3].append(point_index)
        #     elif point[0] < center[0] and point[1] < center[1] and point[2] >= center[2]:
        #         children_point_index[4].append(point_index)
        #     elif point[0] >= center[0] and point[1] < center[1] and point[2] >= center[2]:
        #         children_point_index[5].append(point_index)
        #     elif point[0] < center[0] and point[1] >= center[1] and point[2] >= center[2]:
        #         children_point_index[6].append(point_index)
        #     elif point[0] >= center[0] and point[1] >= center[1] and point[2] >= center[2]:
        #         children_point_index[7].append(point_index)

        # 根据8种情况将父结点种的点放入相应的子节点中：位运算法
        for point_index in point_indices:
            point_db = db[point_index, :]
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_index[morton_code].append(point_index)

        # 创建子结点
        factor = [-0.5, 0.5]
        # 依次创建8个
        for i in range(8):
            # 计算子节点的中心
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            # 计算子节点的边长
            child_extent = 0.5 * extent
            # 创建子节点
            root.children[i] = octree_recursive_build(root.children[i], db, child_center, child_extent,
                                                      children_point_index[i], leaf_size, min_extent)
        # 屏蔽结束
    return root


# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def inside(query: np.ndarray, radius: float, octant: Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant: Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int32)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


# 功能：判断当前query是否包含octant
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def contains(query: np.ndarray, radius: float, octant: Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius


# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 作业5
    # 提示：尽量利用上面的inside、overlaps、contains等函数
    # 屏蔽开始
    # 优化：利用contain函数检查是否query圆能包含整个octant，此时这个octant就不用再细分了
    if contains(query, result_set.worstDist(), root):
        # compare the contents of the octant
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # don't need to check any child
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # no need to go to most relevant child first, because anyway we will go through all children
    for c, child in enumerate(root.children):
        if child is None:
            continue
        if not overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search_fast(child, db, result_set, query):
            return True
    # 屏蔽结束

    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业6 (和k-NN search的代码一样)
    # 屏蔽开始
    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # go to the relevant child first
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_radius_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if not overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search(child, db, result_set, query):
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业7
    # 屏蔽开始
    # 搜索最相关的一个子结点（枚举法）
    # morton_code = 0
    # if query[0] < root.center[0] and query[1] < root.center[1] and query[2] < root.center[2]:
    #     morton_code = 0
    # elif query[0] >= root.center[0] and query[1] < root.center[1] and query[2] < root.center[2]:
    #     morton_code = 1
    # elif query[0] < root.center[0] and query[1] >= root.center[1] and query[2] < root.center[2]:
    #     morton_code = 2
    # elif query[0] >= root.center[0] and query[1] >= root.center[1] and query[2] < root.center[2]:
    #     morton_code = 3
    # elif query[0] < root.center[0] and query[1] < root.center[1] and query[2] >= root.center[2]:
    #     morton_code = 4
    # elif query[0] >= root.center[0] and query[1] < root.center[1] and query[2] >= root.center[2]:
    #     morton_code = 5
    # elif query[0] < root.center[0] and query[1] >= root.center[1] and query[2] >= root.center[2]:
    #     morton_code = 6
    # elif query[0] >= root.center[0] and query[1] >= root.center[1] and query[2] >= root.center[2]:
    #     morton_code = 7
    # if octree_knn_search(root.children[7], db, result_set, query):
    #     return True

    # 搜索最相关的一个子结点（位运算法）
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if not overlaps(query, result_set.worstDist(), child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root


def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 64
    min_extent = 0.0001
    k = 8

    begin_t = time.time()
    # np.random.seed(1)
    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    depth = [0]
    max_depth = [0]
    traverse_octree(root, depth, max_depth)
    construction_t = time.time()
    print("tree max depth: %d" % max_depth[0])
    print("Construction takes %.3fms\n" % ((construction_t - begin_t) * 1000))

    print("k-NN search:")
    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    octree_knn_search(root, db_np, result_set, query)
    print(result_set)

    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print("index:", nn_idx[0:k])
    print("distance:", nn_dist[0:k])
    knn_t = time.time()
    print("k-NN search takes %.3fms\n" % ((knn_t - construction_t) * 1000))

    query = np.random.rand(3)

    begin_t = time.time()
    print("Radius search normal:")
    result_set = RadiusNNResultSet(radius=1)
    octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("Radius search fast:")
    result_set = RadiusNNResultSet(radius=1)
    octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))


if __name__ == '__main__':
    main()
