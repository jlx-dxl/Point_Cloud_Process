# 该文件定义了在树中查找数据所需要的数据结构，类似一个中间件

import copy

import numpy as np

# 类：
class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance

# 解集：自排序容器（KNNResultSet.dist_index_list的每一个元素都是一个DistIndex类的对象）
class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity   # 最大长度（容量）
        self.count = 0   # 目前有效长度
        self.worst_dist = np.Inf
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))   # 按最大长度初始化成distance全为inf，index全为0的DistIndex
        # 统计进行了几次比较
        self.comparison_counter = 0

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def worstDist(self):
        return self.worst_dist

    def add_point(self, dist, index):
        self.comparison_counter += 1
        # 如果dist>worst_dist，不将这个点加入解集，退出函数
        if dist > self.worst_dist:
            return

        # 以下步骤只有dist<worst_dist的时候才执行
        # 如果集合还没满，需要把count+1，如果集合已满（count=capacity），直接操作集合内的元素即可，不用动count
        if self.count < self.capacity:
            self.count += 1

        # 从后往前（从倒数第二个元素开始）依次将value>dist的元素向后移动一位，最后i将停留在插入所需的索引处
        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                i -= 1
            else:
                break

        # 将点插入i索引处即可
        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity-1].distance
        
    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output

# 解集：自排序容器（KNNResultSet.dist_index_list的每一个元素都是一个DistIndex类的对象）
class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []

        self.comparison_counter = 0

    def size(self):
        return self.count

    def worstDist(self):
        return self.radius

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return

        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))

    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count, self.radius, self.comparison_counter)
        return output


