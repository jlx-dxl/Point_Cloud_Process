import torch
import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


# 读取以逗号分隔的txt文件
# 输入：文件名
# 输出：np.array
def read_pcd_from_file(file):
    # np_pts = np.zeros(0)
    with open(file, 'r') as f:
        pts = []
        for line in f:
            one_pt = list(map(float, line[:-1].split(',')))
            pts.append(one_pt[:3])
        np_pts = np.array(pts)
    return np_pts


# 中心归一化
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# 沿Z轴随机旋转点云
def rotate_point_cloud_z(origin_data):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(origin_data, rotation_matrix)
    return rotated_data


# 从文件名list中读取文件名
# 输入：列有一系列文件名的文件的名称
# 输出：包含接下来要读取数据的文件的文件名的List
def read_file_names_from_file(file):
    with open(file, 'r') as f:
        files = []
        for line in f:
            files.append(line.split('\n')[0])
    return files


class PointNetDataset(Dataset):
    def __init__(self, root_dir, train):
        super(PointNetDataset, self).__init__()

        self._train = train
        self._classes = []

        self._features = []
        self._labels = []

        self.load(root_dir)

    def classes(self):
        return self._classes

    # 数据集中sample的数量
    def __len__(self):
        return len(self._features)

    # 给定index就可以通过p[index]的方式取一个sample（p为该类的实例）
    def __getitem__(self, idx):
        feature, label = self._features[idx], self._labels[idx]

        # 数据增强方式：中心归一化，z方向点云随机旋转，点云随机扰动
        # centroid-normalize
        feature = pc_normalize(feature)
        # z-rotation
        feature = rotate_point_cloud_z(feature)
        # jitter（抖动）
        feature += np.random.normal(0, 0.02, size=feature.shape)
        feature = torch.Tensor(feature.T)

        l_lable = [0 for _ in range(len(self._classes))]
        l_lable[self._classes.index(label)] = 1
        label = torch.Tensor(l_lable)

        return feature, label

    def load(self, root_dir):
        things = os.listdir(root_dir)
        files = []
        for f in things:
            # 从modelnet40_train.txt读取包含训练集的samples的文件名
            if self._train == 0:
                if f == 'modelnet40_train.txt':
                    print("Reading datas form:", f)
                    files = read_file_names_from_file(root_dir + '/' + f)
                    # print("which contains:", files)
            # 从modelnet40_test.txt读取包含测试集的samples的文件名
            elif self._train == 1:
                if f == 'modelnet40_test.txt':
                    print("Reading datas form:", f)
                    files = read_file_names_from_file(root_dir + '/' + f)
                    # print("which contains:", files)
            # 从modelnet40_shape_names.txt读出所有类别标签
            if f == "modelnet40_shape_names.txt":
                self._classes = read_file_names_from_file(root_dir + '/' + f)

        tmp_classes = []
        # 根据刚刚得到的文件名list，逐一读文件
        for file in files:
            num = file.split("_")[-1]  # 数字序号部分
            kind = file.split("_" + num)[0]  # 名称部分
            # print("Sussfully read file:", kind, num)
            if kind not in tmp_classes:
                print("Start reading kind:", kind, "(", len(tmp_classes) + 1, "in", len(self._classes), ")")
                tmp_classes.append(kind)
            # 读数据
            pcd_file = root_dir + '/' + kind + '/' + file + '.txt'
            np_pts = read_pcd_from_file(pcd_file)
            # print(np_pts.shape) # (10000, 3)

            # 将sample和对应的label按序堆进list，方便取用
            self._features.append(np_pts)
            self._labels.append(kind)

        if self._train == 0:
            print("There are " + str(len(self._labels)) + " trian files.")
        elif self._train == 1:
            print("There are " + str(len(self._labels)) + " test files.")


if __name__ == "__main__":
    train_data = PointNetDataset("../dataset/modelnet40_normal_resampled", train=0)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)  # 每次取2个sample
    cnt = 0
    for pts, label in train_loader:
        print(pts.shape)
        print(label.shape)  # 两个label是2*40，每个label一个40维的01向量（40是标签的总种类数）
        cnt += 1
        if cnt > 3:
            break
