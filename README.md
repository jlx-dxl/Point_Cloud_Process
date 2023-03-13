# Point_Cloud_Process

# I. Introduction

## 0. Basic Tasks for Point Cloud Processing

1. PCA 主成分分析
2. 降采样
3. 滤波

## 1. How to present 3D objects?

![image.png](https://s2.loli.net/2023/03/13/oiTNISJzVWlxGPZ.png)

- mesh: 三角形网格，在图形学方面比较有效（游戏）
- voxel: 体素，过于稠密
- Octree: 八叉树，只在有物体的地方进行细分
- Point Cloud: 点云，实质上就是矩阵

## 2. Difficulties of Point Cloud Processing

- Sparsity：近稠密，远稀疏
- Irregular：不规则，不容易找neighbor
- Lack of texture information：缺乏纹理信息（三人成车，极易误判）
- Un-ordered：矩阵行交换后，从逻辑上讲表达的仍是同一个物体，但是对于深度学习来说输入变了，可能导致不同的输出
- Rotation equivariance：旋转后，从逻辑上讲表达的仍是同一个物体，但是对于深度学习来说输入变了，可能导致不同的输出

## 3. Methods for Point Cloud Processing

![image.png](https://s2.loli.net/2023/03/13/Pd85sCLGJTRmA7b.png)

![image.png](https://s2.loli.net/2023/03/13/81AMHSYtQdPTk7R.png)

# II. Basic Algorithms

## 1. Principle Component Analysis (PCA)

### (1). Singular Value Decomposition (SVD) 

#### i. 如何理解

任意一个矩阵$M_{m\times n}$都可以被分解成$M=U\Sigma V^T$的形式，其中

- $U_{m\times m}$和$V_{n\times n}$是标准正交矩阵（每一列都是空间中的一组标准正交基），可以用作旋转矩阵
- $\Sigma_{m\times n}$是对角阵+0，表示对$min(m,n)$个基进行的拉伸与缩放
- 其效果可以理解为一种模态分解（如下图）
- ![image.png](https://s2.loli.net/2023/03/13/FZiWVRGUhzowBQ3.png)

![image.png](https://s2.loli.net/2023/03/13/ZJXnPeWUHOxA5kl.png)

假设m>n，则可以做此修改

![image.png](https://s2.loli.net/2023/03/13/VgxeCt1suwmLoPy.png)

若只关心r个奇异值下的模态，则可以做此变换

![image.png](https://s2.loli.net/2023/03/13/1TtXkxOP7zscipD.png)

#### ii. 如何求

$$
To\ clculate:M=U\Sigma V^T\\
M^TM=(U\Sigma V^T)^TU\Sigma V^T\\
=V\Sigma U^TU\Sigma V^T&(U^TU=E)\\
=V\Sigma\Sigma V&(L=\Sigma\Sigma=\begin{bmatrix}\sigma_1^2&0\\0&\sigma_2^2 \end{bmatrix})\\
M^TM=VLV^T\\
The\ same\ can\ be\ proved:\\
MM^T=ULU^T\\
Therefore:\\
V\ is\ the\ Feature\ Vectors\ of\ M^TM\\
M\ is\ the\ Feature\ Vectors\ of\ MM^T\\
L\ is\ the\ Feature\ Values\ of\ MM^T\ or\ M^TM\\
\Sigma\ is\ the\ Singular\ Values\ of\ MM^T\ or\ M^TM\\
$$

![image.png](https://s2.loli.net/2023/03/13/erL39qya87YhCjw.png)

### (2). Principle Component Analysis (PCA)

#### i. 如何理解

![image.png](https://s2.loli.net/2023/03/13/C9OYSH1Usp2latJ.png)

目标：找到一个新的坐标系，这个坐标系的各个轴叫做主成分，使的所有数据点在我们所关心的各个主成分上最分散（方差最大）

在中心且数据在各个维度都符合标准正态分布的数据叫做D，我们手上的数据去中心化（各个维度的均值移动到原点）后叫做$D'$

![image.png](https://s2.loli.net/2023/03/13/5c4DPvbQRdVXBKJ.png)

![image.png](https://s2.loli.net/2023/03/13/cz8xb4aAM6nfD9o.png)

![image.png](https://s2.loli.net/2023/03/13/LmSQjFzKbo4AlTh.png)

其中：

- S为对角阵
- R为标准正交阵（每一列都是空间中的一组标准正交基），可以用作旋转矩阵

因此，我们要求的就是R，它既表示了从$D$到$D'$的旋转角度，也表示了要找的新坐标轴（一组正交基）

如何求R？——协方差矩阵的特征向量就是R



#### ii. 如何求

