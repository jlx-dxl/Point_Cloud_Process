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

![image.png](https://s2.loli.net/2023/03/13/5fS3gRHb6hwqlY1.png)

![image.png](https://s2.loli.net/2023/03/13/cVwlpsHzG4UBAWe.png)

![image.png](https://s2.loli.net/2023/03/13/gRDQczYIGAUaL2e.png)

对于一组白数据$D$：

- 拉伸时，只有对角线元素（方差）变化
- 旋转时，协方差矩阵所有元素都变化

![image.png](https://s2.loli.net/2023/03/13/wCkbqW9hP2HfN35.png)

#### ii. 如何求

**协方差矩阵的特征向量就是R**

![image.png](https://s2.loli.net/2023/03/13/sVgEh5HxktfRBbu.png)
$$
C'=\frac1{n-1}D'D'^T
=\frac1{n-1}RSD(RSD)^T
=\frac1{n-1}RSDD^TS^TR^T\\
其中\frac1{n-1}DD^T=C（白数据的协方差矩阵）=E\\
\therefore C'=RSS^TR^T\\
其中S为对角阵,S=S^T,R为正交阵,R^T=R^{-1}\\
令L=SS^T=\begin{bmatrix}a^2&0\\0&b^2\end{bmatrix}\\
\therefore C'=RLR^{-1}
$$
![image.png](https://s2.loli.net/2023/03/13/I58tjOzwlgJY7bv.png)

![image.png](https://s2.loli.net/2023/03/13/9DN4mjhvgOtq2Sr.png)

### (3). PCA与SVD的关系

由于在SVD中，V是$M^TM$的特征向量，而PCA主成分向量R定义为协方差矩阵$C=\frac1{n}M^TM$的特征向量，因此V与R为同一方向

### (4). PCA-2

#### i. Spectral Theorem（SVD中M为对称阵时的特殊情况）

$$
A=U\Lambda U^T=\sum^n_{i=1}\lambda_iu_iu_i^T,\Lambda=diag(\lambda_1,\dots,\lambda_n)
$$

#### ii. Rayleigh Quotients

- 定义：

- $$
  \lambda_{min}(A)\le\frac{x^TAx}{x^Tx}\le\lambda_{max}(A)
  $$

- 证明：

- ![image.png](https://s2.loli.net/2023/03/16/UWuYXBxEKSVbPRj.png)

#### iii. Calculation of PCA

$$
given\ X_{n\times m}=\{x_1,\dots,x_m\},x_i\in\mathbb{R}^n\\
1.\ Normalize\ by\ the\ center\\
\tilde{X}_{n\times m}=[\tilde x_1,\dots,\tilde x_m],\tilde x_i=x_i-\bar x_i,\bar x_i=\frac1m\sum_{i=1}^mx_i\\
2.\ Compute\ SVD\\
H_{n\times n}=\frac1n\tilde{X}\tilde{X}^T=U_r\Sigma^2U_r^T\\
3.\ The\ Principle\ vectors\ are\ the\ columns\ of\ U_r
$$

#### iiii. Application of PCA: – Dimensionality Reduction

$$
given\ X_{n\times m}=\{x_1,\dots,x_m\},x_i\in\mathbb{R}^n\\
perform\ PCA\ to\ get\ l(l<<n)\ principle\ components\ A_{l\times m}=\{a_1,\dots,a_l\},a_j\in\mathbb{R}^n\\
for\ single\ example:\\
a^l_{l\times1}=(Z^T)_{l\times n}(x_i)_{n\times1}\\
for\ all\ the\ examples:\\
A_{l\times m}=(Z^T)_{l\times n}X_{n\times m}
$$

这个过程称为Encoder，将n维（成百上千）的数据X降为l维（十或数十）进行表达；

还可以利用逆过程，利用主成分（特征向量）$U_r(Z)$和主成分下的坐标$A$，还原原数据，称为Decoder：$\hat{X}_{n\times m}=Z_{n\times l}A_{l\times m}$

---

## 2. Kernel PCA

### (1). 基本思想

将低维的数据通过一定的映射函数$\phi$映射到高维空间中，在高维空间中，数据的分布会更加清晰，易于进行聚类和分割

| Origin                                                       | Lifting function                                             | lifted to high dimension                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image.png](https://s2.loli.net/2023/03/16/DRZnLC2tg6QcEAj.png) | ![image.png](https://s2.loli.net/2023/03/16/WCxJauAhOfk4o6M.png) | ![image-20230316164111730](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20230316164111730.png) |
|                                                              |                                                              |                                                              |

例如上图，通过映射函数将数据从二维映射到三维，就能通过简单的平面将不同颜色的数据分开

### (2). 算法原理

$$
1.Lift\\
X_{n_0\times m}\to\Phi(X)_{n_1\times m}\\
2.Normalize\\
\tilde\Phi(X)_{n_1\times m}=\Phi(X)_{n_1\times m}-\bar\Phi(X)_{1\times m}\\
3.Compute SVD\\
\tilde{H}_{n_1\times n_1}=\frac{1}{n_1}\tilde\Phi(X)_{n_1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\\
4.Solve\ the\ eigenvactors\ of \tilde H\to \tilde Z_{n_1\times l}
$$

但我们不知道$\phi$如何定义，在这种情况下，如何求解$\tilde Z_{n_1\times l}$(假设我们只关心前$l$个主成分（$l\le n_1$）)？

上述特征向量可以表示成数据点的线性组合：
$$
\because\tilde H_{n_1\times n_1}\tilde Z_{n_1\times l}=\tilde\lambda\tilde Z_{n_1\times l}\\
\because\tilde{H}_{n_1\times n_1}=\frac{1}{n_1}\tilde\Phi(X)_{n_1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\\
\therefore\frac{1}{n_1}\tilde\Phi(X)_{n_1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\tilde Z_{n_1\times l}=\tilde\lambda\tilde Z_{n_1\times l}\\
$$
取其中一个特征向量$z_{n_1\times1}$进行观察，有：
$$
\frac{1}{n_1}\tilde\Phi(X)_{n_1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\tilde z_{n_1\times 1}=\tilde\lambda\tilde z_{n_1\times 1}\\
$$
定义：
$$
a_{m\times1}=\frac{1}{n_1\tilde\lambda}(\tilde\Phi(X)^T)_{m\times n_1}\tilde z_{n_1\times 1}\\
$$
则有：
$$
\tilde z_{n_1\times 1}=\tilde\Phi(X)_{n_1\times m}a_{m\times1}=\sum_{i=1}^ma_i\phi(x_i)_{n_1\times1}
$$
因此对$l$个特征向量有：
$$
\tilde Z_{n_1\times l}=\tilde\Phi(X)_{n_1\times m}A_{m\times l}
$$
将其带回特征方程中，有：
$$
\frac{1}{n_1}\tilde\Phi(X)_{n_1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\tilde\Phi(X)_{n_1\times m}A_{m\times l}=\tilde\lambda\tilde\Phi(X)_{n_1\times m}A_{m\times l}\\
$$
定义：
$$
K_{m\times m}=(\tilde\Phi(X)^T)_{m\times n_1}\tilde\Phi(X)_{n_1\times m}
$$
则上式可变形为（左右两边同乘$(\tilde\Phi(X)^T)_{m\times n_1}$）：
$$
KKA=\tilde\lambda n_1KA\\
equal\ to:KA=\Lambda'A
$$
即，A为K的特征向量，$\Lambda'$为K的特征值，但由于$\tilde Z_{n_1\times l}=\tilde\Phi(X)_{n_1\times }A_{m\times l}$，而我们并不知道$\tilde\Phi(X)_{n_1\times m}$，因此$\tilde Z$仍未求出，

但事实上我们并不需要知道$\tilde\Phi(X)_{n_1\times m}$，在应用中，我们只需要知道高维数据$\tilde\Phi(X)_{n_1\times m}$在各主成分下的投影即可，因此我们实际上关心的量是$Z^T_{l\times n_1}\Phi_{n_1\times m}$（见II.1.(4).iiii），令其为$X'_{l\times m}$
$$
X'_{l\times m}=(A^T)_{l\times m}(\tilde\Phi(X)^T)_{m\times n_1}\tilde\Phi(X)_{n_1\times m}=(A^T)_{l\times m}K_{m\times m}
$$
至此，我们就可以求出高维数据在各主成分下的投影了，但仍存在两个小问题：

1. 此时高维数据$\tilde\Phi(X)_{n_1\times m}$并不是零均值的
2. 此时的主成分Z（基底）并不是单位向量

对问题1，进行如下操作：
$$
\tilde\Phi(X)_{n_1\times m}=\Phi(X)_{n_1\times m}-Ones_{n_1\times1}\bar\Phi(X)_{1\times m}\\
\tilde K=\tilde\Phi^T\Phi=(\Phi_{n_1\times m}-Ones_{n_1\times1}\bar\Phi(X)_{1\times m})^T(\Phi_{n_1\times m}-Ones_{n_1\times1}\bar\Phi(X)_{1\times m})\\
=\Phi^T\Phi-2*(\bar\Phi^T)_{m\times1}Ones_{1\times n_1}\Phi_{n_1\times m}+(\bar\Phi^T)_{m\times1}Ones_{1\times n_1}Ones_{n_1\times1}\bar\Phi(X)_{1\times m}\\
=K-2*(\frac1{n_1})*Ones_{m\times m}K-(\frac1{n_1})*Ones_{m\times m}K*(\frac1{n_1})*Ones_{m\times m}
$$
将所得的$\tilde K$带入上式中求解，即可保证高维数据为0均值；

对问题2，进行如下操作：

令每一个高维数据的主成分$\tilde z$满足$\tilde z^T\tilde z=1$，因为$\tilde z_{n_1\times 1}=\tilde\Phi(X)_{n_1\times m}a_{m\times1}$

所以有$(a^T)_{1\times m}(\tilde\Phi(X)^T)_{m\times n_1}\tilde\Phi(X)_{n_1\times m}a_{m\times1}=1$

即$(a^T)_{1\times m}K_{m\times m}a_{m\times1}=1$

由于A是K的特征向量，因此$KA=\Lambda'A$

即，上式可变为：$(a^T)_{1\times m}\lambda'a_{m\times1}=1$

即满足$norm(a_r)=\frac1{\lambda'_r}$

### (3). 应用解法

根据经验，常用的K有以下几种形式，在应用中可以尝试每种形式，观察变化后的数据分布，选择最符合需要的一个；

![image.png](https://s2.loli.net/2023/03/16/IkJdOipjczXHZgm.png)

算法流程：

1. 选择一个kernel function $k$，并计算Gram matrix $K(i,j)_{m\times m}=k((x_i)_{n\times1},(x_j)_{n\times1})$
2. Normalize K：$\tilde K=K-2*(\frac1{n_1})*Ones_{m\times m}K-(\frac1{n_1})^2*Ones_{m\times m}KOnes_{m\times m}$
3. 求$\tilde K$的特征向量$A$和特征值$\Lambda'$
4. Normalize $A$：令$a_r^Ta_r=\frac1{\lambda_r}$
5. 计算高维数据在主成分上的投影：$X'_{l\times m}=Z^T_{l\times n_1}\Phi_{n_1\times m}=(A^T)_{l\times m}K_{m\times m}$

## 3. Surface Normal Estimation 表面法向量估计

### (1). 基本流程

1. 选择一个点P
2. 通过定义一个邻域决定一个表面
3. 进行PCA分析
4. 选取特征值最小的一个特征向量
5. 曲率$Curvature=\frac{\lambda_{min}}{\sum\lambda}$

![image.png](https://s2.loli.net/2023/03/16/eEVvyAjUaF8rPH1.png)

### (2). 基本方法存在的问题

1. 噪声问题
   1. Weighted based on other features
   2. RANSAC
   3. Deep Learning Methods
2. 邻域选择问题
   1. 邻域过大：缺乏精细结构
   2. 邻域过小：易受噪声影响

### (3). 带加权的法向量估计

设计一个权重矩阵$W=diag(w1,\dots,w_m)$，对$\tilde XW\tilde X^T$进行PCA分解，选择最小的特征值对应的特征向量

==**W的设计？**==

### (4). 基于深度学习的法向量估计

![image.png](https://s2.loli.net/2023/03/16/3A1VFRm8nthf6y4.png)

## 4. 滤波

### (1). 作用

1. 去除噪声

2. 降采样（保留特征的同时降低数据量）

3. 上采样（eg. 多传感器融合时，希望将稀疏的点云数据在视觉图像上稠密的表达）【BF：Bilateral Filter 双边滤波；MED：中值滤波；AVE：均值滤波】

   ![image.png](https://s2.loli.net/2023/03/16/1X67zeQtIqZOs9o.png)

### (2). 去除噪声

#### i.  Radius Outlier Removal

![image.png](https://s2.loli.net/2023/03/16/xJfV1iUyma8vTKo.png)

#### ii. Statistical Outlier Removal

[(24条消息) PCL学习笔记（十五）-- StatisticalOutlierRemoval滤波器移除离群点_statistical outlier removal_看到我请叫我学C++的博客-CSDN博客](https://blog.csdn.net/qq_45006390/article/details/119100938)

![image.png](https://s2.loli.net/2023/03/16/FZ9IOeXYWSD45Ha.png)

1. 遍历所有点，计算每个点和其他点的距离，取前k个（全部m个点和邻域内k个点，共m*k个数据：$D_{m\times k}$）
2. 计算这些数据的均值$\mu$和方差$\sigma$
3. 计算k邻域内每个点距离的均值$\bar D_{1\times m}=\{\bar d_1,\dots,\bar d_m\}$
4. $if\ \bar d_i>\mu+3\sigma$，移除该点

### (3). 降采样

#### i. Voxel Grid Downsampling  体素降采样

将整个点云分割成一定尺寸的cells，即Voxel Grids，每个cell中可能存在多个点，这时可以：

1. 取这多个点的平均值
   1. 优点：更准确
   2. 缺点：更慢；且坐标可以取平均，但诸如语义，标签等信息不能做平均
2. 从中随机选取一个点
   1. 优点：更快
   2. 缺点：不够精确

![](https://s2.loli.net/2023/03/16/Zrz53cOXkvY1USx.png)

1. 取坐标上下限，根据设定的voxel的大小，分割成$D_x\times D_y\times D_y$个voxels
2. 计算每个voxel的index，$h=h_x+h_y*D_x+h_z*D_x*D_y$，其中$h_x,h_y,h_z$是和对应坐标下限的差整除r所得
3. 将这些点根据index排序，即可将在同一voxel中的点排到一起

#### ii. Hash Downsampling

![image.png](https://s2.loli.net/2023/03/16/6aUi8zxZWL1yO7D.png)

- 由于排序较为复杂和耗时，设计用一个hash映射，将$(h_x,h_y,h_z)$映射到一个值$G_i\in\{G_1,\dots,G_m\}$，
- 当hash function选取不当时，可能会出现conflict，意为：$hash(h_x,h_y,h_z)=hash(h_x',h_y',h_z')$，但$h_x\ne h_x'orh_y\ne h_y'orh_z\ne h_z'$ 如何解决？

#### iii. Farthest Point Sampling (FPS)

![image.png](https://s2.loli.net/2023/03/17/OCBPLcbTHGh8iMA.png)

1. 随机选取一个点加入FPS set(F)
2. 从原始点云P中找到离它最远的点加入F
3. 遍历P-F中的点，找到与F中的点的最小值最大的一个点，加入F
4. 循环3，直到F中有所需数目的点

#### iiii. Normal Space Sampling (NSS)

每一个点都有一个法向量，每一个法向量都有一个方向，把这些法向量normalize后可以得到一个球面（各个方向分布不均），在法向量空间中（球面），等分为若干份，从每份中保留一定数目的法向量（这时球面内各个方向的法向量数目分布均匀），根据法向量的删减与保留信息处理原始点云

优点：可以保留精细结构（点云对齐）

#### iv. Learning Based Sample

1. Learning to Sample

   ![image.png](https://s2.loli.net/2023/03/17/gefhZtmNdRJxyOs.png)

   - 设计一个降采样网络S-NET
   - 对降采样前后的点云执行特定任务（例如目标识别）
   - 比较降采样前后的识别结果（Task Loss）
   - 根据这两者的差别调整S-NET，直到达到两者近似

### (4). 上采样

#### i. Bilateral Filter

1. 高斯滤波在模糊图像的同时也会抹除边缘信息

   ![image.png](https://s2.loli.net/2023/03/17/7fK3vhXqVDxobYB.png)

2. 双边滤波在模糊细节的同时会保留大致的边缘信息

   ![image.png](https://s2.loli.net/2023/03/17/e5TYGBWhaNuv471.png)

   ![image.png](https://s2.loli.net/2023/03/17/ALjwCy79W4bZ1Dx.png)

   对P点的像素强度$I_p$，遍历其邻域$S$内的点$q$

   - 距离项$G_{\sigma_s}(\norm{p-q})$表示两像素的距离对$I_p$的影响，越远（即$\norm{p-q}$越大），影响越小
   - 强度项$G_{\sigma_r}(I_p-I_q)$表示两像素的强度差对$I_p$的影响，相差越大，影响越小
   - W为归一化项

3. 在深度图中应用：

   ![image.png](https://s2.loli.net/2023/03/17/NX2y9B58coISvtO.png)

4. 在多传感器融合中，将点云深度信息融合到图像上后，图像上的深度信息是稀疏的，对其运用双边滤波，可以有效的将深度信息稠密的表达在全图上，并且保证一定的准确率

---

# III. Nearest Neighbor Problem

## 1. Binary Search Tree (用于解决一维数据的NN Problem)

### (1). K-NN and Raius-NN

![image.png](https://s2.loli.net/2023/03/20/g9G1v8eMDL62kEm.png)

### (2).  二叉树基本数据结构

1. 一个Node最多有两个Child Node
2. 一个Node左边的Child Node一定比自身小，右边的Child Node一定比自身大
3. eg. <img src="https://s2.loli.net/2023/03/20/uWvqtwlRLo8G6Kn.png" alt="image.png" style="zoom:50%;" />

### (3). 代码构建

#### i. 类：结点

<img src="https://s2.loli.net/2023/03/20/Q2zLFqE4Nx6H7my.png" alt="image.png" style="zoom:50%;" />

其中：

- slef.left存放left child node；
- self.right存放right chlid node；
- self.key存放当前node的值；
- self.value存放当前node在原lst中的索引；

#### ii. 函数：建树（需要先找到Median作为Root Node，这样建出的树最balance）

<img src="https://s2.loli.net/2023/03/20/JzyA8MXRgWVfE63.png" alt="image.png" style="zoom:60%;" />

<img src="https://s2.loli.net/2023/03/20/6E7p5VrhciaFToW.png" alt="image.png" style="zoom:60%;" />

#### iii. 函数：搜索（给定一个key值，查找树中是否有key值等于此值的Node）

<img src="https://s2.loli.net/2023/03/20/W7snB2Qapkr4t1Z.png" alt="image.png" style="zoom:67%;" />

#### iv. 函数：遍历树

<img src="https://s2.loli.net/2023/03/20/9TZPQwUStqyvo2R.png" alt="image.png" style="zoom:67%;" />

其中：

- inorder的遍历方法会将所有Nodes按Key值从小到大排序
- preorder用于复制Tree
- postorder用于删除结点

### (4). 用二叉树解决1-NN问题

![image.png](https://s2.loli.net/2023/03/20/yIKUXmE2i9Alrdc.png)

1. 找到Node 8，维护worst distance=11-8=3；
2. 因为11>8，所以访问Node 8的Right Child；
3. 找到Node 10，因为|11-10|<worst distance=3，所以维护worst distance=11-10=1；
4. 因为11>10，所以访问Node 10的Right Child；
5. 找到Node 14， 因为|11-14|>worst distance=1，所以worst distance不变；
6. 因为11<14，所以访问Node 14的Left Child；
7. 找到Node 13，因为|11-13|>worst distance=1，所以worst distance不变；
8. 因为11<13，所以访问Node 13的Left Child；
9. 因为Node 13的Left Child=None，完成查找；

### (5). 用二叉树解决k-NN问题

<img src="https://s2.loli.net/2023/03/20/N7VwgkJ8i95qA3U.png" alt="image.png" style="zoom:80%;" />

1. 初始化一个容器（K维的向量，每个值都是inf）（且这个容器总是从小到大排序的）
2. 按1-NN中的方法搜索树（worst distance总是这个向量最末尾的值）：
   1. 当Container未满时，将每一个遍历到的点都加入容器
   2. 当Container已满时，每遍历到一个点，根据其与目标点的距离，维护该容器

![image.png](https://s2.loli.net/2023/03/20/53biAEjwCGl1JXM.png)

![image.png](https://s2.loli.net/2023/03/20/om1jASft4eawqNZ.png)

### (6). 用二叉树解决Radius-NN问题

令Worst Distance=Radius即可

## 2. KD-Tree (K-Demensions Search Tree)

基本思路是在每个维度上进行BST分割

leaf_size参数：分割到最后每个cell内Points的个数小于leaf_size则不再分割

### (1). 切割策略

- 每个维度交替切
- 根据点的分布特征自适应的切

![image.png](https://s2.loli.net/2023/03/20/7frsB3v8bIqLAtH.png)

### (2). 切割过程（建树过程）

| 切割结果                                                     | 建树结果                                                     | 注释                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image.png](https://s2.loli.net/2023/03/20/TSZWNitzsLydGjn.png) |                                                              |                                                              |
| ![image.png](https://s2.loli.net/2023/03/20/yT96Uq7Eo5Yfj4p.png) | ![image.png](https://s2.loli.net/2023/03/20/WleLd4BU89AYgvK.png) | Node 1:<br />n1.points_indexes={a,b,c,d,e,f,g,h,i}<br />n1.is_leaf=False<br />n1.axis=x<br />n1.value=$x_{s_1}$<br />n1.left=Node whose points_indexes is {a,b,d,e,g}($x<x_{s_1}$的点的集合)<br />n1.reight=Node whose points_indexes is {i,h,f,c}($x>x_{s_1}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/SDKxhWYfv7aglUp.png) | ![image.png](https://s2.loli.net/2023/03/20/Rm6PiZhOBJKuFg7.png) | Node 2:<br />n2.points_indexes={a,b,d,e,g}<br />n2.is_leaf=False<br />n2.axis=y<br />n2.value=$y_{s_2}$<br />n2.left=Node whose points_indexes is {a,b}($y<y_{s_2}$的点的集合)<br />n2.right=Node whose points_indexes is {d,e,g}($y>y_{s_2}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/INckotvbJCuKPDf.png) | ![image.png](https://s2.loli.net/2023/03/20/y46sTRoBenJcUkI.png) | Node 3:<br />n3.points_indexes={a,b}<br />n3.is_leaf=False<br />n3.axis=x<br />n3.value=$x_{s_3}$<br />n3.left=Node whose points_indexes is {a}<br />n3.right=Node whose points_indexes is {b} |
| ![image.png](https://s2.loli.net/2023/03/20/t6zAXewl3pFGMWo.png) | ![image.png](https://s2.loli.net/2023/03/20/YSjQKLUbJV17deP.png) | Node 4<br />n4.points_indexes={a}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/SWJFcauUmQEdhzC.png) | ![image.png](https://s2.loli.net/2023/03/20/qwYJtiMnE7FSfXd.png) | Node 5<br />n5.points_indexes={b}<br />n5.is_leaf=True<br />n5.axis=None<br />n5.value=None<br />n5.left=None<br />n5.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/EFnmf3RS4UL9eZs.png) | ![image.png](https://s2.loli.net/2023/03/20/ChRk9N6SbJFqEaK.png) | Node 6<br />n6.points_indexes={d,e,g}<br />n6.is_leaf=False<br />n6.axis=y<br />n6.value=$y_{s_4}$<br />n6.left=Node whose points_indexes is {d,e}($y<y_{s_4}$的点的集合)<br />n6.right=Node whose points_indexes is {g}($y>y_{s_4}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/zpTSk5HCchyjKRD.png) | ![image.png](https://s2.loli.net/2023/03/20/YfBN4W7ahDXnkIp.png) | Node 7<br />n7.points_indexes={d,e}<br />n7.is_leaf=False<br />n7.axis=x<br />n7.value=$x_{s_5}$<br />n7.left=Node whose points_indexes is {d}($x<x_{s_5}$的点的集合)<br />n7.right=Node whose points_indexes is {e}($x>x_{s_5}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/OL2waCfmj1JtiyI.png) | ![image.png](https://s2.loli.net/2023/03/20/lhYEpa1BT6vycbe.png) | Node 8<br />n4.points_indexes={d}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/oBkuWfGvtRCe92X.png) | ![image.png](https://s2.loli.net/2023/03/20/UMVyZCESiG68bOd.png) | Node 9<br />n4.points_indexes={e}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/WjqzeuTdSO9BJKI.png) | ![image.png](https://s2.loli.net/2023/03/20/YVzA3xDoRG46JHQ.png) | Node 10<br />n4.points_indexes={a}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/kOph6LKcXNM7Sda.png) | ![image.png](https://s2.loli.net/2023/03/20/jv31PUGFMalmn4u.png) | Node 11<br />n11.points_indexes={i,h,f,c}<br />n11.is_leaf=False<br />n11.axis=y<br />n11.value=$y_{s_6}$<br />n11.left=Node whose points_indexes is {c,f}($y<y_{s_6}$的点的集合)<br />n11.right=Node whose points_indexes is {i,h}($y>y_{s_6}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/J4fGxlyLcI1mbod.png) | ![image.png](https://s2.loli.net/2023/03/20/clQLVtm7SrXY5Mv.png) | Node 12<br />n12.points_indexes={c,f}<br />n12.is_leaf=False<br />n12.axis=x<br />n12.value=$x_{s_7}$<br />n12.left=Node whose points_indexes is {c}($x<x_{s_7}$的点的集合)<br />n12.right=Node whose points_indexes is {f}($x>x_{s_7}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/AOaQLWJenqIpfXS.png) | ![image.png](https://s2.loli.net/2023/03/20/EN3BR6HTsYDXLKA.png) | Node 13<br />n4.points_indexes={c}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/AjQKIuHPVUDqokw.png) | ![image.png](https://s2.loli.net/2023/03/20/tFDYUxnV8gATBWf.png) | Node 14<br />n4.points_indexes={f}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/4DHc5Xxb9jRorqv.png) | ![image.png](https://s2.loli.net/2023/03/20/XfETrJjPpDtncbd.png) | Node 15<br />n15.points_indexes={i,h}<br />n15.is_leaf=False<br />n15.axis=y<br />n15.value=$y_{s_8}$<br />n15.left=Node whose points_indexes is {c,f}($y<y_{s_8}$的点的集合)<br />n15.right=Node whose points_indexes is {i,h}($y>y_{s_8}$的点的集合)<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/CybXFLjB7spQqvV.png) | ![image.png](https://s2.loli.net/2023/03/20/ZFkoUs9Xd5InpmP.png) | Node 16<br />n4.points_indexes={h}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/CIhkGVfu56qeyBZ.png) | ![image.png](https://s2.loli.net/2023/03/20/71tJfknVC3DxSLX.png) | Node 17<br />n4.points_indexes={i}<br />n4.is_leaf=True<br />n4.axis=None<br />n4.value=None<br />n4.left=None<br />n4.right=None<br /> |

### (3). 代码构建

#### i. 结点的表达

![image.png](https://s2.loli.net/2023/03/20/RePG8EoaSzBAF1D.png)

#### ii. 函数：建树

![image.png](https://s2.loli.net/2023/03/20/okcAL7SsMvpq19i.png)

每次递归开始时：

1. 先判断node.points（从上一层递归处能够得到points list）的个数是否大于leaf size
   1. 如果大于，继续下一步
   2. 如果小于，return当前node
2. 如果大于，说明需要分割，则选定一个维度（利用axis_round_robin函数进行轮转），将points在这个维度上进行排序，找到中间值，从onset到median传给left node，从median到end传给right node
   - 在应用中可以用（取平均值+逐一和平均值作比较）的方法来分割left和right，这样可以取消排序，加快速度

### (4). 用KD-Tree进行k-NN查找

关键：维护一个worst distance（与BST一样，维护一个K维向量，其末位维护的就是worst distance（动态变化的））

如何判断是否查找一个区域（是否访问一个node）？（对每个维度而言）

![image.png](https://s2.loli.net/2023/03/20/6ZPJ92VjuopNTmS.png)

1. 如果query node在这个区域内，要查找
2. 如果query node在这个维度上的距离与这个node的分割value之差小于worst distance，要查找

![image.png](https://s2.loli.net/2023/03/20/3LjPQk8bsMIytKh.png)

### (5). 用KD-Tree进行Radius-NN查找

令Worst Distance=Radius即可

---

## 3. Octree 

### (1). 构建（以四叉树为例）

| 构建过程 (leaf_size=1)                                       | 建图过程                                                     | 注释                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image.png](https://s2.loli.net/2023/03/20/fN8IY4cPBOVFKqh.png) | ![image.png](https://s2.loli.net/2023/03/20/r1eSOlBWVTzIiRG.png) | 构建一个最大的cube，包围所有points（各个维度的bounding）     |
| ![image.png](https://s2.loli.net/2023/03/20/YbQDIkGzXhfHR1v.png) | ![image.png](https://s2.loli.net/2023/03/20/p1Y3rMZNTCjbRLd.png) | 四等分上一个cube，将点划分到所属的subcube中，判断每个subcube中的点的个数<br />=0则置空<br />=leaf_size则置为node<br />>leaf_size则继续划分<br /> |
| ![image.png](https://s2.loli.net/2023/03/20/NFEDO2RVzvsrSYy.png) | ![](https://s2.loli.net/2023/03/20/j3sgyf7LcVJ5lpC.png)      |                                                              |
| ![image.png](https://s2.loli.net/2023/03/20/w3PJOTcGvyHDm2q.png) | ![image.png](https://s2.loli.net/2023/03/20/KEROZ4hfFkTCsJg.png) |                                                              |

终止条件：

- leaf_size
- min_extent：cube的最小边长，在实际中可能出现多个点坐标一样的情况，这样的话如果用leaf_size将永远无法完成建树

### (2). 代码构建

#### i. Octant的表达

![image.png](https://s2.loli.net/2023/03/20/QnNMbtkvoVh9s7X.png)

#### ii. 建树

![image.png](https://s2.loli.net/2023/03/20/xV6tE1G2YLXuDjn.png)

### (3). 用Octree进行k-NN查找

#### i. 主体函数

关键：维护一个worst distance（与BST一样，维护一个K维向量，其末位维护的就是worst distance（动态变化的））

![image.png](https://s2.loli.net/2023/03/20/CTseO1KGvxqM7Rg.png)

#### ii. 辅助函数1：判断是否可以提前终止搜索（如果一个octant能把以worst_distance为半径的球完全包围）

![image.png](https://s2.loli.net/2023/03/20/E326bXZkTy5MjOm.png)

在各个维度上都满足：蓝色(Extent)>绿色(query-actant.center)+红色(radius即worst_distance)

#### ii. 辅助函数2：判断一个octant是否和以worst_distance为半径的球有重叠

![image.png](https://s2.loli.net/2023/03/21/znT58p4sL7Eqvf6.png)

1. 在任何维度上，绿色>蓝色+红色，则必不会重叠

   <img src="C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20230321094354915.png" alt="image-20230321094354915" style="zoom:50%;" />

2. 不满足1.的情况下，在至少两个维度上，球的中心在octant的bounding内（蓝色>红色）

   ![image.png](https://s2.loli.net/2023/03/21/ewNA8oL1jisyluO.png)

3. 在不满足1.的情况下，球与角点的距离（绿色-蓝色）<半径，（在某个维度上若绿色<蓝色，将其置为0），因此此情况可以囊括对棱和角点重叠的检测

   <img src="https://s2.loli.net/2023/03/21/iJXgeOhUxjaEL1s.png" alt="image.png" style="zoom:50%;" />

   

### (4). 用Octree进行Radius-NN查找

#### i. 固定worst_distance

#### ii. 加速策略：函数：检测包围

如果一个octant被query ball完全包围，那么这个octant不用再进行细分，将其中的点全部纳入解集即可

![image.png](https://s2.loli.net/2023/03/21/ERHx1o6GuqCeXmy.png)

---

# IV. Clustering

## 1. 常见的聚类算法及效果

![image.png](https://s2.loli.net/2023/03/31/bIzDyPlfBKW1Chi.png)

![image.png](https://s2.loli.net/2023/04/03/CXlbSarNz9jOL8u.png)

---

## 2. 基础数学知识

### (1). Joint Probability

联合概率：两个事件同时发生的概率

![image.png](https://s2.loli.net/2023/03/31/7ZyGQubS6AeBaD1.png)

---

### (2). Marginalization

边缘分布：两个事件各自的概率分布

![image.png](https://s2.loli.net/2023/03/31/vPHfmDeZrK1wXV9.png)

![image.png](https://s2.loli.net/2023/03/31/u8ePoThVsJpYdRH.png)

$p(x),p(y)$依旧是合法的概率分布（即总和为1）

---

### (3). Conditional Probability

![image.png](https://s2.loli.net/2023/03/31/8kuc2P13Cd57pgb.png)

直接取出的切片$p(x,Y=y^*)$不是合法的概率分布，需要进行归一化得到概率分布$P(x|Y=y^*)$

---

### (4). Bayes' Probability Theorem

![image.png](https://s2.loli.net/2023/03/31/o6uqlLCyE2t5knD.png)

![image.png](https://s2.loli.net/2023/03/31/1hn82Zj73KPNArw.png)

---

### (5). Directed Graphical Model (DGM)

![image.png](https://s2.loli.net/2023/03/31/qReHVXLUhAxdOg5.png)

有向图用于表示有关联的两个事件的条件概率；相对应的无连接的两个节点相互独立，他们的联合概率即为两者各自概率的乘积；

扩展：Markov Assumption：当一个结点给定了它的父结点时，它只与自己的子节点有关，与其他节点都相互独立，如下图：

<img src="https://s2.loli.net/2023/03/31/tq8JoXAfeypFmna.png" alt="image.png" style="zoom:50%;" />

---

### (6). Undirected Graphical Model (UGM)

![image.png](https://s2.loli.net/2023/03/31/Amnt1frYgLEF8hi.png)

---

### (7). Lagrange Multiplier

问题：求二元函数z=f(x,y)在条件φ(x,y)=0下的极值

解法：

<img src="https://s2.loli.net/2023/03/31/toKxXOw3mN6cfsJ.png" alt="image.png" style="zoom:150%;" />

直观理解：

![image.png](https://s2.loli.net/2023/03/31/EtBN8askuKJF5vL.png)

沿着g(x,y)=c曲线找f(x,y)的最大值，该点的特征是在该点处f(x,y)和g(x,y)相切（梯度方向相同）

---

## 3. K-Means

### (1). 基本步骤

1. 随机初始化K个中心点
2. 将每个数据点分配到一个K中心点下（Expectation）
3. 将K更新为每个类的坐标均值中心（Maximization）
4. 重复2，3

---

### (2). 数学解释

<img src="https://s2.loli.net/2023/03/31/YfNrbGghcSltWdL.png" alt="image.png" style="zoom:50%;" />

$r_{nk}$是布尔矩阵$R_{NK}$的第$(n,k)$个元素，代表第n个点是否属于第k类

<img src="https://s2.loli.net/2023/03/31/M5rsEnq2tvNfRgX.png" alt="image.png" style="zoom:50%;" />

- Expectation Step是固定中心$\mu_k$去解$R_{NK}$
- Maximization Step是固定分类$R_{NK}$去优化中心$\mu_k$

---

#### i. E Step

对每一个点$x_n$，找到距离最近的一个中心$\mu_k$，得到它的k，将$r_{nk}$置为1，将$r_{n:}$置为0

#### ii. M Step

当$r_{nk}$固定后，J是一个只与$\mu_k$有关的函数，要求它的最小值，即令一阶导为0即可；

对每个$\mu_k$单独考虑：
$$
for\ k\ in\ K:\\2\sum^N_{n=1}r_{nk}(x_N-\mu_k)=0\\\mu_k=\frac{\sum^nr_{nk}x_n}{\sum^Nr_{nk}}
$$
**一个类的中心点=这个类的所有点的平均值**

#### iii. 终止条件

1. $r_nk$停止变化
2. 迭代次数限制



![image.png](https://s2.loli.net/2023/03/31/EX5xuN1wjMabHKI.png)

---

### (3). 进阶技巧

![image.png](https://s2.loli.net/2023/03/31/NpY8uPaKyiheDqk.png)

---

### (4). K-Medoids

![image.png](https://s2.loli.net/2023/03/31/ovKztAismYBd4E1.png)

$\Upsilon(x,\mu)$是一个复杂的离散函数，实现该算法的具体步骤是：

- E步骤不变
- M步骤由求平均值改为：遍历当前类中的所有数据点，计算它到该类中其他所有点的距离之和，选择最小的一个

---

### (5). Limitations 局限性

1. K不知道
2. 对噪声很敏感
3. hardmax分类，缺乏置信度（解决：GMM）

---

## 4. Gaussian Mixture Model (GMM)

### (1). 基本思想

将每一个类用一个高斯分布$N(\mu,\sigma^2)$来描述，因此能够给出每一个点属于其类的可能性（置信度）

![image.png](https://s2.loli.net/2023/03/31/JN47ADRCdcSnsYB.png)

---

### (2). Gaussion Distribution

![image.png](https://s2.loli.net/2023/03/31/ZhsOXK61iDdB9te.png)

---

### (3). GMM数学表征

![image.png](https://s2.loli.net/2023/03/31/9V7J6KMZQrfvIEt.png)

1. k需要给定
2. 给定k后$p(x)$就是k个高斯分布的线性组合
3. 对于每个数据点x，都有一个$\pi_k(1\times k)$，其中每个元素表示这个数据点属于第k个高斯分布的概率
4. 假设每一类的数据点都是由一个高斯分布采样得到的，我们已知数据点去反推这个分布的过程叫做**极大似然估计**

---

#### i. 先验概率

![image.png](https://s2.loli.net/2023/03/31/lfNViJpUIQ1P95O.png)

对于一个数据点，我不知道它的具体位置，这是你要我说它属于每个类别的概率分别是多少，我只能根据这个k个分布的特点来推断，这个概率叫做先验概率

---

#### ii. 后验概率

我们要求的是后验概率，即给定点x的条件下，它属于每个类别的概率分别是多少，即$p(z|x)$

<img src="https://s2.loli.net/2023/03/31/OTGixU1RMPZjSQe.png" alt="image.png" style="zoom:50%;" />

---

#### iii. p(z)

![image.png](https://s2.loli.net/2023/03/31/kApZufK3elWDVmw.png)

- z是一个k维的布尔向量，表示点属于某个分布的事件，因此这个向量中只有一个元素为1，其他为0
- $\pi$是一个k维的向量，表示点属于某个分布的概率，因此有$\sum^K_{k=1}\pi_k=1$
- 当z待定的时候，$p(z)$就可以写成上述的连乘形式（一般形式），当z一旦确定，他就会坍缩成②式

---

#### iv. p(x|z)

![image.png](https://s2.loli.net/2023/03/31/cCVPNuDo7UFdGix.png)

1. $p(x|z_k=_0)$：表示我已经确定点x属于某个高斯分布，这是的概率p(x|z)就是该高斯分布的概率分布
2. 因此，在z还没有确定的时候p(x|z)可以写成连成形式，z一旦确定，它就会坍缩成上式

---

#### v. p(x)

根据边缘分布公式：$p(x)=\sum_z p(x,z)$

根据条件概率公式：$p(x|z)=\frac{p(x,z)}{p(z)}$，因此有$p(x,z)=p(x|z)p(z)$

因此有：$p(x)=\sum_zp(z)p(x|z)$

![image.png](https://s2.loli.net/2023/03/31/ndED2yYJmeLa9q4.png)

---

#### vi. p(z|x)

![image.png](https://s2.loli.net/2023/03/31/9oyb4YvzVqrHC26.png)

---

#### vii. 高斯模型的参数估计

以上推断基于高斯模型的参数$\mu,\sigma$已知，但在实际应用中能拿到的只有数据点和k，$\mu_k,\sigma_k$需要估计：**极大似然估计（MLE）**

即，要最大化损失函数：

![image.png](https://s2.loli.net/2023/03/31/F89WEncrsOfojZb.png)

但是存在一个问题，在应用中有可能出现一个数据点单独成一类的情况，一个数据点的高斯分布标准差为0，这会导致概率趋近于无穷大，在应用中有以下几种解决办法：

1. 如果某一个高斯分布的标准差很小，就把它随机初始化成一个数值
2. 采用Maximun-a-Posterior或Bayesian Approach代替MLE

---

### (4). 求解GMM

1. 初始化：
   1. k个类的中心坐标$\mu_k$
   2. k个类在各个方向上的方差$\Sigma_k$
   3. 权重$\pi_k$

2. E-Step：估算后验概率（N*K的矩阵，每一个点对每一个类都有一个值）

   1. 分子：权重\*点在某个高斯分布下的值
   2. 分母：点在每个高斯分布下的值的加权求和

   <img src="https://s2.loli.net/2023/03/31/cizhwsJPRXMU9IV.png" alt="image.png" style="zoom:50%;" />

3. M-Step：根据后验概率反过来优化参数

   1. 目标：最大化损失函数

      <img src="https://s2.loli.net/2023/03/31/lb6ScXjipR1kgUK.png" alt="image.png" style="zoom:50%;" />

   2. 方法：loop每个类，固定两个参数，对另一个求导=0

      <img src="https://s2.loli.net/2023/03/31/eqC5sYvWfJlcX9h.png" alt="image.png" style="zoom:50%;" />

      - $\mu_k$:
        - 分母：第k个类的点数目
        - 分子：用第k类的N个后验概率对点坐标加权求和
      - $\sigma_k$：
        - 分母：第k个类的点数目
        - 分子：用第k类的N个后验概率对方差加权求和
      - $\pi_k$：
        - 分母：所有点的个数
        - 分子：第k个类的点数目



<img src="https://s2.loli.net/2023/03/31/y3XqThGkug7Z1VW.png" alt="image.png" style="zoom:50%;" />

---

## 5. Spectral Clustering

上述两种聚类方法都工作在欧氏空间中；

谱聚类工作在Graph中，看重点与点之间的连接性；

### (1). 无向图

![image.png](https://s2.loli.net/2023/03/31/ROYDLb3f8xP6zhH.png)

- 无向图可以用一个$n\times n$的矩阵W表示（n为点的个数），其中$w_{ij}$表示点i和点j之间边的权重
- 权重定义为点与点之间的相似度（距离越近，相似度越大）
- 未连接的点之间的边权重为0
- 点与自身不连接（$w_{ii}=0$ ，即对角线为0）

---

### (2). 建图

![image.png](https://s2.loli.net/2023/03/31/5eWdYGqX3tVcvFr.png)

三种策略：

1. Radius Neighbors：将每个点的一定半径内的点进行连接
2. KNN Neighbors：将每个点和它的K个邻居进行连接
   1. “or” KNN
   2. “and” KNN
3. Fully Connected：全连接

对于$w_{ij}$的具体表示，通常采用高斯核函数RBF：$exp({-\frac{\norm{x_i-x_j}^2_2}{2\sigma^2}})$

---

### (3). 数据准备

#### i. Degree Matrix

$$
D=diag(d_1,d_2,\dots,d_n)\ where\ d_{ii}=\sum^n_{j=1}w_{ij}
$$

#### ii. Unnormalized Laplacian Matrix

$$
L=D-W
$$

#### iii. Normalized Laplacian Matrix

$$
L_{sym}=I-D^{-\frac12}WD^{-\frac12}\\
L_{rw}=I-D^{-1}W
$$

---

### (4). 算法实施步骤

1. 建立adjacency matrix W (n x n)
2. 计算Laplacian Matrix（unnormalized or normalized）
3. 计算L矩阵的特征值和特征向量
4. 取特征值最小的k个特征向量组成特征向量矩阵V (n x k)
5. 将V的每一行看作一个点，共n个点进行kmeans聚类

---

### (5). 算法分析

#### i. L矩阵归一化与未归一化的区别

![image.png](https://s2.loli.net/2023/03/31/Oxr6yCdTJYZK2Uo.png)

#### ii. 自动找类别数的能力

可以将L矩阵的特征值全部列出，如果发现断崖式上升，则取前k个特征值较小的类即可

![image.png](https://s2.loli.net/2023/03/31/ouUaN1f6ZPgSEjt.png)

---

### (6). 原理分析

最小图割问题

---

## 6. Mean Shift

### (1). 基本思想

给定一些点，给定一个半径固定的圆，求解一个圆的位置，使得其中的点的数量最多

解法：

1. 随机初始化圆
2. 计算圆内部所有点的坐标均值
3. 将圆移到均值位置
4. 重复2,3直到圆的位置不再移动

---

### (2). 实现步骤

1. 随机初始化m个圆（parameter: m,r）
2. 将这m个圆分别移动到各自内部点的坐标均值位置
3. 重复2直到这m个圆的位置都不再移动
4. 对于有重叠部分的圆，进行NMS，保留内部点个数最多的一个
5. 直到所有圆都不重叠
6. 此时圆的个数就是聚类数k（具有自己发现有多少个类的能力）
7. 对于每个数据点，将其分配给离他最近的聚类中心

| ![1680488332443.png](https://s2.loli.net/2023/04/03/OmlCiUGRJQNawXM.png) | ![image.png](https://s2.loli.net/2023/04/03/cHlIAVJi5wxzhgq.png) | ![image.png](https://s2.loli.net/2023/04/03/6MZJEa9RBXfrlbv.png) | ![image.png](https://s2.loli.net/2023/04/03/atASHL4xdrWnE6P.png) | ![image.png](https://s2.loli.net/2023/04/03/MJiKHtEOaUrNz8x.png) | ![image.png](https://s2.loli.net/2023/04/03/yODTAzkxNfdUChe.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

---

## 7. DBSCAN

### (1). 基本原理

漫水算法

---

### (2). 实现步骤

![image.png](https://s2.loli.net/2023/04/03/gxZU68HISeFvm5T.png)

1. 给所有点附加一个bool flag位，初始化为False（unvisited）
2. 设置两个参数r, min_num
3. 从unvisited的点中随机选择一个访问（访问后即将其置为visited）
4. 审阅其半径r内部的所有邻居，统计邻居的个数num：
   1. 如果num>min_num，则称其为core_point，创建一个cluster，将p的所有邻居都归为该cluster，go to 5.
   2. 如果num<min_num，将p置为noise，从data中删除
5. 逐一访问它的的所有邻居neighbor(i)，统计每一个点r范围内的邻居个数
   1. 如果num(i)>min_num，则neighbor(i)也是core_point，将其的所有邻居都归为该cluster，go to 5.
   2. 如果num(i)<min_num，则neighbor(i)称为border_point，不做任何操作（其自身在上一步已经被归为该cluster，这里与core_point的区别是不将它的邻居再归为该cluster）
6. 直到没有新的点能加入该cluster，将该cluster从data中删除
7. go to 3. 直到所有点都被标记过，算法结束

![image.png](https://s2.loli.net/2023/04/03/c4sjpfXYxGBb3g8.png)

---

# V. Model Fitting

## 1. 应用

根据形状的先验信息，对聚类的结果进行形状拟合来将其分离或补全

- 自动驾驶：地面，路沿的识别
- 工业应用：机械臂抓取（物体形状固定（零部件））

---

## 2. Least Square

功能：你和一条空间直线（无论点的维度有多少）

### (1). 基本步骤

1. 根据点的维度建立空间直线方程（二维空间直线：$ax+by+c=0$）
2. 定义损失函数$E=\sum^n_{i=1}(ax_i+by_i+c)^2$
3. 问题建模：求解$\hat{x}=[\hat{a},\hat{b},\hat{c}]^T=\arg \mathop{\min}\limits_{x}\norm{A_{n\times3}x_{3\times1}}^2_2$，其中$\norm{x}^2_2=1$
4. 解法：
   1. 构建A: ![image.png](https://s2.loli.net/2023/04/03/23ZqwxWfVaLjcE5.png)
   2. 求A的特征值和特征向量，$\hat{x}$就是特征值最小的特征向量

![image.png](https://s2.loli.net/2023/04/03/du4wWPjF6kMfLaE.png)

---

### (2). 算法优化

以L2 norm为损失函数的LSQ比较易受outlier的影响，可以用以下损失函数代替

![image.png](https://s2.loli.net/2023/04/03/JCmtYBgoMz2U5cr.png)

其中表现最好的就是Cauchy Loss Function

---

## 3. Hough Transform

### (1). 基本思想

数据空间和参数空间的转换（拟合不同形状时有不同的映射关系，以拟合二维直线为例）：

- 原始空间的一个点→参数空间的一条线
- 原始空间的一条线→参数空间的一个点

![image.png](https://s2.loli.net/2023/04/03/azBQEbhYkU15snf.png)

当数据空间的很多点参与拟合时，会在参数空间生成很多线，这些线不一定都精确的交于一点，这是通过vote机制（需要先将参数空间离散化以便于遍历）选择最优的一个参数值

---

### (2). 基本步骤

1. 用自变量和参数建立要拟合形状的模型（直线：$xcos\theta+ysin\theta=r$ {$\theta,r$}；圆：$(x-a)^2+(y-b)^2=r^2$ {$a,b,r$}）
2. 为每个参数选择一个分辨率，将参数空间离散
3. 根据每个数据点对离散的参数空间进行投票
4. 对离散的参数空间进行高斯滤波
5. 选择vote值最大的一个参数

---

### (3). 霍夫圆检测

| ![image.png](https://s2.loli.net/2023/04/03/Hd54ajb3kESGwU7.png) | ![image.png](https://s2.loli.net/2023/04/03/RzlihF5nWX6ucbt.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

每一个点在三维参数空间中都是一个圆锥

---

### (4). 分析

![image.png](https://s2.loli.net/2023/04/03/NbzOnMiqTLQhuFK.png)

参数空间一般小于3，否则很慢

---

## 4. RANSAC (Random Sample Consensus)

### (1). 实现步骤

1. 随机选择一个sample（sample的意思是确定一个模型需要的最少的点组，例如确定一个直线最少需要两个点）

2. 求解基于此sample的模型

   ![image.png](https://s2.loli.net/2023/04/03/sCNpkyMe7tKv2WV.png)

3. 计算每个点到该模型的距离

   ![image.png](https://s2.loli.net/2023/04/03/nJHpN3wAxKCaRoT.png)

4. 根据threshold参数$\tau$将点划分为inlier和outlier

   ![image.png](https://s2.loli.net/2023/04/03/oJbftcYAFIu17Nd.png)

5. 统计inlier在整个数据中的占比：

   1. 如果大于期望的占比（1-e），则提前退出算法
   2. 如果小于，则未达到效果，go to 1. 重新取样

---

### (2). 关于参数

1. 距离阈值$\tau$根据实验选定

2. 先验的，期望的，预估的outlier点的占比：$e$

3. 每次随机取样的点的个数：$s$

4. 置信度：$p$（经过N次迭代后至少能找到一个good sample的概率，default=0.99）

5. 最大迭代次数N，根据e，s，p可解：

   ![image.png](https://s2.loli.net/2023/04/03/KwRs2Hm9goqe1IW.png)

   ![image.png](https://s2.loli.net/2023/04/03/jVI8acPOGU62pQR.png)

---

### (3). 分析

![image.png](https://s2.loli.net/2023/04/03/dp8hYUwubO29IrV.png)





