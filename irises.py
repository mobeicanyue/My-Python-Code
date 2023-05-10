import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from KMeans import k_means

iris = load_iris()  # 加载鸢尾花数据集
samples = iris.data[:, 2:]  # 表示我们只取特征空间中的后两个维度

clusterCents, sampleTag, = k_means(samples, 3)  # 调用KMeans算法
plt.scatter(clusterCents[:, 0].tolist(), clusterCents[:, 1].tolist(), c='r', marker='^')  # 绘制簇中心
plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))  # 绘制样本点
plt.show()  # 显示图像
print(clusterCents)  # 打印簇中心
