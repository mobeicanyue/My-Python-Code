import matplotlib.pyplot as plt
import numpy as np


def euclid_dist(vecXi, vecXj):
    """
    计算欧氏距离
    para vecXi：点坐标，向量
    para vecXj：点坐标，向量
    return: 两点之间的欧氏距离
    """
    return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))


def k_means(dataSet, k, distMeas=euclid_dist):
    """
    K均值聚类
    para dataSet：样本集，多维数组
    para k：簇个数
    para distMeas：距离度量函数，默认为欧氏距离计算函数
    return sampleTag：一维数组，存储样本对应的簇标记
    return clusterCents：一维数组，各簇中心
    """
    m = np.shape(dataSet)[0]  # 样本总数
    sampleTag = np.zeros(m)

    # 随机产生k个初始簇中心
    n = np.shape(dataSet)[1]  # 样本向量的特征数
    clusterCents = np.mat(np.zeros((k, n)))  # 初始化簇中心
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        clusterCents[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))

    sampleTagChanged = True

    while sampleTagChanged:  # 如果没有点发生分配结果改变，则结束
        sampleTagChanged = False

        # 计算每个样本点到各簇中心的距离
        for i in range(m):
            minD = np.inf
            minIndex = -1
            for j in range(k):
                d = distMeas(clusterCents[j, :], dataSet[i, :])
                if d < minD:
                    minD = d
                    minIndex = j
            if sampleTag[i] != minIndex:
                sampleTagChanged = True
            sampleTag[i] = minIndex

        print(clusterCents)
        plt.scatter(clusterCents[:, 0].tolist(), clusterCents[:, 1].tolist(), c='r', marker='^', linewidths=7)
        plt.scatter(dataSet[:, 0], dataSet[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
        plt.show()

        # 重新计算簇中心
        for i in range(k):
            ClustI = dataSet[np.nonzero(sampleTag[:] == i)[0]]
            clusterCents[i, :] = np.mean(ClustI, axis=0)
    return clusterCents, sampleTag  # 返回簇中心和样本点对应的簇标记

# if __name__ == '__main__':
#     iris = load_iris()
#     X = iris.data[:, 2:]  # 表示我们只取特征空间中的后两个维度
#
#     samples = X
#     clusterCents, sampleTag, = kMeans(samples, 3)
#     plt.scatter(clusterCents[:, 0].tolist(), clusterCents[:, 1].tolist(), c='r', marker='^')
#     plt.scatter(samples[:, 0], samples[:, 1], c=sampleTag, linewidths=np.power(sampleTag + 0.5, 2))
#     plt.show()
#     print(clusterCents)
