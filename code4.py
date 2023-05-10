import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score  # 准确率评估
from sklearn.metrics import confusion_matrix  # 用混淆矩阵评估
from sklearn.model_selection import train_test_split  # 划分数据

from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯分类算法
from sklearn.naive_bayes import BernoulliNB  # 导入伯努利朴素贝叶斯分类算法
from sklearn.naive_bayes import MultinomialNB  # 导入多项式朴素贝叶斯分类算法
from sklearn.naive_bayes import ComplementNB  # 导入补集朴素贝叶斯分类算法

digits = load_digits()  # 加载手写体数字的数据集
X = digits.data
y = digits.target  # 获取数据集的标签

# 画出手写体数字的图像
# subplots() 既创建了一个包含子图区域的画布，又创建了一个 figure 图形对象
# fig , ax = plt.subplots(nrows, ncols)， nrows 与 ncols 表示两个整数参数，它们指定子图所占的行数、列数。
# ax 是一个包含了所有子图区域的数组，可以通过索引来访问每个子图区域
# figsize 表示图像的宽度与高度，单位为英寸,subplot_kw 表示子图区域的属性,gridspec_kw 表示整个网格的属性
# xticks() 与 yticks() 分别用于设置 x 轴与 y 轴的刻度
fig, axes = plt.subplots(8, 8, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='orange')
plt.show()

"""
X_train	划分出的训练数据集数据     X_test划分出的测试数据集数据
y_train	划分出的训练数据集的标签    y_test划分出的测试数据集的标签
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # 划分数据集

# model = GaussianNB()  # 选择高斯朴素贝叶斯分类算法 0.8333
# model = BernoulliNB() # 选择伯努利朴素贝叶斯分类算法 0.8511
model = MultinomialNB()  # 选择多项式朴素贝叶斯分类算法 0.9088
# model = ComplementNB()  # 选择补集朴素贝叶斯分类算法 0.7911

model.fit(X_train, y_train)  # 训练模型
y_model = model.predict(X_test)  # 预测结果

print(accuracy_score(y_test, y_model))  # 准确率评估 参数：y_true：真实值，y_pred：预测值

mat = confusion_matrix(y_test, y_model)  # 用混淆矩阵评估
sb.heatmap(mat, square=True, annot=True, cbar=False)  # 热力图
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

# 预测后的图形与原始图比较
fig, axes = plt.subplots(8, 8, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
test_images = X_test.reshape(-1, 8, 8)  # 将测试集的数据转换为 8*8 的图像
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes, color='green' if (y_test[i] == y_model[i]) else 'red')
plt.show()
