import pandas as pds
from sklearn import tree
from sklearn.model_selection import train_test_split

# 读取csv的数据
# 参数的含义：第一个是文件名, header=None表示没有列名
# names=['a','b','c','d','e']表示列名为a,b,c,d,e
# 可以通过names手动指定、或者生成表头，而文件里面的数据则全部是内容
data = pds.read_csv(r'personal_salary_information.csv', header=None, index_col=False,
                    names=['age', 'units', 'weight', 'degree', 'the education time', 'marriage',
                           'professional', 'family', 'ethnicity', 'gender', 'Income from assets',
                           'loss from assets', 'weekly working hours', 'origin', 'income'])

# 我们选取其中一部分数据处理
data_lite = data[['age', 'units', 'degree', 'gender', 'weekly working hours', 'professional', 'income']]

print("data_lite.head():")
print(data_lite.head())  # 打印前5行数据

# 2.用get_dummies处理数据、分析样本特征
data_dummies = pds.get_dummies(data_lite)
# 使用get_dummies将一个特征变量变为计算机能读懂的特征距离,将文本数据转化为数值

print('打印原始特征:\n', list(data_lite.columns), '\n')
print('打印虚拟变量特征:\n', list(data_dummies.columns), '\n')

print('data_dummies.head():')

print(data_dummies.head(), '\n')  # 显示数据集中的前5行

# 3.特征值定义
features = data_dummies.loc[:, 'age':'professional_ Transport-moving']  # 定义数据集的特征

X = features.values  # 将特征的数值赋值为X

print("features:")
print(features)
print(X)

Y = data_dummies['income_ >50K'].values  # 将收入大于50K作为预测目标
print(data_dummies['income_ >50K'])
print(Y)

print('\n\n')
print('————————————————————————————————')
# 打印数据形态
print(f'特征形态:{X.shape} 标签形态:{Y.shape}')
print('————————————————————————————————')
print('\n\n')

# 4.训练模型
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)  # 将数据集拆分为训练集和测试集
dating_tree = tree.DecisionTreeClassifier(max_depth=5)  # 定义树的深度，可以用来防止过拟合 用最大深度为5的随机森林拟合数据
dating_tree.fit(x_train, y_train)  # 拟合数据

print('结果')
print('————————————————————————————————')
# 打印数据形态
print(f'模型得分:{dating_tree.score(x_test, y_test):.2f}')
print('————————————————————————————————')
print('\n\n')

# 5.使用模型预测
# 将sample的数据进行
sample1 = [
    [37, 40, 0, 0, 0, 0, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
sample2 = [
    [43, 40, 0, 0, 0, 0, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

# 使用模型做出预测
dating_dec1 = dating_tree.predict(sample1)
dating_dec2 = dating_tree.predict(sample2)

print('————————————————————————————————')
if dating_dec1 == 1:
    print("sample1月薪过5万")
else:
    print("sample1月薪不过五万")
print('————————————————————————————————')

if dating_dec2 == 1:
    print("sample2月薪过5万")
else:
    print("sample2月薪不过五万")
print('————————————————————————————————')
