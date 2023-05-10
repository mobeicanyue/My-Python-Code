import seaborn
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, confusion_matrix


class Model:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # 数据预处理
        self.x_train = self.x_train.reshape(-1, 784) / 255.0
        self.x_test = self.x_test.reshape(-1, 784) / 255.0
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        # 创建模型
        # 使用正则化技术：可以帮助防止过拟合，提高泛化能力。
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.AdamW()
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        # 编译模型
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # 训练模型
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

        self._model = model

    def predict(self, x):
        return self._model.predict(x)


model = Model()

# 预测十张图片
x_test = model.x_test[:10]
y_true1 = tf.argmax(model.y_test[:10], axis=1)
y_pred1 = tf.argmax(model.predict(x_test), axis=1)

# 输出预测结果和真实结果
print('预测结果:', y_pred1.numpy().tolist())
print('真实结果:', y_true1.numpy().tolist())

# 画出图片及其预测结果和真实结果
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axs[i // 5, i % 5]
    ax.imshow(x_test[i].reshape((28, 28)))
    ax.set_title(f"true:{y_true1[i]}, pred:{y_pred1[i]}")
    ax.axis('off')
plt.show()

y_model = model.predict(model.x_test)  # 预测结果
y_pred = tf.argmax(y_model, axis=1)  # 获取预测值 y_model 并在第二个轴（axis=1）沿着每一行找到最高值的索引。
y_true = tf.argmax(model.y_test, axis=1)
print(accuracy_score(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred)  # 用混淆矩阵评估，其中每一行代表真实标签的类别，每一列代表模型预测的类别。
seaborn.heatmap(mat, square=True, annot=True, cbar=False, fmt='d')  # 热力图
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
