import nltk
import jieba
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt  # 读取文本文件

with open('Luxun.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用jieba进行中文分词
tokens = jieba.cut(text)

# 构建二元语言模型
big_rams = nltk.ngrams(tokens, 2)
frequency = defaultdict(lambda: defaultdict(int))  # 嵌套的 defaultdict
for w1, w2 in big_rams:
    frequency[w1][w2] += 1


def calculate_probability2(sentence):  # 计算概率
    words = jieba.cut(sentence)
    words = list(words)
    probability = 1.0
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        total_count = sum(frequency[w1].values()) + len(frequency[w1])
        probability *= (frequency[w1][w2] + 1) / (total_count + len(frequency[w1]))
    return probability


# 待分析的语句
sentences = [
    "猛兽总是独行，牛羊才成群结队。",
    "从来如此，便对么?",
    "我寄你的信, 总要送往邮局, 不喜欢放在街边的绿色邮筒中, 我总疑心那里会慢一点。",
    "学医救不了中国人。"
]

# lambda表达式
'''
probabilities2 = []
for sentence in sentences:
    probability = calculate_probability2(sentence)
    probabilities2.append(probability)
'''
probabilities2 = [calculate_probability2(sentence) for sentence in sentences]
print(probabilities2)

# 对probabilities取对数再取绝对值
probabilities2 = [abs(np.log(prob)) for prob in probabilities2]

# 可视化对比
plt.figure(figsize=(8, 6))
plt.bar(['s1', 's2', 's3', 's4'], probabilities2)
plt.xlabel('sentence')
plt.ylabel('probability')
plt.title('sentence probability')
plt.show()

# 构建三元语言模型
trigrams = nltk.ngrams(tokens, 3)
frequency = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for w1, w2, w3 in trigrams:
    frequency[w1][w2][w3] += 1


# 计算概率
def calculate_probability3(sentence):
    tokens = jieba.lcut(sentence)
    prob = 1.0
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        prob *= (frequency[w1][w2][w3] + 1) / (sum(frequency[w1][w2].values()) + len(frequency[w1][w2]) + 1)
    return prob


# 计算概率
probabilities3 = [calculate_probability3(sentence) for sentence in sentences]
print(probabilities3)
# 对概率值进行对数变换
probabilities3 = [abs(np.log(prob)) for prob in probabilities3]

# 可视化对比
plt.figure(figsize=(8, 6))
plt.bar(['s1', 's2', 's3', 's4'], probabilities3)
plt.xlabel('sentence')
plt.ylabel('probability')
plt.title('sentence probability trigrams')
plt.show()
