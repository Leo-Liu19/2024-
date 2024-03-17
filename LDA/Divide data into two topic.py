import pandas as pd
from snownlp import SnowNLP
import re

# 读取CSV文件
df = pd.read_csv('allData.csv')  # 替换为你的文件路径

# 读取停用词
with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:  # 确保文件名和路径正确
    stopwords = [line.strip() for line in f]
    stopwords = set(stopwords)  # 转换为集合以提高查找效率

# 文本预处理函数
def preprocess_text(text):
    # 去除空白
    text = text.strip()
    # 去除特殊字符
    text = re.sub(r'http\S+', '', text)  # 去除网址
    text = re.sub(r'\W', ' ', text)  # 去除非字母数字字符
    # 分词并去除停用词
    words = text.split()  # 分词
    words = [word for word in words if word not in stopwords]  # 去除停用词
    return ' '.join(words)  # 重新组合为字符串

# 应用文本预处理
df['评论'] = df['评论'].apply(preprocess_text)

# 检查并删除空评论
df = df[df['评论'].str.len() > 0]

# 使用SnowNLP计算情感得分
df['score'] = df['评论'].apply(lambda x: SnowNLP(x).sentiments)

# 分割数据集
positive_comments = df[df['score'] >= 0.5]
negative_comments = df[df['score'] < 0.5]

# 保存到CSV文件
positive_comments.to_csv('positive_comments.csv', index=False)
negative_comments.to_csv('negative_comments.csv', index=False)
