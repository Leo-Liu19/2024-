import pandas as pd
from snownlp import SnowNLP
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

# 读取停用词文件
with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# 读取CSV文件
df = pd.read_csv('allData.csv')

# 初始化一个Counter对象来统计词频
word_counts = Counter()

cnt = 0
# 遍历每条评论
for comment in df['评论']:
    # 检查评论是否为空或非字符串
    if not isinstance(comment, str) or comment == '':
        continue

    # 使用正则表达式去除标点符号
    comment = re.sub(r'[^\w\s]', '', comment)

    # 检查去除标点后的评论是否为空
    if comment.strip() == '':
        continue

    s = SnowNLP(comment)

    # 过滤停用词并更新词频统计
    filtered_words = [word for word in s.words if word not in stopwords]
    word_counts.update(filtered_words)
    print('第{}条评论处理完毕'.format(cnt))
    cnt += 1

# 加载蒙版图像
mask = np.array(Image.open('./image/bow.png'))

# 筛选词频大于或等于40的词
filtered_word_counts = {word: count for word, count in word_counts.items() if count >= 0}

# 生成词云对象
wordcloud = WordCloud(font_path='C:\Windows\Fonts\simhei.ttf', width=800, height=400, background_color='white', mask=mask).generate_from_frequencies(filtered_word_counts)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存词云图
wordcloud.to_file('./image/wordcloud.png')

# 输出词频
sorted_word_counts = sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_word_counts:
    print(word, count)
