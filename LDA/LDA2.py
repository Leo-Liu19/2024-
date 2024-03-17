import multiprocessing

import pandas as pd
from gensim import corpora, models
import jieba
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import re

def main():
    # 读取停用词
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)

    # 读取CSV文件
    df = pd.read_csv('allData.csv')
    texts = df['评论'].tolist()  # 假设你的文本数据在'text_column_name'列中


    # 分词、去除停用词和标点符号
    def clean_text(text):
        text = re.sub(r"[^\w\s]", "", text)
        words = jieba.cut(text)
        return [word for word in words if word not in stopwords and not re.match(r"^\s*$", word)]

    texts_processed = [clean_text(text) for text in texts]
    # 创建词袋模型
    dictionary = corpora.Dictionary(texts_processed)
    corpus = [dictionary.doc2bow(text) for text in texts_processed]

    P = []
    C = []


    # 训练LDA模型
    ldamodel = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

    # 打印主题
    print(ldamodel.print_topics())


    x_values = range(2, 2 + len(P))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()