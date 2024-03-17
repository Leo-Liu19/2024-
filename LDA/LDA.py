# 本代码用以测试LDA模型的效果，选取最好的主题数，一般来说自由选定主题数，直接运行LDA2文件即可。
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
    df = pd.read_csv('negative_comments.csv')
    texts = df['评论'].tolist()  # 假设你的文本数据在'text_column_name'列中

    # 文本预处理
    #texts_processed = [[word for word in jieba.cut(text) if word not in stopwords] for text in texts
    #texts_processed = [[word for word in jieba.cut(text) if word not in stopwords] for text in texts]

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

    for i in range(2, 6):
        # 训练LDA模型
        ldamodel = models.LdaModel(corpus, num_topics=i, id2word=dictionary, passes=15)

        # 打印主题
        print(ldamodel.print_topics())

        # 计算困惑度
        perplexity = ldamodel.log_perplexity(corpus)
        P.append(perplexity)

        # 计算一致性得分
        coherence_model = CoherenceModel(model=ldamodel, texts=texts_processed, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        C.append(coherence_score)

        print(f'Perplexity: {perplexity}')
        print(f'Coherence Score: {coherence_score}')

    x_values = range(2, 2 + len(P))

    # 绘制第一幅图
    plt.figure()
    plt.plot(x_values, P)
    #plt.title('')
    #plt.xlabel('')
    plt.ylabel('Perplexity')

    # 绘制第二幅图
    plt.figure()
    plt.plot(x_values, C)
    #plt.title('Line 2')
    #plt.xlabel('X axis starting from 2')
    plt.ylabel('Coherence_score')

    # 显示图表
    plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()