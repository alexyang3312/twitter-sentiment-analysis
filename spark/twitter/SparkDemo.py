from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf = conf)

from pyspark.sql import SparkSession
sparkSession = SparkSession.builder.master("local[*]").appName("pySparkSQL example").getOrCreate()

# ------试验区开始------
print(123)
print(123)
print(123)

import json
import re
import string
import numpy as np

from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

#寻找推文的协调性
#符号化推文的文本
#删除停用词，标点符号，url等
remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation))  # regex to remove special characters
stopwords = [u'rt', u're', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
             u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
             u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
             u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
             u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a',
             u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by',
             u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after',
             u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under',
             u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
             u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not',
             u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don',
             u'should', u'now']

# tokenize函数对tweets内容进行分词
def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore').decode('ascii')  # 去掉乱码符号
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                  text)  # to replace url with ''
    text = remove_spl_char_regex.sub(" ", text)  # Remove special characters
    text = text.lower()

    for word in text.split():
        if word not in stopwords \
                and word not in string.punctuation \
                and len(word) > 1 \
                and word != '``':
            tokens.append(word)
    return tokens

# 读入预先训练好的文本向量化模型word2vecM
# sqlContext = SQLContext(sc)
# lookup = sqlContext.read.parquet("/Users/yangminghan/Desktop/大数据推特美国大选/word2vecM_simple2/data").alias("lookup")
parquetFile = r"./twitter/word2vecM_simple/data/part-r-00000-5d5e245e-1d0c-4281-9274-145495a3565f.snappy.parquet"
lookup = sparkSession.read.parquet(parquetFile).alias("lookup")
lookup.printSchema()
lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

# 定义分词文本转换为向量的函数
def doc2vec(document):
    # 100维的向量
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
        # 查找该词在预训练的word2vec模型中的特征值
            vec = np.array(lookup_bd.value.get(word)) + 1
            # print(vec)
            # 若该特征词在预先训练好的模型中，则添加到向量中
            if vec is not None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec

# 读入tweets.json作为分类器训练数据集
with open('tweets.json', 'r') as f:
    rawTrn_data = json.load(f)
    f.close()

trn_data = []
for obj in rawTrn_data['results']:
    token_text = tokenize(obj['text']) # 规范化推特文本，进行分词
    tweet_text = doc2vec(token_text) # 将文本转换为向量
    # 使用LabeledPoint 将文本对应的情感属性polariy：该条训练数据的标记label，tweet_text：训练分类器的features特征，结合成可作为spark mllib分类训练的数据类型
    trn_data.append(LabeledPoint(obj['polarity'], tweet_text))

trnData = sc.parallelize(trn_data)
# print(trn_data)
# print(trnData)
print("------------------------------------------------------")

# 读入hillary.json作为分类器测试数据集
with open('hillary.json', 'r') as f:
    rawTst_data = json.load(f)
    f.close()

tst_data = []
for obj in rawTst_data['results']:
    token_text = tokenize(obj['text'])
    tweet_text = doc2vec(token_text)
    tst_data.append(LabeledPoint(obj['polarity'], tweet_text))

tst_dataRDD = sc.parallelize(tst_data)

# 训练随机森林分类器模型
print(trnData.count())

model = RandomForest.trainClassifier(trnData, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=6, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=8, maxBins=32)
print(model.numTrees())
# 6
print(model.totalNumNodes())
# 1106

# 利用训练好的模型进行模型性能测试
predictions = model.predict(tst_dataRDD.map(lambda x: x.features))
labelsAndPredictions = tst_dataRDD.map(lambda lp: lp.label).zip(predictions)
# 计算分类错误率
print(labelsAndPredictions)
testErr = labelsAndPredictions.filter(lambda v: v[0] != v[1]).count() / float(tst_dataRDD.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
# 输出训练好的随机森林的分类决策模型
print(model.toDebugString())

print(456)
print(456)
print(456)
# ------试验区结束------

# 关闭与Spark的连接
sc.stop()

# 提交 ./bin/spark-submit
