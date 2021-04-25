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
print('string.punctuation=',string.punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

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
    # print('-----')
    # print(text)
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
            # print(type(vec))
            # 若该特征词在预先训练好的模型中，则添加到向量中
            if vec is not None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    # print('--------------')
    # print(doc_vec)
    # print(tot_words)
    vec = doc_vec / float(tot_words)
    return vec

import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
#读入数据csv数据
def readCSV(csvName):
    # df = pd.read_table(csvName, sep=',', header='infer', lineterminator="\n")
    df = pd.read_csv(csvName, lineterminator="\n")
    # df = sparkSession.read.format('csv').option("header", 'true').load(csvName)
    # df = sparkSession.read.format('csv').option("header", 'true').load("csvName")# .option("multiLine", "true")

    # 因为要escape大量的特殊字符或写正则，所以暂时不用这种方法遍历数据
    # df = sparkSession.read.csv("hashtag_donaldtrump_small.csv", header=True, multiLine=True)
    # df = sparkSession.read.csv("hashtag_donaldtrump_small.csv")
    # df = sparkSession.read.option("multiline", "true").option("quote", '"').option("header", "true").option("escape", "\\").option("escape", '"').csv('hashtag_donaldtrump_small.csv')

    # print(type(df))
    # df.cache()
    # df.show(10)
    # df.printSchema()

    # 删除没有用的列，只用3个元素：推特内容，洲名（地点，用于可视化），情感值（需要第三方语义分析工具转换）
    # df = df.drop('created_at', 'tweet_id', 'likes', 'retweet_count',
    #         'source', 'user_id', 'user_name', 'user_screen', 'user_screen_name',
    #         'user_description', 'user_join_date', 'user_followers_count',
    #         'user_location', 'lat', 'long', 'city', 'country',
    #         'continent', 'state_code', 'collected_at')

    # train, test = df.randomSplit([0.7, 0.3])
    # print(train.rdd.getNumPartitions())
    # print(train.rdd.getNumPartitions())
    # train_rdd = train.rdd.repartition(4)
    # test_rdd = test.rdd.repartition(2)
    # column_names = df.columns
    #
    # def toPandas_partition(instances):
    #     panda_df = pd.DataFrame(columns=column_names)  # using the global variable
    #
    #     for instance in instances:  # each instance is of Row type
    #         panda_df = panda_df.append(instance.asDict(), ignore_index=True)
    #
    #     return [panda_df]
    #
    # rdd_pandas = train_rdd.mapPartitions(toPandas_partition)

    # df = df.drop('created_at').drop('tweet_id').drop('likes').drop('retweet_count').drop('source').\
    #     drop('user_id').drop('user_name').drop('user_screen').drop('user_screen_name').drop('user_description').\
    #     drop('user_join_date').drop('user_followers_count').drop('user_location').drop('lat').drop('long').\
    #     drop('city').drop('country').drop('continent').drop('state_code').drop('collected_at')

    # 遍历数据，做数据预处理
    trn_data = []

    for index, row in df.iterrows():
        # print(index,row['tweet'],row['state'])
        # 语义分析，转换成感情值，详情参考https://textblob.readthedocs.io/en/dev
        # print(TextBlob(row['tweet']).sentiment.polarity)

        if (isinstance(row['tweet'], str) == False):
            continue

        token_text = tokenize(row['tweet'])  # 规范化推特文本，进行分词
        tweet_text = doc2vec(token_text)  # 将文本转换为向量
        # 使用LabeledPoint 将文本对应的情感属性polariy：该条训练数据的标记label，tweet_text：训练分类器的features特征，结合成可作为spark mllib分类训练的数据类型
        tempPolarity = TextBlob(row['tweet']).sentiment.polarity

        if tempPolarity > 0:
            # 积极
            tempPolarity = 0
        if tempPolarity == 0:
            # 无感, str
            tempPolarity = 1
        if tempPolarity < 0:
            # 消极
            tempPolarity = 2

        # LabeledPoint(label, features)
        # Labels should take values {0, 1, ..., numClasses-1}.
        trn_data.append(LabeledPoint(tempPolarity, tweet_text))

    trnData = sc.parallelize(trn_data)
    # print(trn_data)
    # print(trnData)
    print(trnData.count())
    return trnData

# 训练随机森林分类器模型
# train_dataRDD = readCSV('hashtag_donaldtrump_small.csv')
# train_dataRDD = readCSV('hashtag_donaldtrump_medium.csv')
train_dataRDD = readCSV('hashtag_donaldtrump.csv')
train_dataRDD.cache()
model = RandomForest.trainClassifier(train_dataRDD, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=6, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=8, maxBins=32)

print(model.numTrees())
# 6
print(model.totalNumNodes())
# 1106

# 利用训练好的模型进行模型性能测试
# tst_dataRDD = readCSV('hashtag_joebiden_small.csv')
# tst_dataRDD = readCSV('hashtag_joebiden_medium.csv')
tst_dataRDD = readCSV('hashtag_joebiden.csv')
tst_dataRDD.cache()

# 计算正确率
from sklearn.metrics import accuracy_score
textblobData = tst_dataRDD.map(lambda lp: lp.label).collect()
# 使用TextBlob转化出来的labels
predictionData = model.predict(tst_dataRDD.map(lambda x: x.features)).collect()
# 预测好的labels，通过features来预测labels
print('accuracy rate', accuracy_score(textblobData, predictionData))
# 检查使用TextBlob转化出来的labels与使用模型预测出来的labels是否相等

# print('Learned classification tree model:')
# 输出训练好的随机森林模型
# print(model.toDebugString())
print("------------------------------------------------------")
# 地图相关demo代码：

#读入数据csv数据，地理信息数据
def readCSV2(csvName):
    df = pd.read_table(csvName, sep=',', header='infer', lineterminator="\n")
    # 删除没有用的列，只用3个元素：推特内容，洲名（地点，用于可视化），情感值（需要第三方语义分析工具转换）
    # df = df.drop('created_at', 'tweet_id', 'likes', 'retweet_count',
    #         'source', 'user_id', 'user_name', 'user_screen', 'user_screen_name',
    #         'user_description', 'user_join_date', 'user_followers_count',
    #         'user_location', 'lat', 'long', 'city', 'country',
    #         'continent', 'state_code', 'collected_at')

    # 因为要escape大量的特殊字符或写正则，所以暂时不用这种方法遍历数据
    # df = sparkSession.read.format('csv').option("header", 'true').load("hashtag_donaldtrump_small.csv")# .option("multiLine", "true")
    # df = sparkSession.read.csv("hashtag_donaldtrump_small.csv", header=True, multiLine=True)
    # df = sparkSession.read.csv("hashtag_donaldtrump_small.csv")
    # df = sparkSession.read.option("multiline", "true").option("quote", '"').option("header", "true").option("escape", "\\").option("escape", '"').csv('hashtag_donaldtrump_small.csv')

    # print(type(df))
    # df.cache()
    # df.show(10)
    # df.printSchema()

    # print(type(df))
    # 遍历数据，做数据预处理
    trn_data = []

    for index, row in df.iterrows():
        # print(index,row['tweet'],row['state'])
        # 语义分析，转换成感情值，详情参考https://textblob.readthedocs.io/en/dev
        # print(TextBlob(row['tweet']).sentiment.polarity)

        if (type(row['tweet']) != 'str'):
            continue

        token_text = tokenize(row['tweet'])  # 规范化推特文本，进行分词
        tweet_text = doc2vec(token_text)  # 将文本转换为向量
        # 使用LabeledPoint 将文本对应的情感属性polariy：该条训练数据的标记label，tweet_text：训练分类器的features特征，结合成可作为spark mllib分类训练的数据类型
        state = row['state']
        tempPolarity = TextBlob(row['tweet']).sentiment.polarity

        if tempPolarity > 0:
            # 积极
            tempPolarity = 0
        if tempPolarity == 0:
            # 无感
            tempPolarity = 1
        if tempPolarity < 0:
            # 消极
            tempPolarity = 2

        # LabeledPoint(label, features)
        # Labels should take values {0, 1, ..., numClasses-1}.
        dict = {'polarity': tempPolarity, 'state': state}
        trn_data.append(dict)

    trnData = sc.parallelize(trn_data)
    # print(trn_data)
    # print(trnData)
    print(trnData.count())
    return trnData

# popdensity_blank = {
#     'New Jersey':  0., 'Rhode Island': 0., 'Massachusetts': 0., 'Connecticut': 0.,
#     'Maryland': 0.,'New York': 0., 'Delaware': 0., 'Florida': 0., 'Ohio': 0., 'Pennsylvania': 0.,
#     'Illinois': 0., 'California': 0., 'Hawaii': 0., 'Virginia': 0., 'Michigan':    0.,
#     'Indiana': 0., 'North Carolina': 0., 'Georgia': 0., 'Tennessee': 0., 'New Hampshire': 0.,
#     'South Carolina': 0., 'Louisiana': 0., 'Kentucky': 0., 'Wisconsin': 0., 'Washington': 0.,
#     'Alabama':  0., 'Missouri': 0., 'Texas': 0., 'West Virginia': 0., 'Vermont': 0.,
#     'Minnesota':  0., 'Mississippi': 0., 'Iowa': 0., 'Arkansas': 0., 'Oklahoma': 0.,
#     'Arizona': 0., 'Colorado': 0., 'Maine': 0., 'Oregon': 0., 'Kansas': 0., 'Utah': 0.,
#     'Nebraska': 0., 'Nevada': 0., 'Idaho': 0., 'New Mexico':  0., 'South Dakota':    0.,
#     'North Dakota': 0., 'Montana': 0., 'Wyoming': 0., 'Alaska': 0.}
# popdensity_blank2=popdensity_blank.copy()

# predictionsArray = predictions.collect()

# i = 0
# tst_dataRDD = readCSV2('hashtag_joebiden_small.csv')
# for j in tst_dataRDD.collect():
#     try:
#         state = j['state']
#         popdensity_blank[state] += (j['polarity'] - 1)
#         popdensity_blank2[state] += (predictionsArray[i] - 1)
#         i += 1
#     except:
#         continue

# for x in popdensity_blank:
#     print(x)
#
# for y in popdensity_blank2:
#     print(y)

print(456)
print(456)
print(456)
# ------试验区结束------

# 关闭与Spark的连接
sc.stop()

# 提交 ./bin/spark-submit
