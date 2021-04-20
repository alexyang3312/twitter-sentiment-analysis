# 小组讨论

相关资料：
* https://www.cnblogs.com/mrchige/p/6346601.html
* https://www.jianshu.com/p/310ef75e150d
* https://blog.csdn.net/qq_41853758/article/details/82934506
* https://blog.csdn.net/wustjk124/article/details/81320995
* https://blog.csdn.net/zyp199301/article/details/71727278
* https://my.oschina.net/u/4347889/blog/3346852
* https://blog.csdn.net/p_function/article/details/77713611

2021年04月15日会议纪要：
1. 特征上可以把like等加进来；
2. 大家需要读别人找到的文献；
3. 需要看朴素和随机是不是有相同的数据预处理和相同的数据展示；
4. 情感值的描述写在introduction；
5. 文献需要引用的话得先发到群里周知大家；
6. 每周二开周会，23号第一个组内deadline；

周知：
* `大部分代码源自网络，进行了一部分修改，提交到老师前需要一系列修改并做查重，不要抄袭，引用他人代码需要标注；`
* `大数据的小组项目没有宽限期；`
* 真正跑数据的时候，可以用老师新发的邮件里的平台，之前都是跑小规模数据；
* 小组项目不需要使用HDFS，直接从硬盘读取文件即可；

已知问题：
* pandas可能会有性能问题，默认没办法并行，是否需要使用dataframe，写正则？
* 读取数据是否需要使用Python多线程？
* 是否需要使用多分区？
* 对训练好的模型进行评估+性能测试+优化模型？

问老师问题：
* How do we determine the number of test sets and training sets? We only have 2 CSV files, which are hashtag_donaldtrump.csv (971157 rows of data) and hashtag_joebiden.csv (777078 rows of data). 
* How can we improve the accuracy of the Random Forest model?
* We found Spark DataFrame cannot read the CSV normally when some strange strings showed up.
* 豪胜的一些问题（自己）.

please always include a 2 min overview of your current progress.
论文标题叫什么？ Twitter sentiment analysis of the US election
数据预处理是否要抽出来，2个算法公用相同的预处理？
如何使用交叉引用去优化随机森林模型？
结尾段需要对比2个算法的优劣，然后做出总结。
I would probably suggest a Local model (but you still don't know what that means).

论文要求：
* `4月26日前要提交，4个人写一篇论文即可；`
* 论文需要15-25个引用，更多说明在文档上都有；
* 相当于是整个项目的草稿draft，不计分，同学+老师也会参与评价，不要弄得太好，你懂的，自己给自己留一些已知的空间，要将同学的反馈考虑到自己的项目中；
* 论文结构有例子，论文需要引用，Turnitin查重；
* 总结出来问题问老师或助教；
* 大数据的作业最多写8页，是算上图的，图也会占用空间，5-8页就可以，不要在论文中引入过多的表格和图片；
* `第一次提交不用包含代码，只需要写论文，不要包含作者名字，不要透露作者是谁，因为后面有双盲；`

人员/计划安排：
* 论文整体把握-明翰（相关owner写自己的部分）
* 随机森林-明翰
* 朴素贝叶斯-志鹏（比随机森林更简单+适合，有参考资料）
* 数据可视化-雄风（工作量减小，有参考代码，是否需要可视化要问老师？）
* Word2Vec-豪胜
* KNN-豪胜
* 语义转换包-?

依赖关系：
* 志鹏确定一下预处理部分是否能摘出来吗？
* 豪胜确定离线模型是否能搞出来？
* 明翰的输出结果能跟雄峰的对上吗？


Spark MLlib中提供的机器学习模型处理的是向量形式的数据，因此我们需将文本转换为向量形式，
这里我们利用Spark提供的Word2Vec功能结合其提供的text8文件中的一部分单词进行了word2vec模型的预训练，
并将模型保存至word2vecM_simple文件夹中，因此本次实验中将tweets转换为向量时直接调用此模型即可。

可以使用text8自行训练词向量转换模型，或线上搜索利用tweets进行分词训练的word2vec模型。
* https://www.cnblogs.com/tina-smile/p/5204619.html
* https://blog.51cto.com/u_15127586/2670975
* https://blog.csdn.net/chuchus/article/details/71330882
* https://blog.csdn.net/chuchus/article/details/77145579

---

# 推特美国大选情感分析
## 开头段
在当今这个DT（Data Technology）时代，社交媒体已成为深受互联网用户欢迎的沟通工具，如Twitter、Facebook、Instagram等等。
每天都有大量的数据产生，人们每天在移动设备或PC上记录生活，分享或讨论对不同话题的看法，
随着越来越多的用户讨论或表达自己的观点，推特已经成为人们评论与情感信息的宝贵来源。

对于网上海量分布地数据，可以被用来做情感分析，挖掘各种观点，
情感分析可应用于众多领域，通过分析公众的评论，情感分析可以帮助政府评估自身的优势和不足。
例如，"特朗普让美国再次衰落"，这条评论清楚地表达了用户对政府的负面情绪，
情感分析有助于了解公众情绪是如何影响选举结果的。

我们利用推特上的美国大选相关信息数据结合Spark MLlib实现人们对美国这两位总统候选人的情感分析，
通过情感分析的结果，可以直观地看到在美国不同地方的网民们的看法，
应用不同的机器学习算法去理解是否有推特与真实结果是否有关联。

我们主要使用2个算法来做情感分析：随机森林，朴素贝叶斯。
（需要介绍一下随机森林与朴素贝叶斯）

流程：
1. 获取推特数据
2. 数据预处理
3. 数据分析（训练，测试）
4. 评估+优化模型？可以先不写？
5. 应用数据，结果可视化（通过Python的地图可视化工具Basemap）

## 随机森林实现方式
分别从Kaggle网站上获取关于特朗普与拜登的相关推特原始CSV数据，
里面的数据项很多，但被用到的却只有推特内容以及地理信息。
把数据分成2训练集与测试集并分别进行预处理，
在数据预处理过程中，文本数据需要背分词且过滤，
通过Python自然语言处理包TextBlob可以将推特文本内容转化成情感属性值polarity（积极，中立，消极）。

（需要改写+引用）
TextBlob is a Python (2 and 3) library for processing textual data. 
It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, 
noun phrase extraction, sentiment analysis, classification, translation, and more.

此外，还需要使用一个离线模型对文本进行特征提取以及向量化，
之后建立模型并对模型进行训练与测试，评估+优化模型？可以先不写？
应用模型于数据，将将分析结果汇总并可视化，可以清晰的看到推特数据与真实结果的差异性，
推特上的地理信息可以在做数据可视化的时候将数据直观地呈现在美国地图上。

## 朴素贝叶斯实现方式
* http://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes
* https://developer.ibm.com/alert-zh
* https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2
* http://introtopython.org/visualization_earthquakes.html#Adding-detail

## 结尾段
除了CSV文件外，也可以通过推特API获取流式数据，
使用TwitterAPI可以很轻松地收集到数百万条推文用于算法训练。
通过Spark Stream进行情感分析，我们使用了Spark MLlib的随机森林与朴素贝叶斯进行情感分析，
已经使用了BaseMap进行可视化，通过可视化结果，我们可以直观的感受2个候选人在美国各个州的受欢迎程度。

（需要对比一下2种模型的不同与优劣，最后下结论）
情感分析还适用于其他场景，例如房价、物价、交通等等，在大数据中扮演十分重要的角色。

