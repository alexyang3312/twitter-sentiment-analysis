# 小组讨论

2021年04月23日会议纪要：
* 需要朴素贝叶斯的概念以及相关引用，图片等；
* 决定是否要用KNN，如果需要则需要文案；

2021年04月15日会议纪要：
* 特征上可以把like等加进来；
* 大家需要读别人找到的文献；
* 需要看朴素和随机是不是有相同的数据预处理和相同的数据展示；
* 文献需要引用的话得先发到群里周知大家；
* 每周二开周会，23号第一个组内deadline；

周知：
* `大部分代码源自网络，进行了一部分修改，提交到老师前需要一系列修改并做查重，不要抄袭，引用他人代码需要标注；`
* `大数据的小组项目没有宽限期；`
* 真正跑数据的时候，可以用老师新发的邮件里的平台，之前都是跑小规模数据；
* 小组项目不需要使用HDFS，直接从硬盘读取文件即可；

已知问题：
* pandas可能会有性能问题，默认没办法并行，是否需要使用dataframe，写正则？
* 读取数据是否需要使用Python多线程？
* `是否需要使用多分区？需要！`
* 对训练好的模型进行评估+性能测试+优化模型？

问老师问题：
* How do we determine the number of test sets and training sets? We only have 2 CSV files, which are hashtag_donaldtrump.csv (971157 rows of data) and hashtag_joebiden.csv (777078 rows of data). 
* How can we improve the accuracy of the Random Forest model?
* We found Spark DataFrame cannot read the CSV normally when some strange strings showed up.
* 豪胜的一些问题（自己）.

please always include a 2 min overview of your current progress.
数据预处理是否要抽出来，2个算法公用相同的预处理？
如何使用交叉引用去优化随机森林模型？
结尾段需要对比2个算法的优劣，然后做出总结。

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

Ian:
* So the important thing is to talk about how you approached doing the classification in a big-data way that makes sense, not just we did some random forests but how did you specifically exploit big-data techniques to make this faster. It's not just a machine learning coursework, although you need to talk about that you also need to talk about how you exploited big data techniques to make this more efficient. So basically, you need to describe how you used spark to parallelise your data - you should show how you successfully split your dataset into multiple parts and trained the classifier that way, as opposed to just running it normally on a single node.
* 没有标准测量情感值的准确度（除非手动标记每个推特的情感值，当然这不可能），我们要评估的是the accuracy of prediciting TextBlob labels（这不是问题），我们假设Textblob的准确度很高，我们只需要去复制TextBlob的结果。
* You need to label every single instance in your datasets regardless of whether it's going to be used for training or testing. The way the testing works is that you take the entries in your test data, run your classifiers on it and then check to see whether your classifier returns the same result as what TextBlob returned whether your method gives the same label as what TextBlob gave.
* `70%的数据用于训练，30%的数据用于测试。不需要数据用来应用，本作业只需要训练和测试即可，报告测试结果等；`
* 作业的重点是如何应用大数据技术去解决问题，不要花太多的时间去搞数据可视化，可以用于锦上添花；
* 关于local model：举例子，例如只把local model在经济方面的推特调教好，高度拟合（不是过拟合），那这个local model只适用于经济方面的分析，其他方面推特的话这个model就歇菜了。global model会拟合所有训练集数据，把性能平均给数据。local model不会尝试去拟合所有的数据，只关注于数据的一小部分，
因为对于某个特定的实例来说可能大量数据是完全【不相关的】，那就没有必要做拟合，我们只需要拟合【相关的】数据即可。可以搜classification local model，详情可以访问这里https://www.tandfonline.com/doi/abs/10.1198/0003130031423?casa_token=bNOVUPIwxgcAAAAA%3Ahepruxxl3lT7zTolJkgkITOBVlwpi8lnEyWQaI5k8uyWbsAGTkrtmQx0MSUCKeLRfQST-P0yIBRQ&
* `数据预处理之前需要做数据清洗，修改或删除数据，变成需要的方式，细节可以发给Ian再看一眼。`
* 提升分类器的准确度取决于你选择的features，以及参数的调教，可以调整树的数量以及深度，再去对比结果，看是否有效。

---
# Twitter sentiment analysis of the US election
## Abstract
最后写

Keywords: TextBlob, Random Forest, Naive Bayes...

## Introduction
In today's DT (Data Technology) age, social media platforms have become a popular communication tools, such as Twitter, Facebook, Instagram and so on. People record their lives and share or discuss their opinions on different topics on mobile devices or PCs every day. With more and more users discussing or expressing their opinions, Twitter has become a valuable source of comments and emotional information.

For the massive distributed data on the Internet, those data could be used to do sentiment analysis in many fields and excavate all kinds of opinions. By analyzing public comments, it could help the government evaluate its own strengths and weaknesses. For example, "Trump has made America weak again", which clearly expresses users' negative feelings towards the government, and sentiment analysis helps to understand how public sentiment affects election results.

American election related information data of Twitter could be combined with the Spark MLlib techniques used for sentiment analysis, through the analysis of the sentiment results, it could visually see people of different places in the United States thought the two presidential candidates, also it could be applied to different machine learning algorithms to understand whether twitter related with real results.

There are two algorithms are used for sentiment analysis: Random Forest and Naive Bayes.
`（豪胜，确定是否需要KNN，如果需要则要引入相关概念，引用，图片等。）`

Random Forest is a flexible ensemble classifier algorithm, which constructs multiple decision trees based on random samples and random features of the training dataset, and the output is determined by the vote. The random reduces the influence of exceptional samples and features on classification results.（引用：2篇文献）

![image](https://user-images.githubusercontent.com/42234021/115829497-f9a48580-a406-11eb-9ba9-2b2f48e6a140.png)
Figure 2 Classification process based on the random forest algorithm A redesign of the original inspired figure found from the following website: https://www.linkedin.com/pulse/random-forest-algorithm-in- teractive-discussion-niraj-kumar/.

`（志鹏，需要朴素贝叶斯的相关概念，引用，图片等。）`

Before training the model, TextBlob is used to convert the content of tweets into emotional values. TextBlob is a text-processing Python (2 and 3) library. It provides a straightforward API for tackling basic natural language processing (NLP) tasks including part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more. （引用：1篇文献）

The aim of this paper is to evaluate which algorithm is more suitable twitter sentiment analysis and the relationship between sentiment analysis result and reality result.
（志鹏，明翰，阐述论文的目的，论文的观点）

There are several sections about the sentiment analysis, the first section is data selection, cleaning and pre-processing, the data should be perpared before the data analysis. The second section is word to vector, an offline model is needed for feature extraction and vectorization of the text. The thrid section is data analysis, the processed data is divided into train dataset and test dataset, and they are used to train or test 2-3 different models, such as Random Forest, Naive Bayes, and KNN(maybe), after that the models are evaluated and optimized respectively. The final section is data visualization, the analysis results is used to draw two Amercian maps by using Basemap which is a third party package for map visualization in Python, and they could show how people feel about Trump and Biden in different states.

## Data selection, cleaning and pre-processing
The raw data were downloaded from Kaggle website, they are two CVS files which are hashtag_donaldtrump.csv (971157 rows of data) and hashtag_joebiden.csv (777078 rows of data). They included loads of tweets about JoeBiden and DonaldTrump.

There are a lot of data columns in the CSV files, but only tweet text messages and geographic information are used.
The data were divided into two training sets and test sets and preprocessed respectively.

In data preprocessing, text data needs to be splitted and filtered,
TextBlob, a Python natural language processing package, converts the content of Twitter text into an emotional attribute value called polarity (positive, neutral, negative).

we need to read messages from those CSV files, and format data. we need to split them, remove useless keywords, URLs and special characters, and we need to use offline model called Word2Vector to transform words into vectors. Then we use TextBlob, which is a Python natural language processing package, converts the content of Twitter text into an emotional attribute value, such as positive, neutral and negative. then we have two important things: vectors and the corresponding sentiment value, which means we have label and features, label and features make up the training data set.

## Word to vector
（豪胜）


## Data analysis by Random Forest implementation
（需要引入文献）



提升分类器的准确度取决于你选择的features，以及参数的调教，可以调整树的数量以及深度，再去对比结果，看是否有效。

## Data analysis by Naive Bayes implementation
（志鹏）

## Data analysis by KNN implementation
（豪胜）

## Data Visualization
（雄峰）
可以清晰的看到推特数据与真实结果的差异性，
推特上的地理信息可以在做数据可视化的时候将数据直观地呈现在美国地图上。


## Conclusion
除了CSV文件外，也可以通过推特API获取流式数据，
使用TwitterAPI可以很轻松地收集到数百万条推文用于算法训练。
通过Spark Stream进行情感分析，我们使用了Spark MLlib的随机森林与朴素贝叶斯进行情感分析，
已经使用了BaseMap进行可视化，通过可视化结果，我们可以直观的感受2个候选人在美国各个州的受欢迎程度。

（明翰，志鹏，需要对比一下2种模型的不同与优劣，最后下结论）
情感分析还适用于其他场景，例如房价、物价、交通等等，在大数据中扮演十分重要的角色。

## References
（会用工具自动生成）

《How Many Trees in a Random Forest?》
Thais Mayumi Oshiro, Pedro Santoro Perez, and Jos´e Augusto Baranauskas Department of Computer Science and Mathematics
Faculty of Philosophy, Sciences and Languages at Ribeirao Preto University of Sao Paulo

《Application of Support Vector Machine, Random Forest, and Genetic Algorithm Optimized Random Forest Models in Groundwater Potential Mapping》
Seyed Amir Naghibi1 & Kourosh Ahmadi2 & Alireza Daneshi3

《Twitter Sentiment Analysis on Coronavirus using Textblob Chhinder》
Chhinder Kaur and Anand Sharma

---
# 参考资料
## 随机森林
* https://www.cnblogs.com/mrchige/p/6346601.html
* https://www.jianshu.com/p/310ef75e150d
* https://blog.csdn.net/qq_41853758/article/details/82934506
* https://blog.csdn.net/wustjk124/article/details/81320995
* https://blog.csdn.net/zyp199301/article/details/71727278
* https://my.oschina.net/u/4347889/blog/3346852
* https://blog.csdn.net/p_function/article/details/77713611

## 朴素贝叶斯
* http://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes
* https://developer.ibm.com/alert-zh
* https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2
* http://introtopython.org/visualization_earthquakes.html#Adding-detail

## 向量
Spark MLlib中提供的机器学习模型处理的是向量形式的数据，因此我们需将文本转换为向量形式，
这里我们利用Spark提供的Word2Vec功能结合其提供的text8文件中的一部分单词进行了word2vec模型的预训练，
并将模型保存至word2vecM_simple文件夹中，因此本次实验中将tweets转换为向量时直接调用此模型即可。

可以使用text8自行训练词向量转换模型，或线上搜索利用tweets进行分词训练的word2vec模型。
* https://www.cnblogs.com/tina-smile/p/5204619.html
* https://blog.51cto.com/u_15127586/2670975
* https://blog.csdn.net/chuchus/article/details/71330882
* https://blog.csdn.net/chuchus/article/details/77145579
