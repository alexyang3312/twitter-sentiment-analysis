# 小组讨论
* 需要把州名为空的数据过滤掉；

已知问题：
* pandas可能会有性能问题，默认没办法并行，是否需要使用dataframe，写正则？
* `是否需要使用多分区？需要！`
* 对训练好的模型进行评估+性能测试+优化模型？

Ian:
* So the important thing is to talk about how you approached doing the classification in a big-data way that makes sense, not just we did some random forests but how did you specifically exploit big-data techniques to make this faster. It's not just a machine learning coursework, although you need to talk about that you also need to talk about how you exploited big data techniques to make this more efficient. So basically, you need to describe how you used spark to parallelise your data - you should show how you successfully split your dataset into multiple parts and trained the classifier that way, as opposed to just running it normally on a single node.
* 没有标准测量情感值的准确度（除非手动标记每个推特的情感值，当然这不可能），我们要评估的是the accuracy of prediciting TextBlob labels（这不是问题），我们假设Textblob的准确度很高，我们只需要去复制TextBlob的结果。
* You need to label every single instance in your datasets regardless of whether it's going to be used for training or testing. The way the testing works is that you take the entries in your test data, run your classifiers on it and then check to see whether your classifier returns the same result as what TextBlob returned, whether your method gives the same label as what TextBlob gave.
* `70%的数据用于训练，30%的数据用于测试。不需要数据用来应用，本作业只需要训练和测试即可，报告测试结果等；`
* 作业的重点是如何应用大数据技术去解决问题，不要花太多的时间去搞数据可视化，可以用于锦上添花；
* 关于local model：举例子，例如只把local model在经济方面的推特调教好，高度拟合（不是过拟合），那这个local model只适用于经济方面的分析，其他方面推特的话这个model就歇菜了。global model会拟合所有训练集数据，把性能平均给数据。local model不会尝试去拟合所有的数据，只关注于数据的一小部分，
因为对于某个特定的实例来说可能大量数据是完全【不相关的】，那就没有必要做拟合，我们只需要拟合【相关的】数据即可。可以搜classification local model，详情可以访问这里https://www.tandfonline.com/doi/abs/10.1198/0003130031423?casa_token=bNOVUPIwxgcAAAAA%3Ahepruxxl3lT7zTolJkgkITOBVlwpi8lnEyWQaI5k8uyWbsAGTkrtmQx0MSUCKeLRfQST-P0yIBRQ&
* `数据预处理之前需要做数据清洗，修改或删除数据，变成需要的方式，细节可以发给Ian再看一眼。`
* 提升分类器的准确度取决于你选择的features，以及参数的调教，可以调整树的数量以及深度，再去对比结果，看是否有效。

---
# 参考资料
## 向量
Spark MLlib中提供的机器学习模型处理的是向量形式的数据，因此我们需将文本转换为向量形式，
这里我们利用Spark提供的Word2Vec功能结合其提供的text8文件中的一部分单词进行了word2vec模型的预训练，
并将模型保存至word2vecM_simple文件夹中，因此本次实验中将tweets转换为向量时直接调用此模型即可。

可以使用text8自行训练词向量转换模型，或线上搜索利用tweets进行分词训练的word2vec模型。
* https://www.cnblogs.com/tina-smile/p/5204619.html
* https://blog.51cto.com/u_15127586/2670975
* https://blog.csdn.net/chuchus/article/details/71330882
* https://blog.csdn.net/chuchus/article/details/77145579


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

