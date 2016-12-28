import os
# import sys
# spark_home = "/usr/local/spark"
# os.environ['SPARK_HOME'] = spark_home
# sys.path.insert(0, os.path.join(spark_home, 'python'))
# sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.9-src.zip'))
os.environ["PYSPARK_PYTHON"] = "python3"

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import csv
import numpy as np

conf = SparkConf().setMaster("local").setAppName("SVM")
sc = SparkContext(conf=conf)

# ad = sc.textFile('file:///home/hadoop/Programming/Pycharm/SVM/ad_400_feature.txt')
# young = sc.textFile('file:///home/hadoop/Programming/Pycharm/SVM/mid_400_feature.txt')
#######################################################################################################
#######################################################################################################
ad = sc.textFile('hdfs:///user/hadoop/input/ad_400_feature.txt')
young = sc.textFile('hdfs:///user/hadoop/input/old_400_feature.txt')
#######################################################################################################
#######################################################################################################
# print(data.collect())
# sizeData = data.map(lambda line: len(line.split(" ")))
# stringData = data.map(lambda line: line.split(" "))
# intData = stringData.map(lambda line: np.array([float(x) for x in line]))
# floatData = stringData.map(lambda line: [float(x) for x in line])
# denseData = data.map(lambda line: np.array([float(x) for x in line.split(' ')]))
# print(floatData.collect())

labeledAD = ad.map(lambda line: line.split(" ")).map(lambda line: [float(x) for x in line]).map(lambda line: LabeledPoint(1, line))
labeledYoung = young.map(lambda line: line.split(" ")).map(lambda line: [float(x) for x in line]).map(lambda line: LabeledPoint(0, line))
# print(labeledData.collect())
# print(labeledAD.first().features)
labeledData = labeledAD.union(labeledYoung).cache()

sum_accuracy = 0.0
sum_AreaUnderPR = 0.0
sum_AreaUnderROC = 0.0
#######################################################################################################
#######################################################################################################
# File = open("ad_old.txt", "w")
csvfile = open('ad_old.csv', 'w')
writer = csv.writer(csvfile)
# regParams = [0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,9,30]
#######################################################################################################
#######################################################################################################
for seed in range(5, 1000, 50):
    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")
    svmsplits = labeledData.randomSplit([0.7, 0.3], seed=seed)
    training = svmsplits[0]
    numTraining = training.count()
    test = svmsplits[1]
    numTest = test.count()

    model = SVMWithSGD.train(training, iterations=150, step=1.0, regParam=0.003)

    # 计算SVM的准确率
    svmTotalCorrect = test.map(lambda point: 1 if model.predict(point.features) == point.label else 0).sum()
    svmAccuracy = float(svmTotalCorrect) / numTest
    # print("SVM准确率：%s" % svmAccuracy)

    # 计算PR(准确率-召回率)曲线下面积， 以及ROC(受试者工作特征曲线， 真阳性率-假阳性率)曲线下面积即AUC
    scoreAndLabels = test.map(lambda point: (float(model.predict(point.features)), point.label))

    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUP = metrics.areaUnderPR
    AUR = metrics.areaUnderROC

    # File.write("No.：%s" % seed)
    # File.write("\n")
    # File.write("regParam.：%s" % regParam)
    # File.write("\n")
    # File.write("SVM accuracy：%s" % svmAccuracy)
    # File.write("\n")
    # File.write('Area under PR: %2.4f%%, Area under ROC: %2.4f%%' % (AUP * 100, AUR * 100))
    # File.write("\n")
    seed = "seed：%s" % seed
    writer.writerow([seed])
    accuracy = "SVM accuracy：%s" % svmAccuracy
    PR = 'Area under PR: %2.4f%%' % AUP * 100
    ROC = 'Area under ROC: %2.4f%%' % AUR * 100
    data = [accuracy, PR, ROC]
    writer.writerow(data)

    sum_accuracy += svmAccuracy
    sum_AreaUnderPR += AUP
    sum_AreaUnderROC += AUR
    print("#################################################################################")
    print("#################################################################################")
    print("#################################################################################")

sum_accuracy /= 20
sum_AreaUnderPR /= 20
sum_AreaUnderROC /= 20
# File.write("\n")
# File.write("\n")
# File.write("\n")
# File.write("SVM accuracy：%s" % sum_accuracy)
# File.write("\n")
# File.write('Area under PR: %2.4f%%, Area under ROC: %2.4f%%' % (sum_AreaUnderPR * 100, sum_AreaUnderROC * 100))
# File.write("\n")
writer.writerow(["Average"])
accuracy = "SVM accuracy：%s" % sum_accuracy
PR = 'Area under PR: %2.4f%%' % sum_AreaUnderPR * 100
ROC = 'Area under ROC: %2.4f%%' % sum_AreaUnderPR * 100
data = [accuracy, PR, ROC]
writer.writerow(data)

csvfile.close()