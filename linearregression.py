# -*- coding: utf-8 -*-

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz

!tar xf spark-3.2.1-bin-hadoop3.2.tgz

!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"

import findspark
findspark.init()

import pandas as pd


import findspark
findspark.init()

import pyspark

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import lag, lead
from pyspark.sql.window import Window
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

# """
# Spark Setup
# """
# #SparkConf().set('spark.driver.host','127.0.0.1').setAppName("NewsStreamingApp").setMaster("local[2]")
# conf = SparkConf().set('spark.driver.host','127.0.0.1').setMaster("local[2]")
# sc = SparkContext.getOrCreate(conf)
# spark = SQLContext (sc)


# def getSparkSessionInstance(sparkConf):
#     if ('sparkSessionSingletonInstance' not in globals()):
#         globals()['sparkSessionSingletonInstance'] = SparkSession\
#             .builder\
#             .config(conf=sparkConf)\
#             .getOrCreate()
#     return globals()['sparkSessionSingletonInstance']

# # create the Streaming Context from the above spark context with window size 2 seconds
# ssc = StreamingContext(sc, 2)
# # read data from port 5000
# dataStream = ssc.socketTextStream("127.0.0.1",5000)

from google.colab import drive
drive.mount('/content/drive')

# Reading the necessary files

#Input file
Stocks_Data = pd.read_csv("/content/drive/MyDrive/6889proj/Dataset/aapl_history_2yr.csv")

# #Stock data for Apple

# fileName = "/content/aapl_history_3mo.csv"
# Apple = pd.read_csv(fileName)

StocksInfo = spark.createDataFrame(Stocks_Data)

#StocksInfo = spark.read.csv('IBMStockData.csv',header = True, inferSchema=True)
StocksInfo.show()

# #lag stock
# StockInfo_Lead = StocksInfo.withColumn("lead",lead(("Close")).over(Window.partitionBy().orderBy("Date")))

# StockInfo_Lag2 = StocksInfo.withColumn("lead",lead(("Close"),2).over(Window.partitionBy().orderBy("Date")))

# StockInfo_Lag3 = StocksInfo.withColumn("lead",lead(("Close"),3).over(Window.partitionBy().orderBy("Date")))



# StockInfo_Lead.show()
# StockInfo_Lag2.show()
# StockInfo_Lag3.show()

Data = StocksInfo.select("Open","Volume","Sentiment_Score","Close")
# Data = Data.filter(Data["label"].isNotNull())
# Data = StocksInfo.select("Open","Volume","Sentiment_Score")


# Data_Lag2 = StockInfo_Lag2.select("Open","Volume","Sentiment_Score",StockInfo_Lag2.lead.alias("label"))
# Data_Lag2 = Data_Lag2.filter(Data_Lag2["label"].isNotNull())

# Data_Lag3 = StockInfo_Lag3.select("Open","Volume","Sentiment_Score",StockInfo_Lag3.lead.alias("label"))
# Data_Lag3 = Data_Lag3.filter(Data_Lag3["label"].isNotNull())

Data.show()
# print(Data.shape())

assembler = VectorAssembler(
    inputCols=['Open','Volume','Sentiment_Score'],
    outputCol="features")

output = assembler.transform(Data)

output.select("Close","features").show(truncate=False)

assembler = VectorAssembler().setInputCols(['Open','Volume','Sentiment_Score']).setOutputCol('features')
scaler = MinMaxScaler(inputCol="features", outputCol="features_scaled")

pipeline = Pipeline(stages=[assembler, scaler])

lr = LinearRegression()

scalerModel = pipeline.fit(Data)
scaledData = scalerModel.transform(Data)

dataset=scaledData.select("Close",scaledData.features_scaled.alias('features'))
dataset=dataset.withColumnRenamed("Close","label")
dataset.show()


train, test = dataset.randomSplit([0.8,0.2])

model = lr.fit(train)

pred = model.transform(test)

pred.show()
# test.show()
evaluator = RegressionEvaluator()
print("RMSE for stock price prediction",evaluator.evaluate(pred,
{evaluator.metricName: "rmse"})
)
pandaspred = pred.toPandas()
pandaspred['new_col'] = range(1, len(pandaspred) + 1)
pandaspred=pandaspred.rename(columns={"new_col": "date", "label": "valid"})
print(pandaspred)
pandaspred.plot(x="date", y=["valid", "prediction"], kind="line", figsize=(16, 8),title='model')
# pandaspred.plot.title('Model')
# pandaspred.plot.xlabel('Date', fontsize=18)
# pandaspred.plot.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()