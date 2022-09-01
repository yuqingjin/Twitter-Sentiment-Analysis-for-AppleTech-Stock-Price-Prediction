# -*- coding: utf-8 -*-


# !apt-get install openjdk-8-jdk-headless -qq > /dev/null

# !wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz

# !tar xf spark-3.2.1-bin-hadoop3.2.tgz

# !pip install -q findspark

# import os
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"

# import findspark
# findspark.init()

# !pip install pyspark
# !pip install findspark

import pyspark
# import findspark
# findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
# from pyspark.sql.types import *
import pandas as pd
import pyspark.sql.functions as f

# sc.stop()
sc = SparkContext()
sentiment_score = sc.textFile("/content/streaming_test.csv")

sentiment_score.take(5)

sc.stop()
spark = SparkSession.builder\
          .appName("readfromcsv")\
          .master("local[4]")\
          .getOrCreate()

# Load data into a streaming dataframe
schema = StructType([StructField("CreatedTime", TimestampType(), True),
                     StructField("Token", StringType(), True),
                     StructField("Score", FloatType(), True)])
InputStream = spark.readStream.format ("csv").schema(schema).option("header", True).option("maxFilesPerTrigger", 1)\
.load("/content/streaming_test.csv")

# Check status
InputStream.isStreaming

InputStream.printSchema()

type(InputStream)

rdd = InputStream\
.groupBy("CreatedTime")\
.sum("Score").alias("sum_score")\
.sort(asc("CreatedTime"))

query = rdd.writeStream.queryName("alltweets")\
        .outputMode("append").format("csv")\
        .option("path", "./")\
        .option("checkpointLocation", "./check/")\
        .trigger(processingTime='60 seconds').start()

query.awaitTermination()