# -*- coding: utf-8 -*-

from textblob import TextBlob
def polarity_detection(text):
    return TextBlob(text).sentiment.polarity
def subjectivity_detection(text):
    return TextBlob(text).sentiment.subjectivity
def text_classification(words):
    # polarity detection
    polarity_detection_udf = udf(polarity_detection, StringType())
    words = words.withColumn("polarity", polarity_detection_udf("Token"))
    # subjectivity detection
    subjectivity_detection_udf = udf(subjectivity_detection, StringType())
    words = words.withColumn("subjectivity", subjectivity_detection_udf("Token"))
    return words

from csv import reader
from csv import writer
result=[]
# skip first line i.e. read header first and then iterate over each row od csv as a list
with open('/content/clean_data_24months.csv', 'r') as read_obj:
  with open('/content/output.csv', 'w') as write_obj:
    csv_reader = reader(read_obj, delimiter=',')
    csv_writer = writer(write_obj, lineterminator='\n')
    header = next(csv_reader)
    # Check file as empty
    if header != None:
      # print(split_reader)
        # Iterate over each row after the header in the csv
      for row in csv_reader:
        row.append((TextBlob(row[1]).sentiment.polarity))
        result.append(row)
          # row variable is a list that represents a row in csv
        # print(row)
      # print(result)
      csv_writer.writerows(result)