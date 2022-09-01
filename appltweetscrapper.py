# -*- coding: utf-8 -*-

!pip install textblob
!pip install tweepy
!pip install pycountry
!pip install langdetect

from textblob import TextBlob
import sys
import tweepy
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time

#retrive token from the envioment
def auth():
  return os.getenv('TOKEN')

# take bearer token, pass it for authorization and return headers used to access the API
def create_headers(bearer_token):
  headers = {"Authorization": "Bearer {}".format(bearer_token)}
  return headers

# build the request for the endpoint we are going to use and the parameters we want to pass
def create_url(keyword, start_date, end_date,max_results):
    
    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params ={'query': keyword,
            'start_time': start_date,
            'end_time': end_date,
            'max_results': max_results,
            'tweet.fields': 'id,text,author_id,created_at',
            # 'user.fields': 'id,name,username,',
            'next_token': {}}
    return (search_url, query_params)

#connnect to endpoint
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "Apple lang:en"
start_time = "2021-01-01T00:00:00.000Z"
end_time = "2021-12-31T00:00:00.000Z"

url = create_url(keyword, start_time, end_time,10)
json_response = connect_to_endpoint(url[0], headers, url[1])

print(json.dumps(json_response, indent=4, sort_keys=True))

def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 8. Tweet text
        text = tweet['text']
        
        # Assemble all data in a list
        res = [author_id, created_at, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)

#Inputs for tweets
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "Apple lang:en"
start_list =    ['2020-01-01T00:00:00.000Z',
           '2020-02-01T00:00:00.000Z',
           '2020-03-01T00:00:00.000Z',
           '2020-04-01T00:00:00.000Z',
           '2020-05-01T00:00:00.000Z',
           '2020-06-01T00:00:00.000Z',
           '2020-07-01T00:00:00.000Z',
           '2020-08-01T00:00:00.000Z',
           '2020-09-01T00:00:00.000Z',
           '2020-10-01T00:00:00.000Z',
           '2020-11-01T00:00:00.000Z',
           '2020-12-01T00:00:00.000Z',
           '2021-01-01T00:00:00.000Z',
           '2021-02-01T00:00:00.000Z',
           '2021-03-01T00:00:00.000Z',
           '2021-04-01T00:00:00.000Z',
           '2021-05-01T00:00:00.000Z',
           '2021-06-01T00:00:00.000Z',
           '2021-07-01T00:00:00.000Z',
           '2021-08-01T00:00:00.000Z',
           '2021-09-01T00:00:00.000Z',
           '2021-10-01T00:00:00.000Z',
           '2021-11-01T00:00:00.000Z',
           '2021-12-01T00:00:00.000Z']

end_list =     ['2020-1-31T00:00:00.000Z',
           '2020-2-27T00:00:00.000Z',
           '2020-3-31T00:00:00.000Z',
           '2020-4-30T00:00:00.000Z',
           '2020-5-31T00:00:00.000Z',
           '2020-6-30T00:00:00.000Z',
           '2020-7-31T00:00:00.000Z',
           '2020-8-31T00:00:00.000Z',
           '2020-9-30T00:00:00.000Z',
           '2020-10-31T00:00:00.000Z',
           '2020-11-30T00:00:00.000Z',
           '2020-12-31T00:00:00.000Z',
           '2021-1-31T00:00:00.000Z',
           '2021-2-27T00:00:00.000Z',
           '2021-3-31T00:00:00.000Z',
           '2021-4-30T00:00:00.000Z',
           '2021-5-31T00:00:00.000Z',
           '2021-6-30T00:00:00.000Z',
           '2021-7-31T00:00:00.000Z',
           '2021-8-31T00:00:00.000Z',
           '2021-9-30T00:00:00.000Z',
           '2021-10-31T00:00:00.000Z',
           '2021-11-30T00:00:00.000Z',
           '2021-12-31T00:00:00.000Z']
max_results = 500

#Total number of tweets we collected from the loop
total_tweets = 0

# Create file
csvFile = open("data_24months.csv", "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save, in this example, we only want save these columns in our dataset
csvWriter.writerow(['author id', 'created_at','tweet'])
csvFile.close()

for i in range(0,len(start_list)):

    # Inputs
    count = 0 # Counting tweets per time period
    max_count = 500 # Max tweets per time period
    flag = True
    next_token = None
    
    # Check if flag is true
    while flag:
        # Check if max_count reached
        if count >= max_count:
            break
        # print("-------------------")
        # print("Token: ", next_token)
        url = create_url(keyword, start_list[i],end_list[i], max_results)
        json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
        result_count = json_response['meta']['result_count']

        if 'next_token' in json_response['meta']:
            # Save the token to use for next call
            next_token = json_response['meta']['next_token']
            print("Next Token: ", next_token)
            if result_count is not None and result_count > 0 and next_token is not None:
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data_24months.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)                
        # If no next token exists
        else:
            if result_count is not None and result_count > 0:
                print("-------------------")
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data_24months.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)
            
            #Since this is the final request, turn flag to false to move to the next time period.
            flag = False
            next_token = None
        time.sleep(5)
print("Total number of results: ", total_tweets)