

import uvicorn
from fastapi import FastAPI
import nltk
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import pickle



app = FastAPI(debug=True)
pickle_in = open("tfidf","rb")
classifier=pickle.load(pickle_in)
model=pickle.load(open('model','rb'))
def tweet_cleaner(tweet) :
        tweet = re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", tweet)
        return tweet


@app.get('/')
def index():
    return {'message': 'Hello, World'}




@app.post('/predict')

def HSD(text):
    text=tweet_cleaner(text)
    lemmatizer = WordNetLemmatizer()
    text=[lemmatizer.lemmatize(text)]
    text=classifier.transform(text)
    result=model.predict(text)
    if result==0:
            return {'Prediction' : 'Normal Tweet'}
    else:
            return {'Prediction' : 'Hate Tweet'}

       
   
      

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)
    
