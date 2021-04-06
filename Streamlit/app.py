import uvicorn
from fastapi import FastAPI
import nltk
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import pickle
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)
pickle_in = open("tfidf","rb")
classifier=pickle.load(pickle_in)
model=pickle.load(open('model','rb'))

#@app.route('/')
def welcome():
    return "Welcome All"

def tweet_cleaner(tweet) :
        tweet = re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", tweet)
        return tweet


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

def main():
    st.title("HATE SPEECH ANALYSIS")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Hate Speech Analysis ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text = st.text_input("Tweet","Type Here")
    
    result=""
    if st.button("Predict"):
        result=HSD(text)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()