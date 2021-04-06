
import nltk
import numpy as np
import pandas as pd
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("C:\\Users\\SIDDHARTH\\Desktop\\hate speech\\train.csv")

import re
def tweet_cleaner(tweet) :
    tweet = re.sub("(@[A-Za-z0-9]+)|(#)|(RT[\s]+)|(https?:\/\/\S+)|([^a-zA-Z0-9 -])", "", tweet)
    return tweet


df['clean_tweet'] = df['tweet'].apply(tweet_cleaner)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([word for word in x.split() if not word in set(stopwords.words('english'))]))
df.drop(['tweet','id'],inplace = True,axis=1)
#Lemmitization
lemmatizer = WordNetLemmatizer()
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


train, test = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state=21)

train.label.value_counts(normalize=True)
test.label.value_counts(normalize=True)

tfidf_vectorizer = TfidfVectorizer(lowercase= True,  stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(train.clean_tweet)


train_idf = tfidf_vectorizer.transform(train.clean_tweet)
test_idf  = tfidf_vectorizer.transform(test.clean_tweet)

model_LR = LogisticRegression(C=24,class_weight='balanced')
model_LR.fit(train_idf, train.label)

predict_train = model_LR.predict(train_idf)

predict_test = model_LR.predict(test_idf)


# f1 score on train data
f1_score(y_true= train.label, y_pred= predict_train)


pickle.dump(tfidf_vectorizer,open('tfidf','wb'))
pickle.dump(model_LR,open('model','wb'))
