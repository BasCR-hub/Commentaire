import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import nltk
import numpy as np 
import pandas as pd 
 
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
 
st.title("Restaurant : analyse des commentaires")
  
df = pd.read_csv('resampled_comments.csv')
df.head()
 
vec = CountVectorizer(stop_words = 'english')
vec.fit(df.comment)
X = vec.transform(df.comment).toarray()
df.loc[df["rating"]>2,"sentiment"] = "positive"
df.loc[df["rating"]<=2,"sentiment"] = "negative"
y = df["sentiment"]
 
X_train, X_test, y_train, y_test = train_test_split(X ,y , stratify=y, test_size=0.3, random_state = 1)
 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
 
input_box = st.text_input("Opinion", value='', max_chars=None, key=None, type='default')
tempdf = pd.DataFrame(columns= ["comment"])
tempdf.loc[0,"comment"] = input_box
preproc_text = vec.transform([input_box]).toarray()
Ypred=gnb.predict(preproc_text)
result = st.text(Ypred[0])
