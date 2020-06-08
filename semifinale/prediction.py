import pandas as pd
import numpy as np
from nltk import RegexpTokenizer
import string
import re
from nltk.corpus import stopwords 
import nltk 
from nltk.stem.snowball import SnowballStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes, svm
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

#nltk.download('stopwords')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle

# Preprocessing
class Prediction:
    def __init__(self, text):
        self.text = text

    #"first stage preprocessing" steps
    def preprocessing(self,df):
        #remove "Plus"
        df["comment"] = df["comment"].apply(lambda x: x.replace("Plus",''))

        #make everything lowercase 
        df["comment"] = df["comment"].map(lambda x: x.lower())

        #remove punctuation
        def remove_punct(string):
            lst_to_remove = [".","!","?","/","\'", "’"]         
            for element in lst_to_remove:
                string = string.replace(element,' ')
            return string
        df["comment"] = df["comment"].apply(remove_punct)

        #stemming
        def stem(text):
            words = text.split(" ")
            stemmed_words = [stemmer.stem(word) for word in words]
            result = " ".join(stemmed_words)
            return result
        stemmer = SnowballStemmer(language='french')
        df["comment"] = df["comment"].apply(stem)

        return df

    #create the vectorizer that will be used in production
    def getprediction(self, input, option):
        clf = pickle.load(open("ensemble_classifier.p", 'rb'))
        logreg = pickle.load(open("Logistic_Regression_model.p", 'rb'))
        mnb = pickle.load(open("Multinomial_NB_model.p","rb"))
        svc = pickle.load(open("SVC_model.p","rb"))
        vectorizer = pickle.load(open("vectorizer.p", 'rb'))

        # cv = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('french'))
        def preprocessing_phase_1(df):
            #remove "Plus"
            df["comment"] = df["comment"].apply(lambda x: x.replace("Plus",''))
            #make everything lowercase 
            df["comment"] = df["comment"].map(lambda x: x.lower())
            #remove punctuation
            def remove_punct(string):
                lst_to_remove = [".","!","?","/","\'", "’"]         
                for element in lst_to_remove:
                    string = string.replace(element,' ')
                return string
            df["comment"] = df["comment"].apply(remove_punct)
            #stemming
            def stem(text):
                words = text.split(" ")
                stemmed_words = [stemmer.stem(word) for word in words]
                result = " ".join(stemmed_words)
                return result
            stemmer = SnowballStemmer(language='french')
            df["comment"] = df["comment"].apply(stem)
            return df
        # Separate data and labels

        # X = self.comments['comment']
        # y = self.comments['sentiment']
        
        # Using a hashing vectorizer to keep model size low
        df = pd.DataFrame(columns=["comment"])
        df.loc[0,"comment"] = input
        df = preprocessing_phase_1(df)
        X = vectorizer.transform(df.comment)

        # X_train, X_test, y_train, y_test = train_test_split(X_fitted, y, test_size=0.25, random_state=1)

        # Linear SVM powered by SGD Classifier
        if(option == "svm"):
            print("*********** using svm *******************")
            y_pred = mnb.predict(X)

        # RandomForestClassifier
        elif(option == "rc"):
            print("*********** using RandomForestClassifier *******************")
            # clf = RandomForestClassifier(n_estimators=1000, random_state=0)
            # clf.fit(X_train, y_train)
            y_pred = svc.predict(X)

        elif(option =="sc"):
            # TF-IDF matrice
            y_pred = logreg.predict(X)


        # cf_matrix = confusion_matrix(y_test, y_pred)
        # self.df_cm = pd.DataFrame(cf_matrix, range(3),range(3))

        #Classification Report
        # self.report = classification_report(y_test, y_pred, output_dict=True)
        return y_pred

    def getsampledata(self, size):
        return self.df.head(size)

    def getsampledata2(self, size):
        df2 = pd.read_csv('static/models/resampled_comments_1.csv')
        df2 = df2[['rating','comment','bonus_info','city']]
        #df2.drop('unnamed:0', axis=1)
        return df2.head(size)

    def getcr(self):
        return pd.DataFrame(self.report).transpose()

    def create_cm(self):
        # plot (powered by seaborn)
        fig = Figure()
        plt.rcParams["figure.figsize"] = (15,10)
        ax = fig.add_subplot(1, 1, 1)
        
        sn.set(font_scale=1)
        sn.heatmap(self.df_cm, ax = ax, annot=True,annot_kws={"size": 16}, fmt='g')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(['negative', 'positive', 'neutral'])
        ax.yaxis.set_ticklabels(['negative', 'positive', 'neutral'])
        
        return fig

