import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from pathlib import Path
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

my_file = Path("model.pkl")

def classify(input):
    
    if not my_file.exists():

        # Using SVM as classifier
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', CalibratedClassifierCV(LinearSVC()))])

        # Reading data file (.csv)
        df = pd.read_csv('shuffled-full-set-hashed.csv', header=None, sep=',', names=['class', 'documents'])
        df.dropna(inplace=True)
        X = df['documents']
        y = df['class']

        # Splitting data into training and validation (67%: training; 33%: validation)
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, df.index, test_size=0.33, random_state=0)

        # Fit data to the classifier
        text_clf.fit(X_train,y_train)

        # Saving model in pickel file if the model is not already trained
        joblib.dump(text_clf, 'model.pkl')


    # Reading file to get the model trained
    text_clf = joblib.load('model.pkl')

    arr = []
    arr.append(input)

    #predicting the given input
    y_pred = text_clf.predict(arr)[0]

    # Getting confidence score
    prob_dist = text_clf.predict_proba(arr)[0]
    max_prob = (str(prob_dist[text_clf.classes_.tolist().index(y_pred)] * 100)+"%")


    # Returning class with its confidence score
    return y_pred, max_prob

