from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split  
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import time

tfidf = TfidfVectorizer(analyzer = 'word', stop_words = stopwords.words('english'), max_features= 3000)

def get_data (rootdir):
  spam_list, ham_list = [], []
  for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        for filename in files:      
            with open(os.path.join(directories, filename), encoding = "latin-1") as f:
                data = f.read()
                ham_list.append(data) 
    if (os.path.split(directories)[1]  == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding = "latin-1") as f:
                data = f.read()
                spam_list.append(data)
  label = []
  for idx in range(0, len(ham_list)):
    label.append(0)
  for idx in range(0, len(spam_list)):
    label.append(1)
  combined = ham_list + spam_list  
  return combined, label

def gen_features_tfidf (text):
  X = tfidf.fit_transform(text).todense()
  joblib.dump(tfidf, 'tfidf.pkl')
  return X

def train_tfidf_NB (X, label):
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = GaussianNB()#is sequential learning possible
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier

def train_tfidf_SVM_Lin (X, label):
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = LinearSVC(C = 1) 
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier

def train_tfidf_tree (X, label) :
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = tree.DecisionTreeClassifier(criterion= "gini")#max split #gini
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier

def train_tfidf_KNN (X, label) :
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = KNeighborsClassifier()
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier

def train_tfidf_RandomForest (X, label) :
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = RandomForestClassifier(n_estimators= 100, criterion= "entropy")#incremental learning
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier

def train_tfidf_AdaBoost (X, label) :
  x_train, x_test, y_train, y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)  
  classifier = AdaBoostClassifier()
  classifier.fit(x_train, y_train)
  return x_test, y_test, classifier 

def predict (x_test, y_test, clf):
  y_pred = clf.predict(x_test)
  #print(classification_report(y_test,y_pred))  
  return accuracy_score(y_test, y_pred)

def classify_input(clf):
  tfidf = joblib.load('tfidf.pkl')
  text = []
  text.append(input("Enter text to classify : "))
  X_inp = tfidf.transform(text).todense()
  if clf.predict(X_inp):
    return "spam"
  else:
    return "ham"  
