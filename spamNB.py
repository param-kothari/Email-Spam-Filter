import nltk
import random
import os
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

stop_wrd = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

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

def preprocess (text):
  lower_text = text.lower()
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  words = word_tokenize(lower_text)
  words_withoutstop = set(word for word in words if word not in stop_wrd)
  final_words = [lemm.lemmatize(word) for word in words_withoutstop]
  feature_nopunc = list()
  for char in final_words:
    if char not in punctuations:
      feature_nopunc.append(char)
  return feature_nopunc

def gen_features (text):
  dict_features = []
  for (block, label) in text:
    features = {word : True for word in preprocess(block)}
    dict_features.append((features, label))
  return dict_features  


def train (feature_set, split):
  train_size = int(len(feature_set) * split)
  x_train_set, x_test_set = list(), list()
  random.shuffle(feature_set)
  x_train_set = feature_set[:train_size]
  x_test_set = feature_set[train_size:]
  clf = NaiveBayesClassifier.train(x_train_set)
  return x_train_set, x_test_set, clf

def accuracy (test_set, clf):
  return classify.accuracy(clf, test_set)

def predict_spam (text, clf):
  #input_text = {word : True for word in preprocess(text)}
  print(clf.show_most_informative_features(5))
  

  


