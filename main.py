from spamNB import get_data, gen_features, train, accuracy
from spamTfIdf import gen_features_tfidf, train_tfidf_NB, train_tfidf_SVM_Lin, train_tfidf_tree, train_tfidf_KNN, train_tfidf_RandomForest, train_tfidf_AdaBoost, predict, classify_input
from spamCrossval import crossval
import time

all_emails, labels = get_data("C:\\Users\\Param\\Documents\\IIT Guwahati\\Eternus ML\\enron1")
# Naive Bayes with NLTK :
combine = zip(all_emails, labels)
start = time.time()
feature_set = gen_features(combine)
train_set, test_set, clf = train(feature_set, 0.8)
total = time.time() - start
print("Time taken to generate features, train and predict : %0.3f s\n" % (total))
print("\nAccuracy for Naive Bayes with NLTK : %0.2f%%\n" % (accuracy(test_set, clf) * 100))

# Naive Bayes (sklearn) with Tf-Idf :
start = time.time()
X = gen_features_tfidf(all_emails)
#x_test, y_test, clf_NB = train_tfidf_NB(X, labels)
#print("Accuracy for Naive Bayes with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_NB) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X, labels, "NB")
#print(classify_input(clf))

# SVM with linear kernel, with Tf-Idf :
start = time.time()
X_lin = gen_features_tfidf(all_emails)
#x_test, y_test, clf_SVM_lin = train_tfidf_SVM_Lin(X_lin, labels)
#print("Accuracy for SVM (Linear) with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_SVM_lin) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X_lin, labels, "SVM_lin")
#print(classify_input(clf))

# Decision Tree, with Tf-Idf :
start = time.time()
X_tree = gen_features_tfidf(all_emails)
#x_test, y_test, clf_tree = train_tfidf_tree(X_tree, labels)
#print("Accuracy for Decision Tree with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_tree) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X_tree, labels, "tree")
#print(classify_input(clf))

# K-Nearest Neighbors, with Tf-Idf :
start = time.time()
X_knn = gen_features_tfidf(all_emails)
#x_test, y_test, clf_knn = train_tfidf_KNN(X_knn, labels)
#print("Accuracy for KNN with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_knn) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X_tree, labels, "KNN")
#print(classify_input(clf))

# Random Forest, with Tf-Idf :
start = time.time()
X_randomforest = gen_features_tfidf(all_emails)
#x_test, y_test, clf_randomforest = train_tfidf_RandomForest(X_randomforest, labels)
#print("Accuracy for Random Forest with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_randomforest) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X_randomforest, labels, "random_forest")
#print(classify_input(clf))

# Ada Boost, with Tf-Idf :
start = time.time()
X_adaboost = gen_features_tfidf(all_emails)
#x_test, y_test, clf_adaboost = train_tfidf_AdaBoost(X_adaboost, labels)
#print("Accuracy for Ada Boost with Tf-Idf : %0.2f%%" % (predict(x_test, y_test, clf_adaboost) * 100))
total = time.time() - start
print("\nTime taken to generate features, train and predict : %0.3f s\n" % (total))
crossval(X_adaboost, labels, "adaboost")
#print(classify_input(clf))

