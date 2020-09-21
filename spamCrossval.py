from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def crossval (X, labels, key) :
    if key == "NB" :
        scores = cross_val_score(GaussianNB(), X, labels, cv = 5)
        print("Accuracy for Naive Bayes with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))
    elif key == "SVM_lin" :
        scores = cross_val_score(LinearSVC(C = 1), X, labels, cv = 5)
        print("Accuracy for SVM (linear) with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))
    elif key == "tree" :
        scores = cross_val_score(tree.DecisionTreeClassifier(criterion= 'gini'), X, labels, cv = 5)
        print("Accuracy for Decision Tree with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))
    elif key == "KNN" :
        scores = cross_val_score(KNeighborsClassifier(), X, labels, cv = 5)
        print("Accuracy for KNN with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))    
    elif key == "random_forest" :
        scores = cross_val_score(RandomForestClassifier(n_estimators= 100, criterion= 'entropy'), X, labels, cv = 5)
        print("Accuracy for Random Forest with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))    
    elif key == "adaboost" :
        scores = cross_val_score(AdaBoostClassifier(), X, labels, cv = 5)
        print("Accuracy for AdaBoost with Tf-Idf: %0.2f %% (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 200))    

    return