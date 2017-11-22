from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
# import pandas
# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def NeuralNet(data,kf):
    print('\nNeural Nets')
    AccListNn =[]
    x = np.array(data.data)
    y = np.array(data.target)
    mlp=MLPClassifier(hidden_layer_sizes=(5),activation='relu',solver='lbfgs', alpha=0.01,learning_rate='adaptive',learning_rate_init=0.5, max_iter=100)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mlp.fit(x_train, y_train)
        predict = mlp.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListNn.append(accuracy)
    print('\n-> Accuracy List', AccListNn)
    print('\n-> Mean Accuracy of Neural Nets', np.mean(AccListNn),'%')

def DecisionTree(data,kf):
    print('\nDecision Tree')
    AccListDtree = []
    Dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)
    x =  np.array(data.data)
    y = np.array(data.target)
    for train_index, test_index in kf.split(x):
        # print(train_index,test_index)
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Dtree.fit(x_train,y_train)
        predict = Dtree.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        # print(accuracy, '%')
        AccListDtree.append(accuracy)
    print('\n-> Accuracy List', AccListDtree)
    print('\n-> Mean Accuracy of Dtree',np.mean(AccListDtree),'%')

def Perceptron(data, kf):
    print('\nPerceptron')
    AccListPercep = []
    x = np.array(data.data)
    y = np.array(data.target)
    perceptron = MLPClassifier(hidden_layer_sizes=(1),learning_rate_init=0.5, max_iter=100)
    for train_index, test_index in kf.split(x):
        # print(train_index,test_index)
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        perceptron.fit(x_train, y_train)
        predict = perceptron.predict(x_test)
        accuracy = accuracy_score(y_test, predict)*100
        AccListPercep.append(accuracy)
    print('\n-> Accuracy List', AccListPercep)
    print('\n-> Mean Accuracy of Perceptron', np.mean(AccListPercep), '%')

def DeepLearning(data,kf):
    print('\nDeep Learning')
    AccListDl = []
    x = np.array(data.data)
    y = np.array(data.target)
    mlp=MLPClassifier(hidden_layer_sizes=(10,15,10),activation='relu',solver='lbfgs', alpha=0.01,learning_rate='adaptive',learning_rate_init=0.5)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mlp.fit(x_train, y_train)
        predict = mlp.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListDl.append(accuracy)
    print('\n-> Accuracy List', AccListDl)
    print('\n-> Mean Accuracy of Deep Learning', np.mean(AccListDl), '%')

def SVM(data, kf):
    print('\nSupport Vector Machine')
    AccListSvm = []
    x = np.array(data.data)
    y = np.array(data.target)
    Svm = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Svm.fit(x_train, y_train)
        predict = Svm.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListSvm.append(accuracy)
    print('\n-> Accuracy List', AccListSvm)
    print('\n-> Mean Accuracy of Support Vector Machines', np.mean(AccListSvm), '%')

def NaiveBayes(data,kf):
    print('\nNaive Bayes')
    AccListGnb = []
    AccListMnb = []
    AccListBnb = []
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    x = np.array(data.data)
    y = np.array(data.target)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb.fit(x_train, y_train)
        mnb.fit(x_train, y_train)
        bnb.fit(x_train, y_train)
        GnbPredict = gnb.predict(x_test)
        MnbPredict = mnb.predict(x_test)
        BnbPredict = bnb.predict(x_test)
        GnbAccuracy = accuracy_score(GnbPredict, y_test)*100
        MnbAccuracy = accuracy_score(MnbPredict, y_test)*100
        BnbAccuracy = accuracy_score(BnbPredict, y_test)*100
        AccListGnb.append(GnbAccuracy)
        AccListMnb.append(MnbAccuracy)
        AccListBnb.append(BnbAccuracy)
    print('\n-> Accuracy List GNB', AccListGnb)
    print('\n-> Accuracy List MNB', AccListMnb)
    print('\n-> Accuracy List BNB', AccListBnb)
    print('\n-> Mean Accuracy of GNB', np.mean(AccListGnb), '%')
    print('\n-> Mean Accuracy of MNB', np.mean(AccListMnb), '%')
    print('\n-> Mean Accuracy of BNB', np.mean(AccListBnb), '%')
    # print(max(np.mean(AccListGnb), np.mean(AccListMnb), np.mean(AccListBnb)))

def LogisRegression(data,kf):
    print('\nLogistic Regression')
    AccListLr = []
    x = np.array(data.data)
    y = np.array(data.target)
    Lr = LogisticRegression(solver='saga', max_iter=100, penalty='l2')
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Lr.fit(x_train, y_train)
        predict = Lr.predict(x_test)
        accuracy = accuracy_score(y_test, predict)*100
        AccListLr.append(accuracy)
    print('\n-> Accuracy List Logistic Regression', AccListLr)
    print('\n-> Mean accuracy of Logistic Regression', np.mean(AccListLr), '%')

def kNearest(data,kf):
    print('\nK-Nearest Neighbour')
    AccListKnn = []
    x = np.array(data.data)
    y = np.array(data.target)
    knn = KNeighborsClassifier(n_neighbors=50, algorithm='auto', leaf_size=10)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(x_train,y_train)
        predict = knn.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListKnn.append(accuracy)
    print('\n-> Accuracy List of K-nearest neighbour', AccListKnn)
    print('\n-> Mean accuracy of K-nearest neighbour', np.mean(AccListKnn), '%')

def Bagging(data, kf):
    print('\nBagging')
    AccListBgg = []
    x = np.array(data.data)
    y = np.array(data.target)
    bgg = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_estimators=30,bootstrap_features=False)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        bgg.fit(x_train,y_train)
        predict = bgg.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListBgg.append(accuracy)
    print('\n-> Accuracy List of Bagging', AccListBgg)
    print('\n-> Mean Accuracy of Bagging', np.mean(AccListBgg),'%')

def RandomForest(data,kf):
    print('\n Random Forest')
    AccListRf = []
    x = np.array(data.data)
    y = np.array(data.target)
    Rf = RandomForestClassifier(n_estimators=25, criterion='entropy', max_features='auto',random_state=20)
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Rf.fit(x_train, y_train)
        predict = Rf.predict(x_test)
        accuracy = accuracy_score(y_test, predict)*100
        AccListRf.append(accuracy)
    print('\n-> Accuracy List of Random Forest', AccListRf)
    print('\n-> Mean Accuracy of Random Forest', np.mean(AccListRf),'%')

def AdaBoost(data, kf):
    print('\nAdaboost')
    AccListAdb = []
    x = np.array(data.data)
    y = np.array(data.target)
    adb = AdaBoostClassifier(n_estimators=30, learning_rate=1, algorithm='SAMME.R')
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        adb.fit(x_train,y_train)
        predict = adb.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListAdb.append(accuracy)
    print('\n-> Accuracy List of Adaboost', AccListAdb)
    print('\n-> Mean Accuracy of Adaboost', np.mean(AccListAdb), '%')

def GradientBoosting(data, kf):
    print('\nGradient Boosting')
    AccListGb = []
    x = np.array(data.data)
    y = np.array(data.target)
    Gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=10, random_state=50, max_features='auto')
    for train_index, test_index in kf.split(x):
        x_train, x_test, = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Gb.fit(x_train, y_train)
        predict = Gb.predict(x_test)
        accuracy = accuracy_score(y_test,predict)*100
        AccListGb.append(accuracy)
    print('\n-> Accuracy List of Gradient Boosting', AccListGb)
    print('\n-> Mean Accuracy of Gradient Boosting', np.mean(AccListGb), '%')

if __name__ == "__main__":
    iris = load_iris()
    print('Number of Instances: ', len(iris.data))
    print('Names of Features: ',iris.feature_names)
    print('Number Of Features: ', len(iris.feature_names))
    print('Classes: ', iris.target_names)
    print('Number of Classes: ', len(iris.target_names))
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # plt.show()
    kfold = KFold(n_splits=10, shuffle=True, random_state=50)
    # NeuralNet(iris, kfold)
    # DecisionTree(iris, kfold)
    # Perceptron(iris, kfold)
    # DeepLearning(iris, kfold)
    # SVM(iris, kfold)
    # NaiveBayes(iris, kfold)
    # LogisRegression(iris, kfold)
    # kNearest(iris, kfold)
    # Bagging(iris, kfold)
    # RandomForest(iris,     kfold)
    # AdaBoost(iris, kfold)
    # GradientBoosting(iris, kfold)
    # NewPerceptron(iris, kfold)
    x = np.array(iris.data)
    y = np.array(iris.target)
    AccListNn = []
    AccListDtree = []
    AccListPercep = []
    AccListDl = []
    AccListSvm = []
    AccListGnb = []
    AccListLr = []
    AccListKnn = []
    AccListBgg = []
    AccListRf = []
    AccListAdb = []
    AccListGb = []
    NnPredict = []
    NnAccuracy = []
    DtPredict = []
    DtreeAccuracy = []
    PercepPredict = []
    PercepAccuracy = []
    DlPredict = []
    DlAccuracy = []
    SvmPredict = []
    SvmAccuracy = []
    GnbPredict = []
    GnbAccuracy = []
    LrPredict = []
    LrAccuracy = []
    KnnPredict = []
    KnnAccuracy = []
    BggPredict = []
    BggAccuracy = []
    RfPredict = []
    RfAccuracy = []
    AdbPredict = []
    AdbAccuracy = []
    GbPredict = []
    GbAccuracy = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #Neural Net
        mlp = MLPClassifier(hidden_layer_sizes=(5), activation='relu', solver='lbfgs', alpha=0.01,learning_rate='adaptive', learning_rate_init=0.5)
        mlp.fit(x_train, y_train)
        NnPredict = mlp.predict(x_test)
        NnAccuracy = accuracy_score(y_test, NnPredict) * 100
        AccListNn.append(NnAccuracy)
        #Decision Tree
        Dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5)
        Dtree.fit(x_train, y_train)
        DtPredict = Dtree.predict(x_test)
        DtreeAccuracy = accuracy_score(y_test, DtPredict) * 100
        AccListDtree.append(DtreeAccuracy)
        #Perceptron
        perceptron = MLPClassifier(hidden_layer_sizes=(1),learning_rate_init=0.5)
        perceptron.fit(x_train, y_train)
        PercepPredict = perceptron.predict(x_test)
        PercepAccuracy = accuracy_score(y_test, PercepPredict) * 100
        AccListPercep.append(PercepAccuracy)
        #Deep Learning
        Dl = MLPClassifier(hidden_layer_sizes=(10, 15, 10), activation='relu', solver='lbfgs', alpha=0.01, learning_rate='adaptive', learning_rate_init=0.5)
        Dl.fit(x_train, y_train)
        DlPredict = mlp.predict(x_test)
        DlAccuracy = accuracy_score(y_test, DlPredict) * 100
        AccListDl.append(DlAccuracy)
        #Support Vectore Machines
        Svm = svm.SVC(C=1.0, cache_size=300, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        Svm.fit(x_train, y_train)
        SvmPredict = Svm.predict(x_test)
        SvmAccuracy = accuracy_score(y_test, SvmPredict) * 100
        AccListSvm.append(SvmAccuracy)
        #Gaussian Naive Bayes
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        GnbPredict = gnb.predict(x_test)
        GnbAccuracy = accuracy_score(y_test, GnbPredict) * 100
        AccListGnb.append(GnbAccuracy)
        #Logistic Regression
        Lr = LogisticRegression(solver='liblinear', max_iter=100, penalty='l2')
        Lr.fit(x_train, y_train)
        LrPredict = Lr.predict(x_test)
        LrAccuracy = accuracy_score(y_test, LrPredict) * 100
        AccListLr.append(LrAccuracy)
        #K-nearest Neighbour
        knn = KNeighborsClassifier(n_neighbors=15, algorithm='auto', leaf_size=50)
        knn.fit(x_train, y_train)
        KnnPredict = knn.predict(x_test)
        KnnAccuracy = accuracy_score(y_test, KnnPredict) * 100
        AccListKnn.append(KnnAccuracy)
        #Bagging
        bgg = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_estimators=30, bootstrap_features=False)
        bgg.fit(x_train, y_train)
        BggPredict = bgg.predict(x_test)
        BggAccuracy = accuracy_score(y_test, BggPredict) * 100
        AccListBgg.append(BggAccuracy)
        #Randon Forest
        Rf = RandomForestClassifier(n_estimators=25, criterion='entropy', max_features='auto', random_state=20)
        Rf.fit(x_train, y_train)
        RfPredict = Rf.predict(x_test)
        RfAccuracy = accuracy_score(y_test, RfPredict) * 100
        AccListRf.append(RfAccuracy)
        #adaboost
        adb = AdaBoostClassifier(n_estimators=30, learning_rate=1,algorithm='SAMME.R')
        adb.fit(x_train,y_train)
        AdbPredict = adb.predict(x_test)
        AdbAccuracy = accuracy_score(y_test,AdbPredict)*100
        AccListAdb.append(AdbAccuracy)
        #gaussian boosting
        Gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=10, random_state=50, max_features='auto')
        Gb.fit(x_train, y_train)
        GbPredict = Gb.predict(x_test)
        GbAccuracy = accuracy_score(y_test, GbPredict) * 100
        AccListGb.append(GbAccuracy)
    # print('\n-> Accuracy List of Gradient Boosting', AccListGb)
    print('\n-> Mean Accuracy of Gradient Boosting', np.mean(AccListGb), '%')
    print('\n-> Gradient Boosting Precision Score', precision_score(y_test, GbPredict, average='weighted', labels=np.unique(y_test))*100,'%')
    # print('\n-> Accuracy List of Adaboost', AccListAdb)
    print('\n-> Mean Accuracy of Adaboost', np.mean(AccListAdb), '%')
    print('\n-> Adaboost Precision Score', precision_score(y_test, AdbPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List of Random Forest', AccListRf)
    print('\n-> Mean Accuracy of Random Forest', np.mean(AccListRf), '%')
    print('\n-> Random Forest Precision Score', precision_score(y_test, RfPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List of Bagging', AccListBgg)
    print('\n-> Mean Accuracy of Bagging', np.mean(AccListBgg), '%')
    print('\n-> Bagging Precision Score', precision_score(y_test, BggPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List of K-nearest neighbour', AccListKnn)
    print('\n-> Mean accuracy of K-nearest neighbour', np.mean(AccListKnn), '%')
    print('\n-> K-nearest neighbour Precision Score', precision_score(y_test, KnnPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List Logistic Regression', AccListLr)
    print('\n-> Mean accuracy of Logistic Regression', np.mean(AccListLr), '%')
    print('\n-> Logistic Regression Precision Score', precision_score(y_test, LrPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List GNB', AccListGnb)
    print('\n-> Mean Accuracy of Gaussian Naive Bayes', np.mean(AccListGnb), '%')
    print('\n-> Gaussian Naive Bayes Precision Score', precision_score(y_test, GnbPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List', AccListSvm)
    print('\n-> Mean Accuracy of Support Vector Machines', np.mean(AccListSvm), '%')
    print('\n-> Support Vector Machines Precision Score', precision_score(y_test, SvmPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List', AccListDl)
    print('\n-> Mean Accuracy of Deep Learning', np.mean(AccListDl), '%')
    print('\n-> Deep Learning Precision Score', precision_score(y_test, DlPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List', AccListPercep)
    print('\n-> Mean Accuracy of Perceptron', np.mean(AccListPercep), '%')
    print('\n-> Perceptron Precision Score', precision_score(y_test, PercepPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List', AccListDtree)
    print('\n-> Mean Accuracy of Dtree', np.mean(AccListDtree), '%')
    print('\n-> D tree Precision Score', precision_score(y_test, DtPredict, average='weighted', labels=np.unique(y_test))*100, '%')
    # print('\n-> Accuracy List', AccListNn)
    print('\n-> Mean Accuracy of Neural Nets', np.mean(AccListNn), '%')
    print('\n-> Neural Nets Precision Score', precision_score(y_test, NnPredict, average='weighted', labels=np.unique(y_test))*100, '%')

