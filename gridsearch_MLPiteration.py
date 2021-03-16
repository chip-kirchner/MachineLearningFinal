"""
CSS 581 Term Project
Chip Kirchner
Winter Quarter 2021
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import linear_model
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

#function to hot-shot encode specific columns, if present
def encode(data):
    #encode non-binary categorical data to vectors and append to dataframe
    #categorical columns: Race Code, Ethnicity Code, Registered Part Code, Gender Code
    cats = ["Race Code","Ethnicity Code","Registered Part Code","Gender Code"]
    temp=[]
    for col in data.columns:
        if col in cats:
            temp.append(col)
    newDf = pd.get_dummies(data,columns=temp)
    #Drivers license is only binary data point. convert from Y/N to 1/0
    #if it exists in selected columns convert it
    if any("Drivers License" in col for col in newDf.columns):
        newDf['Drivers License'] = newDf['Drivers License'].map({'Y': 1, 'N': 0})
    return newDf

#set scores used for evaluation and return dictionary
def getScores(Y_pred,Y_true):
    temp = dict()
    temp["true"] = Y_true
    temp["prediction"] = Y_pred
    temp["accuracy"] = metrics.accuracy_score(Y_true,Y_pred)
    temp["fscore"] = metrics.f1_score(Y_true,Y_pred,average='weighted')
    temp["recall"] = metrics.recall_score(Y_true,Y_pred)
    temp["precision"] = metrics.precision_score(Y_true,Y_pred)
    temp["hamming"] = metrics.hamming_loss(Y_true,Y_pred)
    temp["ROC_AUC"] = metrics.roc_auc_score(Y_true,Y_pred)
    temp["confusion_matrix"] = metrics.confusion_matrix(Y_true,Y_pred)
    return temp

#set global random state variable to 12 for repeatability
np.random.RandomState(12)

#cross-validation used on all gridsearchCV methods
cv=10

#set nrows of data to be read in (dataset too large to run on most devices)
nrows=1000

#bring in data from registration and election history
data = pd.read_csv("data/train/TRAINING_RegHistory.csv",nrows=nrows,converters={'Registration ID': lambda x: str(x)})
#drop duplicate/error records
data = data.loc[(data["11/03/2020 GENERAL"]<2) & (data["11/06/2018 GENERAL"]<2) & 
                (data["11/08/2016 GENERAL"]<2) & (data["11/04/2014 GENERAL"]<2) & 
                (data["11/06/2012 GENERAL"]<2) & (data["03/15/2016 PRIMARY"]<2)]

#get label for classification
labels = data["11/03/2020 GENERAL"]
#drop label, index, voter regiration, geographic info, and local elections
data.drop(["11/03/2020 GENERAL","Unnamed: 0","Registration ID","County ID",
           "Address Zip Code","Birth State","04/30/2019 PRIMARY",
           "05/14/2019 PRIMARY","09/10/2013 PRIMARY","09/10/2019 GENERAL",
           "09/10/2019 PRIMARY","11/03/2015 GENERAL","11/05/2013 GENERAL",
           "11/05/2019 GENERAL","11/07/2017 GENERAL","11/08/2011 GENERAL"],
          axis=1,inplace=True)

#ecode categorical data
encodedData = encode(data)

#split for gridsearchCV validation and full training validation
X_train, X_test, Y_train, Y_test = train_test_split(encodedData, labels,test_size = 1/3, random_state=12)

#PCA Plot and Visualization
pca = PCA()
scaler = MinMaxScaler()
#fit scaled dataset
pca.fit(scaler.fit_transform(X_train))

fig = plt.figure()

plt.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
plt.ylabel('PCA explained variance ratio')
plt.xlabel('n_components')
plt.savefig('pca.png',dpi=400,format='png')
plt.show()



### Begin set up for gridsearchCV runs ###

#set n components for PCA decomp
n_components = list(range(1,X_train.shape[1],10))
#set undersampling ratio
sampling_strategy = [0.4,0.7,1.0]


### Logistic Regression ###
logitReg = linear_model.LogisticRegression(solver='liblinear')

#create pipeline for preprocessing and data fit/predict
pipeline = Pipeline(steps= [("MinMax",MinMaxScaler()),
                            ("pca", PCA()),
                            ("RandomUnderSampler", RandomUnderSampler()),
                            ("RandomOverSampler", RandomOverSampler()),
                            ("LogitReg",logitReg)])

#paramets for logistic regression
C = np.logspace(-3, 2,10)

penalty = ['l1','l2']

parameters = dict(pca__n_components=n_components, 
                  RandomUnderSampler__sampling_strategy=sampling_strategy,
                  LogitReg__C=C,
                  LogitReg__penalty=penalty)

#initialize grid search
clf_GS = GridSearchCV(pipeline, parameters,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=10)

startTime = time.time()

clf_GS.fit(X_train, Y_train)


print("Logistic Regression Grid Search Time: %d" % (time.time() - startTime))
print('Best Logit Regression Penalty:', clf_GS.best_estimator_.get_params()['LogitReg__penalty'])
print('Best Logit Regression C:', clf_GS.best_estimator_.get_params()['LogitReg__C'])
print('Best under sampling strategy:', clf_GS.best_estimator_.get_params()['RandomUnderSampler__sampling_strategy'])
print('n_components: ', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['LogitReg'])
print("Best Score: ", clf_GS.best_score_)
print("\n\n")


#fit on full training dataset
pipeline.set_params(**clf_GS.best_estimator_.get_params())
fitStart = time.time()
pipeline.fit(X_train,Y_train)
fitTime = time.time() - fitStart

#test on the test set
pred_logit = pipeline.predict(X_test)
#get probabilities for ROC curve
probs = pipeline.predict_proba(X_test)
#populate dictionary with scores
logit_score = getScores(pred_logit,Y_test)
logit_score["fit_time"] = fitTime

#calculate ROC Curve for later
logit_fpr, logit_tpr, logit_thresholds = metrics.roc_curve(Y_test, probs[:,1])


### SVM ###
#initialize SVM classifier
svc = svm.SVC()

#set up parameters
kernel = ['linear','poly','rbf']

degree = [3,4]


pipeline = Pipeline(steps= [("MinMax",MinMaxScaler()),
                            ("pca", PCA()),
                            ("RandomUnderSampler", RandomUnderSampler()),
                            ("RandomOverSampler", RandomOverSampler()),
                            ("svc",svc)])

parameters = dict(pca__n_components=n_components, 
                  RandomUnderSampler__sampling_strategy=sampling_strategy,
                  svc__kernel=kernel,
                  svc__degree=degree)

#initialize gridsearch
clf_GS = GridSearchCV(pipeline, parameters,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=10)

startTime=time.time()

clf_GS.fit(X_train, Y_train)

print("SVM Grid Search Time: %d" % (time.time() - startTime))
print('Best SVM kernel:', clf_GS.best_estimator_.get_params()['svc__kernel'])
print('Best SVM C:', clf_GS.best_estimator_.get_params()['svc__C'])
print('Best under sampling strategy:', clf_GS.best_estimator_.get_params()['RandomUnderSampler__sampling_strategy'])
print('n_components: ', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['svc'])
print("Best Score: ", clf_GS.best_score_)
print("\n\n")

#fit on the full training set
pipeline.set_params(**clf_GS.best_estimator_.get_params())
fitStart = time.time() 
pipeline.fit(X_train,Y_train)
fitTime = time.time() - fitStart

#test on the test set
pred_svm = pipeline.predict(X_test)
#get probabilities for ROC curve
probs = pipeline.decision_function(X_test)

#populate scores dictionary
svm_score = getScores(pred_svm,Y_test)
svm_score["fit_time"] = fitTime

#calculate ROC Curve
svm_fpr, svm_tpr, svm_thresholds = metrics.roc_curve(Y_test, probs)



### K Nearest Neighbors ###

#initialize classifier
knn = KNeighborsClassifier()

#set up parameters for serach
n_neighbors = [5,10,20,50]
weights = ['uniform','distance']

pipeline = Pipeline(steps= [("MinMax",MinMaxScaler()),
                            ("pca", PCA()),
                            ("RandomUnderSampler", RandomUnderSampler()),
                            ("RandomOverSampler", RandomOverSampler()),
                            ("knn",knn)])

parameters = dict(pca__n_components=n_components, 
                  RandomUnderSampler__sampling_strategy=sampling_strategy,
                  knn__n_neighbors=n_neighbors,
                  knn__weights=weights)

#initialize gridsearch
clf_GS = GridSearchCV(pipeline, parameters,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=1)

startTime=time.time()

clf_GS.fit(X_train, Y_train)

print("kNN Grid Search Time: %d" % (time.time() - startTime))
print('Best kNN Neighbors:', clf_GS.best_estimator_.get_params()['knn__n_neighbors'])
print('Best kNN Weight:', clf_GS.best_estimator_.get_params()['knn__weights'])
print('Best under sampling strategy:', clf_GS.best_estimator_.get_params()['RandomUnderSampler__sampling_strategy'])
print('n_components: ', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['knn'])
print("Best Score: ", clf_GS.best_score_)
print("\n\n")

#fit on full training set
pipeline.set_params(**clf_GS.best_estimator_.get_params())
fitStart = time.time()
pipeline.fit(X_train,Y_train)
fitTime = time.time() - fitStart

#test on test set
pred_knn = pipeline.predict(X_test)
#get probabilies for ROC curve
probs = pipeline.predict_proba(X_test)

#populate dictionary with scores
knn_score = getScores(pred_knn,Y_test)
knn_score["fit_time"] = fitTime

#calculate ROC Curve for later
knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(Y_test, probs[:,1])



### Decision Tree Classifier ###

#create pipeline for preprocessing and data fit/predict
pipeline = Pipeline(steps= [("MinMax",MinMaxScaler()),
                            ("pca", PCA()),
                            ("RandomUnderSampler", RandomUnderSampler()),
                            ("RandomOverSampler", RandomOverSampler()),
                            ("tree",DecisionTreeClassifier())])

#set up parameters for search
max_depth = [6,7,8,9]
min_samples_split = [10,100,500]
min_samples_leaf = [50,100,500]

parameters = dict(pca__n_components=n_components, 
                  RandomUnderSampler__sampling_strategy=sampling_strategy,
                  tree__max_depth=max_depth,
                  tree__min_samples_split=min_samples_split,
                  tree__min_samples_leaf=min_samples_leaf)



#initialize grid search
clf_GS = GridSearchCV(pipeline, parameters,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=1)

startTime = time.time()

clf_GS.fit(X_train, Y_train)

print("Decision Tree Search Time: %d" % (time.time() - startTime))
print('Best max_depth:', clf_GS.best_estimator_.get_params()['tree__max_depth'])
print('Best min_samples_split:', clf_GS.best_estimator_.get_params()['tree__min_samples_split'])
print('Best min_samples_lef:', clf_GS.best_estimator_.get_params()['tree__min_samples_leaf'])
print('Best under sampling strategy:', clf_GS.best_estimator_.get_params()['RandomUnderSampler__sampling_strategy'])
print('n_components: ', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['tree'])
print("Best Score: ", clf_GS.best_score_)
print("\n\n")

#train on the whole training set
pipeline.set_params(**clf_GS.best_estimator_.get_params())
fitStart = time.time()
pipeline.fit(X_train,Y_train)
fitTime = time.time() - fitStart
#test on the test set
pred_tree = pipeline.predict(X_test)
#get probabilities for ROC
probs = pipeline.predict_proba(X_test)

#populate scores dictionary
tree_score = getScores(pred_tree,Y_test)
tree_score["fit_time"] = fitTime

#calculate ROC Curve
tree_fpr, tree_tpr, tree_thresholds = metrics.roc_curve(Y_test, probs[:,1])


### MLP Neural Network ###

#create pipeline for preprocessing and data fit/predict
pipeline = Pipeline(steps= [("MinMax",MinMaxScaler()),
                            ("pca", PCA()),
                            ("RandomUnderSampler", RandomUnderSampler()),
                            ("RandomOverSampler", RandomOverSampler()),
                            ("mlp",MLPClassifier(solver='sgd'))])

#set up parameters for search
hidden_layer_sizes = [[20],[40],[40,5],[40,20,10],[20,10,5],[40,20,10,5]]
activation = ['relu','identity','logistic']
learning_rate = ['adaptive']
learning_rate_init = [0.1]
alpha=[0.001]
max_iter=[80]

parameters = dict(pca__n_components=n_components, 
                  RandomUnderSampler__sampling_strategy=sampling_strategy,
                  mlp__hidden_layer_sizes=hidden_layer_sizes,
                  mlp__activation=activation,
                  mlp__learning_rate=learning_rate,
                  mlp__learning_rate_init=learning_rate_init,
                  mlp__alpha=alpha,
                  mlp__max_iter=max_iter)



#initialize gridsearch
clf_GS = GridSearchCV(pipeline, parameters,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=1)

startTime = time.time()

clf_GS.fit(X_train, Y_train)

print("MLP Neural Network Grid Search Time: %d" % (time.time() - startTime))
print('Best hidden_layer_size:', clf_GS.best_estimator_.get_params()['mlp__hidden_layer_sizes'])
print('Best activation:', clf_GS.best_estimator_.get_params()['mlp__activation'])
print('Best learning_rate:', clf_GS.best_estimator_.get_params()['mlp__learning_rate'])
print('Best learning_rate_init:', clf_GS.best_estimator_.get_params()['mlp__learning_rate_init'])
print('Best alpha:', clf_GS.best_estimator_.get_params()['mlp__alpha'])
print('Best under sampling strategy:', clf_GS.best_estimator_.get_params()['RandomUnderSampler__sampling_strategy'])
print('n_components: ', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['mlp'])
print("Best Score: ", clf_GS.best_score_)
print("\n\n")

#train on the full training set
pipeline.set_params(**clf_GS.best_estimator_.get_params())
fitStart = time.time()
pipeline.fit(X_train,Y_train)
fitTime = time.time() - fitStart

#test on the test set
pred_mlp = pipeline.predict(X_test)
#get probabilities for ROC curve
probs = pipeline.predict_proba(X_test)

#populate scores dictionary
mlp_score = getScores(pred_mlp,Y_test)
mlp_score["fit_time"] = fitTime

#calculate ROC Curve for later
mlp_fpr, mlp_tpr, mlp_thresholds = metrics.roc_curve(Y_test, probs[:,1])



#print scores
print("Estimator \t Accuracy \t Hamming Loss \t F-Score \t Recall \t Precision \t ROC_AUC \t fit time")
print("Logit \t\t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (logit_score["accuracy"],logit_score["hamming"],logit_score["fscore"],logit_score["recall"],logit_score["precision"],logit_score["ROC_AUC"],logit_score["fit_time"]))
print("SVM \t\t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (svm_score["accuracy"],svm_score["hamming"],svm_score["fscore"],svm_score["recall"],svm_score["precision"],svm_score["ROC_AUC"],svm_score["fit_time"]))
print("kNN \t\t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (knn_score["accuracy"],knn_score["hamming"],knn_score["fscore"],knn_score["recall"],knn_score["precision"],knn_score["ROC_AUC"],knn_score["fit_time"]))
print("ANN \t\t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (mlp_score["accuracy"],mlp_score["hamming"],mlp_score["fscore"],mlp_score["recall"],mlp_score["precision"],mlp_score["ROC_AUC"],mlp_score["fit_time"]))
print("Decision Tree%0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (tree_score["accuracy"],tree_score["hamming"],tree_score["fscore"],tree_score["recall"],tree_score["precision"],tree_score["ROC_AUC"],tree_score["fit_time"]))



#plot ROC curves
plt.figure()
lw = 2
plt.plot(logit_fpr, logit_tpr, color='darkorange',
         lw=lw, label='Logistic Regression (%0.3f)' % logit_score["ROC_AUC"])
plt.plot(svm_fpr, svm_tpr, color='cyan',
         lw=lw, label='SVM (%0.3f)' % svm_score["ROC_AUC"])
plt.plot(knn_fpr, knn_tpr, color='navy',
         lw=lw, label='kNN (%0.3f)' % knn_score["ROC_AUC"])
plt.plot(mlp_fpr, mlp_tpr, color='olive',
         lw=lw, label='MLP Neural Network (%0.3f)' % mlp_score["ROC_AUC"])
plt.plot(tree_fpr, tree_tpr, color='grey',
         lw=lw, label='Decision Tree (%0.3f)' % tree_score["ROC_AUC"])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='no skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc.png',dpi=400,format='png')
plt.show()



#additional training epochs against the MLP Model


kf = KFold(n_splits=cv,shuffle=True,random_state=12)

#initialize dictionaries to capture data
train_accuracy = dict()
test_accuracy = dict()
train_hamming = dict()
test_hamming = dict()
train_f1 = dict()
test_f1 = dict()
train_time = dict()

n_epochs = 100

#initialize empty lists for training data
for i in range(n_epochs):
    train_accuracy[i] = []
    test_accuracy[i] = []
    train_hamming[i] = []
    test_hamming[i] = []
    train_f1[i] = []
    test_f1[i] = []
    train_time[i] = []

#k-fold cross validation
for train_index, test_index in kf.split(X_train.to_numpy()):
    #get out test splits
    K_train, K_test = X_train.to_numpy()[train_index], X_train.to_numpy()[test_index]
    L_train, L_test = Y_train.to_numpy()[train_index], Y_train.to_numpy()[test_index]
    #scale per best mlp estimator
    K_train=scaler.fit_transform(K_train)
    K_test=scaler.transform(K_test)
    #pca per best mlp estimator
    K_train=pca.fit_transform(K_train)
    K_test=pca.transform(K_test)
    #under and over sample strategy per best mlp estimator
    K_train_under, L_train_under = RandomUnderSampler(sampling_strategy=0.4).fit_resample(K_train,L_train)
    K_train_under, L_train_under = RandomOverSampler().fit_resample(K_train_under,L_train_under)
    #initalize classifier with parameters from gridsearch best estimator
    mlp = MLPClassifier(solver='sgd',hidden_layer_sizes=[20,10,5],alpha=0.001,
                        learning_rate_init=0.1,learning_rate='adaptive',max_iter=500,activation='logistic')
    
    c=1
    
    startTime=time.time()
    #partial fit for i epochs
    for i in range(n_epochs):
        print("Fold # %d of %d" % (c,cv))
        print("Running Epoch %d of %d" % (i,n_epochs))
        print("Current time: %d" %(time.time()-startTime))
        print("Avg time per epoch: %0.2f" %((time.time()-startTime)/(i+1)))
        c+=1
        #partial fit
        mlp.partial_fit(K_train_under,L_train_under.ravel(),classes=[0,1])
        
        #get results, fit time, predictions for each epoch
        fit_time = time.time()-startTime
        pred_train = mlp.predict(K_train_under)
        pred_test = mlp.predict(K_test)
        
        #add results to our dictionaries
        train_time[i].append(fit_time)
        train_accuracy[i].append(metrics.accuracy_score(L_train_under,pred_train))
        test_accuracy[i].append(metrics.accuracy_score(L_test,pred_test))
        train_hamming[i].append(metrics.hamming_loss(L_train_under,pred_train))
        test_hamming[i].append(metrics.hamming_loss(L_test,pred_test))
        train_f1[i].append(metrics.f1_score(L_train_under,pred_train,average='weighted'))
        test_f1[i].append(metrics.f1_score(L_test,pred_test,average='weighted'))
        

#initialize dictionaries with empty lists
train_accuracy["mean"] = []
train_accuracy["min"] = []
train_accuracy["max"] = []
test_accuracy["mean"] = []
test_accuracy["min"] = []
test_accuracy["max"] = []
train_hamming["mean"] = []
train_hamming["min"] = []
train_hamming["max"] = []
test_hamming["mean"] = []
test_hamming["min"] = []
test_hamming["max"] = []
train_f1["mean"] = []
train_f1["min"] = []
train_f1["max"] = []
test_f1["mean"] = []
test_f1["min"] = []
test_f1["max"] = []
train_time["mean"] = []
train_time["min"] = []
train_time["max"] = []


#append data to lists for each epoch
for i in range(n_epochs):
    train_accuracy["mean"].append(np.mean(train_accuracy[i]))
    train_accuracy["max"].append(max(train_accuracy[i]))
    train_accuracy["min"].append(min(train_accuracy[i]))
    test_accuracy["mean"].append(np.mean(test_accuracy[i]))
    test_accuracy["max"].append(max(test_accuracy[i]))
    test_accuracy["min"].append(min(test_accuracy[i]))
    train_hamming["mean"].append(np.mean(train_hamming[i]))
    train_hamming["min"].append(min(train_hamming[i]))
    train_hamming["max"].append(max(train_hamming[i]))
    test_hamming["mean"].append(np.mean(test_hamming[i]))
    test_hamming["min"].append(min(test_hamming[i]))
    test_hamming["max"].append(max(test_hamming[i]))
    train_f1["mean"].append(np.mean(train_f1[i]))
    train_f1["min"].append(min(train_f1[i]))
    train_f1["max"].append(max(train_f1[i]))
    test_f1["mean"].append(np.mean(test_f1[i]))
    test_f1["min"].append(min(test_f1[i]))
    test_f1["max"].append(max(test_f1[i]))
    train_time["mean"].append(np.mean(train_time[i]))
    train_time["min"].append(min(train_time[i]))
    train_time["max"].append(max(train_time[i]))
    
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

x = list(range(1,n_epochs+1))

ax.plot(x,test_hamming["mean"],color='orange',label="Validation")

ax.fill_between(x=x,y1=test_hamming["max"],y2=test_hamming["min"],color='orange',alpha=.3)

ax.plot(x,train_hamming["mean"],color='blue',label="Training")

ax.fill_between(x=x,y1=train_hamming["max"],y2=train_hamming["min"],color='blue',alpha=.3)


ax.set_xlabel("Epochs", fontsize=15)
ax.set_ylabel("Hamming Loss (Error)", fontsize=18)

ax.legend(loc="best")
plt.savefig('validation_curve.png',dpi=400,format='png')
fig.show()

#plot validation accuracy and loss per epoch
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(x,test_f1["mean"],color='orange',label="Validation Weighted F-Score")

ax.fill_between(x=x,y1=test_f1["max"],y2=test_f1["min"],color='orange',alpha=.3)

ax.plot(x,test_hamming["mean"],color='blue',label="Validation Hamming Loss")

ax.fill_between(x=x,y1=test_hamming["max"],y2=test_hamming["min"],color='blue',alpha=.3)


ax.set_xlabel("Epochs", fontsize=15)
ax.set_ylabel("Score", fontsize=18)

ax.legend(loc="best")
plt.savefig('f_curve.png',dpi=400,format='png')
fig.show()


## fit against final model

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#pca per best mlp estimator
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
#under and over sample strategy per best mlp estimator
X_train, Y_train = RandomUnderSampler(sampling_strategy=0.4).fit_resample(X_train,Y_train)
X_train, Y_train = RandomOverSampler().fit_resample(X_train,Y_train)
#initalize classifier with parameters from gridsearch best estimator

mlp = MLPClassifier(solver='sgd',hidden_layer_sizes=[20,10,5],alpha=0.001,
                        learning_rate_init=0.1,learning_rate='adaptive',max_iter=n_epochs,activation='logistic')

mlp.fit(X_train,Y_train)

prediction_train = mlp.predict(X_train)
prediction_test = mlp.predict(X_test)
proba_test = mlp.predict_proba(X_test)

final_score = getScores(prediction_test,Y_test)

#print scores
print("Estimator \t Accuracy \t Hamming Loss \t F-Score \t Recall \t Precision \t ROC_AUC")
print("Final MLP \t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f" % (final_score["accuracy"],final_score["hamming"],final_score["fscore"],final_score["recall"],final_score["precision"],final_score["ROC_AUC"]))
print("MLP Grid  \t %0.3f\t\t %0.3f\t\t\t %0.3f\t\t %0.3f\t\t %0.3f\t\t %0.3f \t\t %d" % (mlp_score["accuracy"],mlp_score["hamming"],mlp_score["fscore"],mlp_score["recall"],mlp_score["precision"],mlp_score["ROC_AUC"],mlp_score["fit_time"]))


final_fpr, final_tpr, final_thresholds = metrics.roc_curve(Y_test, proba_test[:,1])


#plot ROC curves
plt.figure()
lw = 2
plt.plot(logit_fpr, logit_tpr, color='darkorange',
         lw=lw, label='Logistic Regression (%0.3f)' % logit_score["ROC_AUC"])
plt.plot(svm_fpr, svm_tpr, color='cyan',
         lw=lw, label='SVM (%0.3f)' % svm_score["ROC_AUC"])
plt.plot(knn_fpr, knn_tpr, color='navy',
         lw=lw, label='kNN (%0.3f)' % knn_score["ROC_AUC"])
plt.plot(mlp_fpr, mlp_tpr, color='olive',
         lw=lw, label='MLP Neural Network (%0.3f)' % mlp_score["ROC_AUC"])
plt.plot(tree_fpr, tree_tpr, color='grey',
         lw=lw, label='Decision Tree (%0.3f)' % tree_score["ROC_AUC"])
plt.plot(final_fpr, final_tpr, color='red',
         lw=lw, label='Final MLP Neural Network (%0.3f)' % final_score["ROC_AUC"])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='no skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_final.png',dpi=400,format='png')
plt.show()
