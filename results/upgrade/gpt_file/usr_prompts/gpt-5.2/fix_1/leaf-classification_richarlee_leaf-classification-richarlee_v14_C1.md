# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.5

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Target score

1.39277

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import time
import pandas as pd
import numpy as np
from pandas import DataFrame,Series

from sklearn import linear_model, cross_validation, feature_selection, manifold, decomposition, random_projection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier,BaggingRegressor,RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3699494449.py in <cell line: 0>()
      4 from pandas import DataFrame,Series
      5 
----> 6 from sklearn import linear_model, cross_validation, feature_selection, manifold, decomposition, random_projection
      7 from sklearn.preprocessing import MinMaxScaler,LabelEncoder
      8 from sklearn.ensemble import GradientBoostingClassifier,BaggingRegressor,RandomForestClassifier

ImportError: cannot import name 'cross_validation' from 'sklearn' (/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py)

## === cell 1
train_df = pd.read_csv('../input/train.csv')
train_df.fillna(0,inplace=True)
train_df


## === cell 2
le = LabelEncoder().fit(train_df.species)
labels = le.transform(train_df.species)
labels


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1676018832.py in <cell line: 0>()
----> 1 le = LabelEncoder().fit(train_df.species)
      2 labels = le.transform(train_df.species)
      3 labels

NameError: name 'LabelEncoder' is not defined

## === cell 3
df = train_df.copy()
df.species = labels
df.species


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1212218038.py in <cell line: 0>()
      1 df = train_df.copy()
----> 2 df.species = labels
      3 df.species

NameError: name 'labels' is not defined

## === cell 4
df.ix[:,2:] = MinMaxScaler().fit_transform(train_df.ix[:,2:])
df


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3195526557.py in <cell line: 0>()
----> 1 df.ix[:,2:] = MinMaxScaler().fit_transform(train_df.ix[:,2:])
      2 df

NameError: name 'MinMaxScaler' is not defined

## === cell 5
X = df.as_matrix()[:,2:]
y = df.as_matrix()[:,1]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2197276210.py in <cell line: 0>()
----> 1 X = df.as_matrix()[:,2:]
      2 y = df.as_matrix()[:,1]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'as_matrix'

## === cell 6
def plot_learning_curve(estimator, title, X, y, ylim=(0.4,1.1), cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    start_time = time.time()
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.annotate(test_scores_mean[-1],xy=(train_sizes[-1],test_scores_mean[-1]))

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title+'(time=%fs)'%(time.time()-start_time))
    plt.show()


## === cell 7
rf = RandomForestClassifier(n_estimators=5)
plot_learning_curve(rf,'RandomForestClassifier',X,y)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1517807747.py in <cell line: 0>()
----> 1 rf = RandomForestClassifier(n_estimators=5)
      2 plot_learning_curve(rf,'RandomForestClassifier',X,y)

NameError: name 'RandomForestClassifier' is not defined

## === cell 8
lr = linear_model.LogisticRegression()
plot_learning_curve(lr,'LogisticRegression',X,y)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358501290.py in <cell line: 0>()
      1 lr = linear_model.LogisticRegression()
----> 2 plot_learning_curve(lr,'LogisticRegression',X,y)

NameError: name 'X' is not defined

## === cell 9
lsvc = LinearSVC()
plot_learning_curve(lsvc,'LinearSVC',X,y)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3600274842.py in <cell line: 0>()
----> 1 lsvc = LinearSVC()
      2 plot_learning_curve(lsvc,'LinearSVC',X,y)

NameError: name 'LinearSVC' is not defined

## === cell 10
svc = SVC(probability=True,kernel='rbf',C=0.025)
plot_learning_curve(svc,'SVC',X,y)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/153997018.py in <cell line: 0>()
----> 1 svc = SVC(probability=True,kernel='rbf',C=0.025)
      2 plot_learning_curve(svc,'SVC',X,y)

NameError: name 'SVC' is not defined

## === cell 11
lda = LinearDiscriminantAnalysis()
plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/931187123.py in <cell line: 0>()
----> 1 lda = LinearDiscriminantAnalysis()
      2 plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)

NameError: name 'LinearDiscriminantAnalysis' is not defined

## === cell 12
def cal_log_loss(estimator):
    loss = []
    split = StratifiedKFold(y)
    for train_index, test_index in split:
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        estimator.fit(X_train,y_train)
        y_prediction = estimator.predict_proba(X_test)
        loss.append(log_loss(y_test,y_prediction))
    return np.mean(loss)
    


## === cell 13
cal_log_loss(lr)
cal_log_loss(rf)
cal_log_loss(lda)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4247190043.py in <cell line: 0>()
----> 1 cal_log_loss(lr)
      2 cal_log_loss(rf)
      3 cal_log_loss(lda)

/tmp/ipykernel_11/660066412.py in cal_log_loss(estimator)
      1 def cal_log_loss(estimator):
      2     loss = []
----> 3     split = StratifiedKFold(y)
      4     for train_index, test_index in split:
      5         X_train,X_test = X[train_index],X[test_index]

NameError: name 'StratifiedKFold' is not defined

## === cell 14
estimator = lda
estimator.fit(X,y)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/916630180.py in <cell line: 0>()
----> 1 estimator = lda
      2 estimator.fit(X,y)

NameError: name 'lda' is not defined

## === cell 15
test_data = pd.read_csv('../input/test.csv')
test_df = DataFrame(MinMaxScaler().fit_transform(test_data.ix[:,1:]))
test_df


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3624648946.py in <cell line: 0>()
      1 test_data = pd.read_csv('../input/test.csv')
----> 2 test_df = DataFrame(MinMaxScaler().fit_transform(test_data.ix[:,1:]))
      3 test_df

NameError: name 'MinMaxScaler' is not defined

## === cell 16
predict = estimator.predict_proba(test_df.as_matrix())
result = DataFrame(predict,columns=train_df.species.unique().sort())
result


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4222214833.py in <cell line: 0>()
----> 1 predict = estimator.predict_proba(test_df.as_matrix())
      2 result = DataFrame(predict,columns=train_df.species.unique().sort())
      3 result
      4 # print(predict)
      5 # decision = estimator.decision_function(test_df.as_matrix())

NameError: name 'estimator' is not defined

## === cell 17
result.insert(0,'id',test_data.id)
result.to_csv('result.csv',index=False)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626270204.py in <cell line: 0>()
----> 1 result.insert(0,'id',test_data.id)
      2 result.to_csv('result.csv',index=False)

NameError: name 'result' is not defined

## === cell 19
import time
import pandas as pd
import numpy as np
from pandas import DataFrame,Series

from sklearn import linear_model, cross_validation, feature_selection, manifold, decomposition, random_projection
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier,BaggingRegressor,RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3699494449.py in <cell line: 0>()
      4 from pandas import DataFrame,Series
      5 
----> 6 from sklearn import linear_model, cross_validation, feature_selection, manifold, decomposition, random_projection
      7 from sklearn.preprocessing import MinMaxScaler,LabelEncoder
      8 from sklearn.ensemble import GradientBoostingClassifier,BaggingRegressor,RandomForestClassifier

ImportError: cannot import name 'cross_validation' from 'sklearn' (/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py)

## === cell 20
train_df = pd.read_csv('../input/train.csv')
train_df.fillna(0,inplace=True)
train_df


## === cell 21
le = LabelEncoder().fit(train_df.species)
labels = le.transform(train_df.species)
labels


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1676018832.py in <cell line: 0>()
----> 1 le = LabelEncoder().fit(train_df.species)
      2 labels = le.transform(train_df.species)
      3 labels

NameError: name 'LabelEncoder' is not defined

## === cell 22
df = train_df.copy()
df.species = labels
df.species


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1212218038.py in <cell line: 0>()
      1 df = train_df.copy()
----> 2 df.species = labels
      3 df.species

NameError: name 'labels' is not defined

## === cell 23
df.ix[:,2:] = MinMaxScaler().fit_transform(train_df.ix[:,2:])
df


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3195526557.py in <cell line: 0>()
----> 1 df.ix[:,2:] = MinMaxScaler().fit_transform(train_df.ix[:,2:])
      2 df

NameError: name 'MinMaxScaler' is not defined

## === cell 24
X = df.as_matrix()[:,2:]
y = df.as_matrix()[:,1]


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2197276210.py in <cell line: 0>()
----> 1 X = df.as_matrix()[:,2:]
      2 y = df.as_matrix()[:,1]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'as_matrix'

## === cell 25
def plot_learning_curve(estimator, title, X, y, ylim=(0.4,1.1), cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    start_time = time.time()
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.annotate(test_scores_mean[-1],xy=(train_sizes[-1],test_scores_mean[-1]))

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title+'(time=%fs)'%(time.time()-start_time))
    plt.show()


## === cell 26
rf = RandomForestClassifier(n_estimators=5)
plot_learning_curve(rf,'RandomForestClassifier',X,y)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1517807747.py in <cell line: 0>()
----> 1 rf = RandomForestClassifier(n_estimators=5)
      2 plot_learning_curve(rf,'RandomForestClassifier',X,y)

NameError: name 'RandomForestClassifier' is not defined

## === cell 27
lr = linear_model.LogisticRegression()
plot_learning_curve(lr,'LogisticRegression',X,y)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358501290.py in <cell line: 0>()
      1 lr = linear_model.LogisticRegression()
----> 2 plot_learning_curve(lr,'LogisticRegression',X,y)

NameError: name 'X' is not defined

## === cell 28
lsvc = LinearSVC()
plot_learning_curve(lsvc,'LinearSVC',X,y)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3600274842.py in <cell line: 0>()
----> 1 lsvc = LinearSVC()
      2 plot_learning_curve(lsvc,'LinearSVC',X,y)

NameError: name 'LinearSVC' is not defined

## === cell 29
svc = SVC(probability=True,kernel='rbf',C=0.025)
plot_learning_curve(svc,'SVC',X,y)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/153997018.py in <cell line: 0>()
----> 1 svc = SVC(probability=True,kernel='rbf',C=0.025)
      2 plot_learning_curve(svc,'SVC',X,y)

NameError: name 'SVC' is not defined

## === cell 30
lda = LinearDiscriminantAnalysis()
plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/931187123.py in <cell line: 0>()
----> 1 lda = LinearDiscriminantAnalysis()
      2 plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)

NameError: name 'LinearDiscriminantAnalysis' is not defined

## === cell 31
def cal_log_loss(estimator):
    loss = []
    split = StratifiedKFold(y)
    for train_index, test_index in split:
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        estimator.fit(X_train,y_train)
        y_prediction = estimator.predict_proba(X_test)
        loss.append(log_loss(y_test,y_prediction))
    return np.mean(loss)
    


## === cell 32
cal_log_loss(lr)
cal_log_loss(rf)
cal_log_loss(lda)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4247190043.py in <cell line: 0>()
----> 1 cal_log_loss(lr)
      2 cal_log_loss(rf)
      3 cal_log_loss(lda)

/tmp/ipykernel_11/660066412.py in cal_log_loss(estimator)
      1 def cal_log_loss(estimator):
      2     loss = []
----> 3     split = StratifiedKFold(y)
      4     for train_index, test_index in split:
      5         X_train,X_test = X[train_index],X[test_index]

NameError: name 'StratifiedKFold' is not defined

## === cell 33
estimator = lda
estimator.fit(X,y)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/916630180.py in <cell line: 0>()
----> 1 estimator = lda
      2 estimator.fit(X,y)

NameError: name 'lda' is not defined

## === cell 34
test_data = pd.read_csv('../input/test.csv')
test_df = DataFrame(MinMaxScaler().fit_transform(test_data.ix[:,1:]))
test_df


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3624648946.py in <cell line: 0>()
      1 test_data = pd.read_csv('../input/test.csv')
----> 2 test_df = DataFrame(MinMaxScaler().fit_transform(test_data.ix[:,1:]))
      3 test_df

NameError: name 'MinMaxScaler' is not defined

## === cell 35
species = train_df.species.unique()
species.sort()

predict = estimator.predict_proba(test_df.as_matrix())
result = DataFrame(predict,columns=species)
result


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2648019491.py in <cell line: 0>()
      2 species.sort()
      3 
----> 4 predict = estimator.predict_proba(test_df.as_matrix())
      5 result = DataFrame(predict,columns=species)
      6 result

NameError: name 'estimator' is not defined

## === cell 36
result.insert(0,'id',test_data.id)
result.to_csv('result.csv',index=False)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626270204.py in <cell line: 0>()
----> 1 result.insert(0,'id',test_data.id)
      2 result.to_csv('result.csv',index=False)

NameError: name 'result' is not defined
