# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_train= pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')
df_test = pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')


## === cell 2
df_train.info()


## === cell 3
df_test.info()


## === cell 4
df_train.head(10)


## === cell 5
df_train.describe()


## === cell 6
df_test.describe()


## === cell 7
df_train.isnull().sum()


## === cell 8
df_test.isnull().sum()


## === cell 9
print(df_train.shape)
print(df_test.shape)


## === cell 10
df_train.duplicated().sum()


## === cell 11
df_train['species'].nunique()


## === cell 12
df_train.columns.values


## === cell 13
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit # we will know about that package while using it


## === cell 14
def encode(df_train,df_test):
    le = LabelEncoder().fit(df_train.species)
    labels = le.transform(df_train.species) # Species are in stings 
    
    classes = list(le.classes_) #creating list of column names for submission
    
    test_ids = df_test.id # creating variable for IDs
    
    df_train = df_train.drop(['species','id'],axis = 1) #droping columns 
    df_test = df_test.drop(['id'],axis = 1)

    return df_train, labels, classes, test_ids, df_test

df_train, labels, classes, test_ids, df_test = encode(df_train,df_test)


## === cell 15
df_train.head()


## === cell 16
df_train.shape


## === cell 17
labels


## === cell 18
X = df_train.values
y = labels


## === cell 19
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=5)
>>> sss.get_n_splits(X, y)


## === cell 20
sss


## === cell 21
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


## === cell 22

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel='rbf',C = 0.025, probability= True),
    NuSVC(probability = True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]


## === cell 23
log_cols=["Classifier", "Accuracy", "Log Loss"] # created list 
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train) # fit method for applying algorithm 
    name = clf.__class__.__name__
    
    print("="*30) ### printing = in 30 times
    print(name) ### printing name of algorithm
    
    print('****Results****') 
    train_predictions = clf.predict(X_test)   ### predicting y_predict
    acc = accuracy_score(y_test, train_predictions) # now compairing y predict(train_prediction) with y_test
    print("Accuracy: {:.4%}".format(acc)) ### printing accuracy  with 4 decimal with pecentage mark 
    
    train_predictions = clf.predict_proba(X_test) ### Probability estimates y_predict_probability(train_predictions)
    ll = log_loss(y_test, train_predictions) ### appllying log loss on y_test and y_predict_probability(train_predictions)
    print("Log Loss: {}".format(ll)) ### printing log loss  
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols) ## creating new dataframe with 
    log = log.append(log_entry) ### append those entry in log 
    
print("="*30)
    


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2326765306.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m                                                                           [0;31m## accuracy score(this score not in pecentage form hence it multiply by 100)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m                                                                             [0;31m##logloss stored each iteratin value in log_cols[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m     [0mlog[0m [0;34m=[0m [0mlog[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mlog_entry[0m[0;34m)[0m [0;31m### append those entry in log[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;34m[0m[0m
[1;32m     27[0m [0mprint[0m[0;34m([0m[0;34m"="[0m[0;34m*[0m[0;36m30[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m   6297[0m         ):
[1;32m   6298[0m             [0;32mreturn[0m [0mself[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6299[0;31m         [0;32mreturn[0m [0mobject[0m[0;34m.[0m[0m__getattribute__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6300[0m [0;34m[0m[0m
[1;32m   6301[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DataFrame' object has no attribute 'append'

## === cell 24
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_color_codes("muted")
sns.barplot(y='Classifier', x='Accuracy', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
