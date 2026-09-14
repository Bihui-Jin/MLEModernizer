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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.93237

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import time
from datetime import datetime

start_time = time.time()

%matplotlib inline

import os, warnings
import numpy as np 
from numpy.random import seed
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, precision_score,recall_score, f1_score, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from category_encoders import MEstimateEncoder

sns.set(style='white', context='notebook', palette='deep', rc={'figure.figsize':(10,8)})
print("loaded ...")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def set_seed(sd):
    seed(sd)
    np.random.seed(sd)
    tf.random.set_seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
RandomSeed = 13
set_seed(RandomSeed)


## === cell 2
train_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
test_data['target'] = -1
train_data['Set'] = "Train"
test_data['Set'] = "Test"
DATA = train_data.append(test_data)
DATA.reset_index(inplace=True)
DATA.info()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3765626029.py in <cell line: 0>()
      4 train_data['Set'] = "Train"
      5 test_data['Set'] = "Test"
----> 6 DATA = train_data.append(test_data)
      7 DATA.reset_index(inplace=True)
      8 DATA.info()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 3
features = [f for f in DATA.columns if "f_" in f]
float_features = [f for f in features if DATA[f].dtype == "float64"]
int_features = [f for f in features if DATA[f].dtype == "int64"]
str_features = [f for f in features if DATA[f].dtype == "object"]


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3532589121.py in <cell line: 0>()
----> 1 features = [f for f in DATA.columns if "f_" in f]
      2 float_features = [f for f in features if DATA[f].dtype == "float64"]
      3 int_features = [f for f in features if DATA[f].dtype == "int64"]
      4 str_features = [f for f in features if DATA[f].dtype == "object"]

NameError: name 'DATA' is not defined

## === cell 4
DATA[int_features].describe()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1619592972.py in <cell line: 0>()
----> 1 DATA[int_features].describe()

NameError: name 'DATA' is not defined

## === cell 5
DATA[float_features].describe()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4180166295.py in <cell line: 0>()
----> 1 DATA[float_features].describe()

NameError: name 'DATA' is not defined

## === cell 6
DATA[str_features].head()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/789001011.py in <cell line: 0>()
----> 1 DATA[str_features].head()

NameError: name 'DATA' is not defined

## === cell 7
DATA['NumUnique'] = DATA['f_27'].apply(lambda row: len(set(row)))
features_from_string = ['NumUnique']
features_from_string


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/397384406.py in <cell line: 0>()
----> 1 DATA['NumUnique'] = DATA['f_27'].apply(lambda row: len(set(row)))
      2 features_from_string = ['NumUnique']
      3 features_from_string

NameError: name 'DATA' is not defined

## === cell 8
import string
letters = list(string.ascii_uppercase)[:20]

for L in letters:
    DATA['count_'+L] = DATA['f_27'].str.count(L)
features_from_string.extend([c for c in DATA.columns if c.startswith("count_")])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1649934977.py in <cell line: 0>()
      3 
      4 for L in letters:
----> 5     DATA['count_'+L] = DATA['f_27'].str.count(L)
      6 features_from_string.extend([c for c in DATA.columns if c.startswith("count_")])

NameError: name 'DATA' is not defined

## === cell 9
scaler = MinMaxScaler()
DATA[[*float_features,*int_features,*features_from_string]] = scaler.fit_transform(DATA[[*float_features,*int_features,*features_from_string]])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1988326256.py in <cell line: 0>()
      1 scaler = MinMaxScaler()
----> 2 DATA[[*float_features,*int_features,*features_from_string]] = scaler.fit_transform(DATA[[*float_features,*int_features,*features_from_string]])

NameError: name 'DATA' is not defined

## === cell 10
def plot_hist(features, title):
    N_cols = 4
    col_width = 8
    N_rows = round(len(features) / N_cols + 0.49)
    fig, axs = plt.subplots(nrows = N_rows, ncols=N_cols, figsize=(col_width * N_cols, N_rows * col_width))
    for i,f in enumerate(features):
        axs[i//N_cols, i%N_cols].hist(DATA[DATA.Set == 'Train'][f], bins=20);
        axs[i//N_cols, i%N_cols].set_title(f)
        axs[i//N_cols, i%N_cols].legend();


## === cell 11
%%time
plot_hist(float_features, 'float_features')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'float_features' is not defined

## === cell 12
%%time
plot_hist(int_features, 'int_features')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'int_features' is not defined

## === cell 13
fig, ax = plt.subplots(figsize=(30,30)) 
ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*int_features, *features_from_string,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
ax.set_title("Target -  correlation to int features");


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1132681083.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(30,30))
----> 2 ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*int_features, *features_from_string,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
      3 ax.set_title("Target -  correlation to int features");

NameError: name 'DATA' is not defined

## === cell 14
fig, ax = plt.subplots(figsize=(20,20)) 
ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*float_features,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
ax.set_title("Target -  correlation to float features");


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2236855742.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20,20))
----> 2 ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*float_features,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
      3 ax.set_title("Target -  correlation to float features");

NameError: name 'DATA' is not defined

## === cell 15
TRAIN = DATA[DATA.Set == 'Train']
TEST = DATA[DATA.Set == 'Test'][[*float_features, *int_features,*features_from_string]]
X = TRAIN[[*float_features, *int_features, *features_from_string]]
y = TRAIN.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = RandomSeed, stratify=y)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1896595111.py in <cell line: 0>()
----> 1 TRAIN = DATA[DATA.Set == 'Train']
      2 TEST = DATA[DATA.Set == 'Test'][[*float_features, *int_features,*features_from_string]]
      3 X = TRAIN[[*float_features, *int_features, *features_from_string]]
      4 y = TRAIN.target
      5 X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = RandomSeed, stratify=y)

NameError: name 'DATA' is not defined

## === cell 16
def CM(y_test, val_pred, title):
    labels = [0,1]
    cm = confusion_matrix(y_test, val_pred, normalize = 'pred')
    cm_train = confusion_matrix(y_train, train_pred, normalize = 'pred')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels= labels);
    disp_train.plot(ax=ax1, values_format='.1%', xticks_rotation='horizontal');
    disp_train.ax_.set_title('Train set', {'fontsize':20});

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels);
    disp_test.plot(ax=ax2, values_format='.1%', xticks_rotation='horizontal');
    disp_test.ax_.set_title('Validation set',{'fontsize':20});
    fig.suptitle(title, fontsize=16);
    
def IMP(model, label, columns = X_train.columns):
    features = {}
    for feature, importance in zip(columns, model.feature_importances_):
        features[feature] = importance

    importances = pd.DataFrame({label:features})
    importances.sort_values(label, ascending = True, inplace=True)
    importances[:10].plot.barh()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/851898707.py in <cell line: 0>()
     13     fig.suptitle(title, fontsize=16);
     14 
---> 15 def IMP(model, label, columns = X_train.columns):
     16     features = {}
     17     for feature, importance in zip(columns, model.feature_importances_):

NameError: name 'X_train' is not defined

## === cell 17
rf_params = {
    'n_jobs':-1,
    'random_state': RandomSeed,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'auto',
    'max_samples': None
}


## === cell 18
rf_grid = {
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf':[1],
    'max_features': ['auto'],
    'max_samples': [None, 0.9],
    'n_estimators': [100],
}


## === cell 21
%%time
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)
rf_train_score = rf_model.score(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print("Train: {:.2f} %".format(rf_train_score * 100))
print("Test: {:.2f} %".format(rf_accuracy*100))
print('Overfit: {:.2f} %'.format((rf_train_score-rf_accuracy)*100))


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'X_train' is not defined

## === cell 22
%%time
train_pred = rf_model.predict(X_train)
val_pred = rf_model.predict(X_test)
print(classification_report(y_test, val_pred))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'X_train' is not defined

## === cell 23
CM(y_test, val_pred, 'Random Forest Classifier')


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3195623170.py in <cell line: 0>()
----> 1 CM(y_test, val_pred, 'Random Forest Classifier')

NameError: name 'y_test' is not defined

## === cell 24
IMP(rf_model, "RF")


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2679139917.py in <cell line: 0>()
----> 1 IMP(rf_model, "RF")

NameError: name 'IMP' is not defined

## === cell 25
print("Accuracy Scores:")
print("==========================================================")
print("RandomForest: {:.3f}".format(rf_accuracy))
print("==========================================================")


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/928622157.py in <cell line: 0>()
      2 print("==========================================================")
      3 # print("DNN: {:.3f}".format(dnn_accuracy))
----> 4 print("RandomForest: {:.3f}".format(rf_accuracy))
      5 # print("XGBoost classifier: {:.3f}".format(xgb_accuracy))
      6 # print("SVM classifier: {:.3f}".format(SVM_accuracy))

NameError: name 'rf_accuracy' is not defined

## === cell 26
models = [rf_model]
model_names = ["RF"]
print("using", len(models), "classifiers")


## === cell 27
%%time
SVC_ALL_PREDICTIONS = pd.DataFrame({'id': DATA[DATA.Set == 'Test']['id']})
for i, m in enumerate(models):
    SVC_ALL_PREDICTIONS[model_names[i]] = m.predict_proba(TEST)[:,1]
SVC_ALL_PREDICTIONS['MedianVote'] = SVC_ALL_PREDICTIONS[model_names].median(axis=1)
SVC_ALL_PREDICTIONS['SoftVote'] = SVC_ALL_PREDICTIONS[model_names].mean(axis=1)
svc_predictions = SVC_ALL_PREDICTIONS.SoftVote
SVC_ALL_PREDICTIONS.head(10)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'DATA' is not defined

## === cell 29
output = pd.DataFrame({'id': DATA[DATA.Set == 'Test']['id'], 'target': svc_predictions})
output.head(10)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1520321503.py in <cell line: 0>()
----> 1 output = pd.DataFrame({'id': DATA[DATA.Set == 'Test']['id'], 'target': svc_predictions})
      2 output.head(10)

NameError: name 'DATA' is not defined

## === cell 30
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2007446378.py in <cell line: 0>()
      1 #output
----> 2 output.to_csv('submission.csv', index=False)
      3 print("Submission was successfully saved!")

NameError: name 'output' is not defined

## === cell 31
end_time = time.time()
print("Notebook run time: {:.1f} seconds. Finished at {}".format(end_time - start_time, datetime.now()) )
