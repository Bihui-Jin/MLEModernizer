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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
joblib==1.5.2
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
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.0508228306890561

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE # for over-sampling
import joblib # for saving models
import warnings
warnings.filterwarnings('ignore')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2910630633.py in <cell line: 0>()
      8 from sklearn.naive_bayes import MultinomialNB
      9 from sklearn.metrics import classification_report
---> 10 from imblearn.over_sampling import SMOTE # for over-sampling
     11 import joblib # for saving models
     12 import warnings

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 2
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip

## === cell 3
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## === cell 35
df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/161715770.py in <cell line: 0>()
----> 1 df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
      2 df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

NameError: name 'joblib' is not defined

## === cell 36
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
      print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## === cell 37
df=reduce_mem_usage(df)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/164482069.py in <cell line: 0>()
----> 1 df=reduce_mem_usage(df)

NameError: name 'df' is not defined

## === cell 38
df_test=reduce_mem_usage(df_test)

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4231178207.py in <cell line: 0>()
----> 1 df_test=reduce_mem_usage(df_test)

NameError: name 'df_test' is not defined

## === cell 39
df.head()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined

## === cell 40
df.isnull().sum()

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/286341606.py in <cell line: 0>()
----> 1 df.isnull().sum()

NameError: name 'df' is not defined

## === cell 41
df_test.head()

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685298186.py in <cell line: 0>()
----> 1 df_test.head()

NameError: name 'df_test' is not defined

## === cell 42
df_test.isnull().sum()

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/65500498.py in <cell line: 0>()
----> 1 df_test.isnull().sum()

NameError: name 'df_test' is not defined

## === cell 45
fig,axes=plt.subplots(3,2,figsize=(15,15))

for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
    ax.set_title('{} Distribution'.format(class_name))
    ax.set_xticks(range(2),[0,1])
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')

plt.show()

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/686868683.py in <cell line: 0>()
      2 
      3 for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
----> 4     pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
      5     ax.set_title('{} Distribution'.format(class_name))
      6     ax.set_xticks(range(2),[0,1])

NameError: name 'df' is not defined

## === cell 48
from sklearn.feature_extraction.text import TfidfVectorizer

## === cell 49
vec = TfidfVectorizer(max_features=1000, stop_words='english',analyzer='word')

## === cell 50
tfidf = vec.fit_transform(df['lemmatized'])
tfidf.shape

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/375799090.py in <cell line: 0>()
      1 # create TFIDF for train
----> 2 tfidf = vec.fit_transform(df['lemmatized'])
      3 # shape of TFIDF
      4 tfidf.shape

NameError: name 'df' is not defined

## === cell 51
tfidf

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294693912.py in <cell line: 0>()
----> 1 tfidf

NameError: name 'tfidf' is not defined

## === cell 52
tfidf_test = vec.transform(df_test['lemmatized'])

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781774660.py in <cell line: 0>()
      1 # create TFIDF for test
----> 2 tfidf_test = vec.transform(df_test['lemmatized'])

NameError: name 'df_test' is not defined

## === cell 53
tfidf_test

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3198579238.py in <cell line: 0>()
----> 1 tfidf_test

NameError: name 'tfidf_test' is not defined

## === cell 55
X=tfidf.toarray()
X_test=tfidf_test.toarray()
target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
X

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/541843002.py in <cell line: 0>()
      1 #Preparing Dataset for Modeling
----> 2 X=tfidf.toarray()
      3 X_test=tfidf_test.toarray()
      4 target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
      5 X

NameError: name 'tfidf' is not defined

## === cell 56
prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
prob['id']=df_test['id']

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668213629.py in <cell line: 0>()
      1 #Dataframe for final probabilties
----> 2 prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
      3 prob['id']=df_test['id']

NameError: name 'df_test' is not defined

## === cell 57
del df,tfidf,tfidf_test,vec,df_test

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4156608415.py in <cell line: 0>()
----> 1 del df,tfidf,tfidf_test,vec,df_test

NameError: name 'df' is not defined

## === cell 58
model=[]

for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    print('{} - Model:\n'.format(value))
    
    y=target[:,index]
    print('Before Over-Sampling [{}] dataset shape=>({},{})'.format(value,X.shape,y.shape))
    
    smk=SMOTE(random_state=42,n_jobs=4)
    X_temp,y=smk.fit_sample(X,y)
    print('After Over-Sampling [{}] dataset shape=>({},{})'.format(value,X_temp.shape,y.shape))
    
    x_train, x_test, y_train, y_test = train_test_split(X_temp, y,stratify=y, test_size=0.2, random_state=42)
    test_model=MultinomialNB()
    test_model=test_model.fit(x_train,y_train)
    train_pred=test_model.predict(x_train)
    print('In-sample Evaluation:\n',classification_report(y_train,train_pred))
    test_pred=test_model.predict(x_test)
    print('Out-sample Evaluation\n',classification_report(y_test,test_pred))
    
    model.append(MultinomialNB())
    model[index] = model[index].fit(X_temp, y)
    prob[value]=model[index].predict_proba(X_test)
    
    joblib.dump(model[index],'{}-model.pkl'.format(value))

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3484252843.py in <cell line: 0>()
      5     print('{} - Model:\n'.format(value))
      6 
----> 7     y=target[:,index]
      8     print('Before Over-Sampling [{}] dataset shape=>({},{})'.format(value,X.shape,y.shape))
      9 

NameError: name 'target' is not defined

## === cell 59
prob

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1085545036.py in <cell line: 0>()
----> 1 prob

NameError: name 'prob' is not defined

## === cell 60
prob.to_csv('submission-NB-tfidf-smote.csv',index=False)

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3701483399.py in <cell line: 0>()
----> 1 prob.to_csv('submission-NB-tfidf-smote.csv',index=False)

NameError: name 'prob' is not defined
