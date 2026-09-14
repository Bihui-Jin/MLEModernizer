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

gensim==4.4.0
geopandas==0.14.4
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

0.9612273619104004

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
import gc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import joblib # for saving models
import warnings
warnings.filterwarnings('ignore')

## === cell 3
df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/161715770.py in <cell line: 0>()
----> 1 df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
      2 df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/toxic-comment-classification-cleaned/df.pkl'

## === cell 4
df.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined

## === cell 5
df.isnull().sum()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/286341606.py in <cell line: 0>()
----> 1 df.isnull().sum()

NameError: name 'df' is not defined

## === cell 6
df_test.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685298186.py in <cell line: 0>()
----> 1 df_test.head()

NameError: name 'df_test' is not defined

## === cell 7
df.isnull().sum()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/286341606.py in <cell line: 0>()
----> 1 df.isnull().sum()

NameError: name 'df' is not defined

## === cell 9
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

## === cell 10
df=reduce_mem_usage(df)
gc.collect()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3694206692.py in <cell line: 0>()
----> 1 df=reduce_mem_usage(df)
      2 gc.collect()

NameError: name 'df' is not defined

## === cell 11
df_test=reduce_mem_usage(df_test)
gc.collect()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/396199544.py in <cell line: 0>()
----> 1 df_test=reduce_mem_usage(df_test)
      2 gc.collect()

NameError: name 'df_test' is not defined

## === cell 14
fig,axes=plt.subplots(3,2,figsize=(15,15))

for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
    ax.set_title('{} Distribution'.format(class_name))
    ax.set_xticks(range(2),[0,1])
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')

plt.show()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/686868683.py in <cell line: 0>()
      2 
      3 for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
----> 4     pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
      5     ax.set_title('{} Distribution'.format(class_name))
      6     ax.set_xticks(range(2),[0,1])

NameError: name 'df' is not defined

## === cell 16
all_text=pd.concat([df['lemmatized'], df_test['lemmatized']]).reset_index(drop=True)
all_text.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1339081227.py in <cell line: 0>()
----> 1 all_text=pd.concat([df['lemmatized'], df_test['lemmatized']]).reset_index(drop=True)
      2 all_text.head()

NameError: name 'df' is not defined

## === cell 32
comments=[]
for i in all_text:
    comments.append(i.split())
comments[:5]

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1444017258.py in <cell line: 0>()
      1 # Creating data for the model training
      2 comments=[]
----> 3 for i in all_text:
      4     comments.append(i.split())
      5 comments[:5]

NameError: name 'all_text' is not defined

## === cell 33
from gensim.models import Word2Vec

## === cell 34
w2v_model = Word2Vec(comments, size=300, min_count=2,window=4, sg=1,workers=4)

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1828455378.py in <cell line: 0>()
      1 # training a word2vec model from the given data set
----> 2 w2v_model = Word2Vec(comments, size=300, min_count=2,window=4, sg=1,workers=4)

TypeError: Word2Vec.__init__() got an unexpected keyword argument 'size'

## === cell 35
print('vocabulary size:', len(w2v_model.wv.vocab))

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1834470577.py in <cell line: 0>()
      1 # vocabulary size
----> 2 print('vocabulary size:', len(w2v_model.wv.vocab))

NameError: name 'w2v_model' is not defined

## === cell 36
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.vocab:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        return np.mean(embeddings, axis=0)

## === cell 37
X=df['lemmatized'].apply(lambda x :get_embedding_w2v(x.split()))
X=pd.DataFrame(X.tolist())
print('Shape of X=>',X.shape)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1204482772.py in <cell line: 0>()
----> 1 X=df['lemmatized'].apply(lambda x :get_embedding_w2v(x.split()))
      2 X=pd.DataFrame(X.tolist())
      3 print('Shape of X=>',X.shape)

NameError: name 'df' is not defined

## === cell 38
X_test=df_test['lemmatized'].apply(lambda x:get_embedding_w2v(x.split()))
X_test=pd.DataFrame(X_test.tolist())
print('Shape of X_test=>',X_test.shape)

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1233598207.py in <cell line: 0>()
----> 1 X_test=df_test['lemmatized'].apply(lambda x:get_embedding_w2v(x.split()))
      2 X_test=pd.DataFrame(X_test.tolist())
      3 print('Shape of X_test=>',X_test.shape)

NameError: name 'df_test' is not defined

## === cell 40
target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3364935035.py in <cell line: 0>()
----> 1 target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

NameError: name 'df' is not defined

## === cell 41
prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
prob['id']=df_test['id']

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668213629.py in <cell line: 0>()
      1 #Dataframe for final probabilties
----> 2 prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
      3 prob['id']=df_test['id']

NameError: name 'df_test' is not defined

## === cell 42
from sklearn.linear_model import LogisticRegression

## === cell 43
for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    print('{} - Model:\n'.format(value))
    
    y=target[:,index]
    print('Y=>',y)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=42)
    test_model=LogisticRegression(random_state=42)
    test_model=test_model.fit(x_train,y_train)
    train_pred=test_model.predict(x_train)
    print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_train,train_pred))
    test_pred=test_model.predict(x_test)
    print('Out-sample Evaluation ROC-AUC Score\n',roc_auc_score(y_test,test_pred))
    
    model=LogisticRegression(random_state=42)
    model = model.fit(X, y)
    y_pred=model.predict(X)
    print('In-sample Evaluation on Whole Dataset ROC-AUC Score:\n',roc_auc_score(y,y_pred))
    print('Model=>',model)
    prob[value]=model.predict_proba(X_test)[:, 1]

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2839798920.py in <cell line: 0>()
      3     print('{} - Model:\n'.format(value))
      4 
----> 5     y=target[:,index]
      6     print('Y=>',y)
      7 

NameError: name 'target' is not defined

## === cell 44
prob

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1085545036.py in <cell line: 0>()
----> 1 prob

NameError: name 'prob' is not defined

## === cell 45
prob.to_csv('submission-LR-w2v-train-all.csv',index=False)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/685514110.py in <cell line: 0>()
----> 1 prob.to_csv('submission-LR-w2v-train-all.csv',index=False)

NameError: name 'prob' is not defined
