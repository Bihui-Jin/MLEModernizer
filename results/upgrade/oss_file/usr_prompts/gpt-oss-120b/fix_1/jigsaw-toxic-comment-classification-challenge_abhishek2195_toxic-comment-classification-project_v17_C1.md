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
wordcloud==1.9.4

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

0.0352123120601053

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
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE # for over-sampling
import joblib # for saving models
import warnings
warnings.filterwarnings('ignore')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/287680418.py in <cell line: 0>()
      9 from sklearn.naive_bayes import MultinomialNB
     10 from sklearn.metrics import classification_report
---> 11 from imblearn.over_sampling import SMOTE # for over-sampling
     12 import joblib # for saving models
     13 import warnings

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

## === cell 33
df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/161715770.py in <cell line: 0>()
----> 1 df=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df.pkl')
      2 df_test=joblib.load('/kaggle/input/toxic-comment-classification-cleaned/df_test.pkl')

NameError: name 'joblib' is not defined

## === cell 34
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

## === cell 35
df=reduce_mem_usage(df)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/164482069.py in <cell line: 0>()
----> 1 df=reduce_mem_usage(df)

NameError: name 'df' is not defined

## === cell 36
df_test=reduce_mem_usage(df_test)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4231178207.py in <cell line: 0>()
----> 1 df_test=reduce_mem_usage(df_test)

NameError: name 'df_test' is not defined

## === cell 37
df.head()

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined

## === cell 38
df.isnull().sum()

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/286341606.py in <cell line: 0>()
----> 1 df.isnull().sum()

NameError: name 'df' is not defined

## === cell 39
df_test.head()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685298186.py in <cell line: 0>()
----> 1 df_test.head()

NameError: name 'df_test' is not defined

## === cell 40
df_test.isnull().sum()

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/65500498.py in <cell line: 0>()
----> 1 df_test.isnull().sum()

NameError: name 'df_test' is not defined

## === cell 43
fig,axes=plt.subplots(3,2,figsize=(15,15))

for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
    ax.set_title('{} Distribution'.format(class_name))
    ax.set_xticks(range(2),[0,1])
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')

plt.show()

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/686868683.py in <cell line: 0>()
      2 
      3 for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
----> 4     pd.value_counts(df[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)
      5     ax.set_title('{} Distribution'.format(class_name))
      6     ax.set_xticks(range(2),[0,1])

NameError: name 'df' is not defined

## === cell 44
df_toxic=df[(df['toxic']==1) | (df['severe_toxic']==1) | (df['obscene']==1) | (df['threat']==1) | (df['insult']==1) | (df['identity_hate']==1)].reset_index(drop=True)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2460158193.py in <cell line: 0>()
----> 1 df_toxic=df[(df['toxic']==1) | (df['severe_toxic']==1) | (df['obscene']==1) | (df['threat']==1) | (df['insult']==1) | (df['identity_hate']==1)].reset_index(drop=True)

NameError: name 'df' is not defined

## === cell 45
df_toxic

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2967537003.py in <cell line: 0>()
----> 1 df_toxic

NameError: name 'df_toxic' is not defined

## === cell 47
text_word_count = []

for i in df_toxic['lemmatized']:
      text_word_count.append(len(i.split()))

length_df = pd.DataFrame({'Toxic Word Count Distribution':text_word_count})
length_df.hist(bins = 100, range=(0,length_df['Toxic Word Count Distribution'].max()),figsize=(10,8))
plt.show()

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1413518282.py in <cell line: 0>()
      2 
      3 #populate the lists with comments lengths
----> 4 for i in df_toxic['lemmatized']:
      5       text_word_count.append(len(i.split()))
      6 

NameError: name 'df_toxic' is not defined

## === cell 49
temp=[]

for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
    temp.append(df_toxic.groupby(i)['lemmatized'].apply(lambda x: ' '.join(x))[1])

df_for_dtm=pd.DataFrame(columns=['type','text'])
df_for_dtm['type']=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
df_for_dtm['text']=temp
df_for_dtm=df_for_dtm.set_index('type',drop=True)
df_for_dtm

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2761835534.py in <cell line: 0>()
      2 
      3 for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
----> 4     temp.append(df_toxic.groupby(i)['lemmatized'].apply(lambda x: ' '.join(x))[1])
      5 
      6 df_for_dtm=pd.DataFrame(columns=['type','text'])

NameError: name 'df_toxic' is not defined

## === cell 50
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(analyzer='word')

data=cv.fit_transform(df_for_dtm['text'])

df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_for_dtm.index
df_dtm=df_dtm.transpose()
df_dtm

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/955808490.py in <cell line: 0>()
      4 cv=CountVectorizer(analyzer='word')
      5 
----> 6 data=cv.fit_transform(df_for_dtm['text'])
      7 
      8 df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())

NameError: name 'df_for_dtm' is not defined

## === cell 51
from textwrap import wrap
from wordcloud import WordCloud
def generate_wordcloud(data,title):
    wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(title,60)),fontsize=13)
    plt.show()

## === cell 52
for index,type_of_text in enumerate(df_dtm.columns):
    generate_wordcloud(df_dtm[type_of_text].sort_values(ascending=False),type_of_text)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3868763120.py in <cell line: 0>()
----> 1 for index,type_of_text in enumerate(df_dtm.columns):
      2     generate_wordcloud(df_dtm[type_of_text].sort_values(ascending=False),type_of_text)

NameError: name 'df_dtm' is not defined

## === cell 55
from sklearn.feature_extraction.text import TfidfVectorizer

## === cell 56
vec = TfidfVectorizer(max_features=1000,ngram_range=(1,3), stop_words='english',analyzer='word',dtype=np.float32)

## === cell 57
vec=vec.fit(df_toxic['lemmatized'])
tfidf=vec.transform(df['lemmatized'])
tfidf.shape

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1000935406.py in <cell line: 0>()
      1 # create TFIDF for train
----> 2 vec=vec.fit(df_toxic['lemmatized'])
      3 tfidf=vec.transform(df['lemmatized'])
      4 # shape of TFIDF
      5 tfidf.shape

NameError: name 'df_toxic' is not defined

## === cell 58
tfidf

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294693912.py in <cell line: 0>()
----> 1 tfidf

NameError: name 'tfidf' is not defined

## === cell 59
tfidf_test = vec.transform(df_test['lemmatized'])

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781774660.py in <cell line: 0>()
      1 # create TFIDF for test
----> 2 tfidf_test = vec.transform(df_test['lemmatized'])

NameError: name 'df_test' is not defined

## === cell 60
tfidf_test

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3198579238.py in <cell line: 0>()
----> 1 tfidf_test

NameError: name 'tfidf_test' is not defined

## === cell 62
X=tfidf.toarray()
X_test=tfidf_test.toarray()
target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
X

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/541843002.py in <cell line: 0>()
      1 #Preparing Dataset for Modeling
----> 2 X=tfidf.toarray()
      3 X_test=tfidf_test.toarray()
      4 target=df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
      5 X

NameError: name 'tfidf' is not defined

## === cell 63
prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
prob['id']=df_test['id']

## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668213629.py in <cell line: 0>()
      1 #Dataframe for final probabilties
----> 2 prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'],index=df_test.index)
      3 prob['id']=df_test['id']

NameError: name 'df_test' is not defined

## === cell 64
del df,tfidf,tfidf_test,vec,df_test,df_toxic
gc.collect()

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2076530802.py in <cell line: 0>()
----> 1 del df,tfidf,tfidf_test,vec,df_test,df_toxic
      2 gc.collect()

NameError: name 'df' is not defined

## === cell 65
from sklearn.linear_model import LogisticRegression

## === cell 67
model=[]

for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
    print('{} - Model:\n'.format(value))
    
    y=target[:,index]
    X_temp=X
    
    
    x_train, x_test, y_train, y_test = train_test_split(X_temp, y,stratify=y, test_size=0.2, random_state=42)
    test_model=LogisticRegression(random_state=42)
    test_model=test_model.fit(x_train,y_train)
    train_pred=test_model.predict(x_train)
    print('In-sample Evaluation:\n',classification_report(y_train,train_pred))
    test_pred=test_model.predict(x_test)
    print('Out-sample Evaluation\n',classification_report(y_test,test_pred))
    
    model.append(LogisticRegression(random_state=42))
    model[index] = model[index].fit(X_temp, y)
    prob[value]=model[index].predict_proba(X_test)
    
    joblib.dump(model[index],'{} LR-model.pkl'.format(value))

## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4023599922.py in <cell line: 0>()
      5     print('{} - Model:\n'.format(value))
      6 
----> 7     y=target[:,index]
      8     X_temp=X
      9 #     print('Before Over-Sampling [{}] dataset shape=>({},{})'.format(value,X.shape,y.shape))

NameError: name 'target' is not defined

## === cell 69
prob

## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1085545036.py in <cell line: 0>()
----> 1 prob

NameError: name 'prob' is not defined

## === cell 70
prob.to_csv('submission-LR-tfidf-toxic.csv',index=False)

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1168323327.py in <cell line: 0>()
----> 1 prob.to_csv('submission-LR-tfidf-toxic.csv',index=False)

NameError: name 'prob' is not defined
