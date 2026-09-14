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
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
wordcloud==1.9.4
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.79899

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import multiprocessing
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


## === cell 1
set(stopwords.words('english'))


## === cell 2
files=['../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv',
       '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv',
       '../input/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv',
       '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'
      ]

def load_data(file):
    return pd.read_csv(file)
with multiprocessing.Pool() as pool:
    test,train,all_data,sub=pool.map(load_data,files)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/3008787514.py", line 8, in load_data
    return pd.read_csv(file)
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '../input/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3008787514.py in <cell line: 0>()
      8     return pd.read_csv(file)
      9 with multiprocessing.Pool() as pool:
---> 10     test,train,all_data,sub=pool.map(load_data,files)

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

FileNotFoundError: [Errno 2] No such file or directory: '../input/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv'

## === cell 3
train.info()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1956042609.py in <cell line: 0>()
      2 #     print("{} -----------> {}".format(col,all_data[col].dtypes))
      3 #     print("{} ===========> {}".format(col,train[col].dtypes))
----> 4 train.info()

NameError: name 'train' is not defined

## === cell 4
train.target.value_counts(dropna=True).head()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1560133445.py in <cell line: 0>()
----> 1 train.target.value_counts(dropna=True).head()

NameError: name 'train' is not defined

## === cell 5
train.shape


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612428091.py in <cell line: 0>()
----> 1 train.shape

NameError: name 'train' is not defined

## === cell 6
train['target'].isnull().sum()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1549820804.py in <cell line: 0>()
----> 1 train['target'].isnull().sum()

NameError: name 'train' is not defined

## === cell 7
X=train[['id','comment_text','target']]
train.columns.values


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1525350335.py in <cell line: 0>()
----> 1 X=train[['id','comment_text','target']]
      2 train.columns.values

NameError: name 'train' is not defined

## === cell 8
train['hindu'].head()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3543226343.py in <cell line: 0>()
----> 1 train['hindu'].head()
      2 # train['hindu'].dtype

NameError: name 'train' is not defined

## === cell 9
tox=0
neut=0
no_of_rows=X.shape[0]
for row in range(no_of_rows):
    if X['target'][row]>0.7:
        tox+=1
    else:
        neut+=1


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2757449413.py in <cell line: 0>()
      1 tox=0
      2 neut=0
----> 3 no_of_rows=X.shape[0]
      4 for row in range(no_of_rows):
      5     if X['target'][row]>0.7:

NameError: name 'X' is not defined

## === cell 10
print(f'{round((tox*100)/no_of_rows,3)}% data contains toxic comments')
print(f'{round((neut*100/no_of_rows),3)}% data contains neutral comments')


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2546383170.py in <cell line: 0>()
----> 1 print(f'{round((tox*100)/no_of_rows,3)}% data contains toxic comments')
      2 print(f'{round((neut*100/no_of_rows),3)}% data contains neutral comments')

NameError: name 'no_of_rows' is not defined

## === cell 11
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

remove_n = lambda x: re.sub("\n", " ", x)

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

X['comment_text'] = X['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2147776861.py in <cell line: 0>()
     13 
     14 # Apply all the lambda functions wrote previously through .map on the comments column
---> 15 X['comment_text'] = X['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

NameError: name 'X' is not defined

## === cell 12
test['comment_text'] = test['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1537167352.py in <cell line: 0>()
----> 1 test['comment_text'] = test['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

NameError: name 'test' is not defined

## === cell 13
import wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def wordcloud(df, label):
    
    subset=df[df[label]>0.7]
    text=subset.comment_text.values
    wc= WordCloud(background_color="white",max_words=4000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)


## === cell 14
wordcloud(X,'target')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1606487363.py in <cell line: 0>()
----> 1 wordcloud(X,'target')

NameError: name 'X' is not defined

## === cell 15
toxic_train=X[X['target']>0.7].iloc[0:3000,:]
toxic_train.shape


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/456295198.py in <cell line: 0>()
----> 1 toxic_train=X[X['target']>0.7].iloc[0:3000,:]
      2 toxic_train.shape

NameError: name 'X' is not defined

## === cell 16
neutral_train=X[X['target']<=0.7].iloc[0:7000,:]
neutral_train.shape


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4022218805.py in <cell line: 0>()
----> 1 neutral_train=X[X['target']<=0.7].iloc[0:7000,:]
      2 neutral_train.shape

NameError: name 'X' is not defined

## === cell 17
balanced_train=pd.concat([toxic_train,neutral_train],axis=0)
balanced_train.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/461372648.py in <cell line: 0>()
----> 1 balanced_train=pd.concat([toxic_train,neutral_train],axis=0)
      2 balanced_train.shape
      3 
      4 # balanced_train=X

NameError: name 'toxic_train' is not defined

## === cell 18
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, fbeta_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


## === cell 19
'''
df_done: data_tox_done, data_sev_done, ...
label: toxic, severe_toxic, ...
vectorizer values: CountVectorizer, TfidfVectorizer
gram_range values: (1,1) for unigram, (2,2) for bigram
'''
def cv_tf_train_test(df_done,label,vectorizer,ngram):

    ''' Train/Test split'''
    X = df_done.comment_text
    y = df_done[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ''' Count Vectorizer/TF-IDF '''

    cv1 = vectorizer(ngram_range=(ngram), stop_words='english')
    
    X_train_cv1 = cv1.fit_transform(X_train) # Learn the vocabulary dictionary and return term-document matrix
    X_test_cv1  = cv1.transform(X_test)      # Learn a vocabulary dictionary of all tokens in the raw documents.
    
        
    ''' Initialize all model objects and fit the models on the training data '''
    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    print('lr done')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_cv1, y_train)

    

    xgb=XGBClassifier()
    xgb.fit(X_train_cv1,y_train)
    
    svm_model = LinearSVC()
    svm_model.fit(X_train_cv1, y_train)

    randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
    randomforest.fit(X_train_cv1, y_train)
    print('rdf done')
    
    f1_score_data = {'F1 Score':[f1_score(lr.predict(X_test_cv1), y_test), f1_score(knn.predict(X_test_cv1), y_test), 
                                f1_score(xgb.predict(X_test_cv1),y_test),
                                f1_score(svm_model.predict(X_test_cv1), y_test), f1_score(randomforest.predict(X_test_cv1), y_test)]} 
                          
    df_f1 = pd.DataFrame(f1_score_data, index=['Log Regression','KNN', 'XGB', 'SVM', 'Random Forest'])  
    return df_f1


## === cell 20
balanced_train['target']=np.where(balanced_train['target']>0.7,1.0,0.0)
balanced_train.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2469672480.py in <cell line: 0>()
----> 1 balanced_train['target']=np.where(balanced_train['target']>0.7,1.0,0.0)
      2 balanced_train.head()

NameError: name 'balanced_train' is not defined

## === cell 21
import time

t0 = time.time()

df_tox_cv = cv_tf_train_test(balanced_train, 'target', TfidfVectorizer, (1,1))
df_tox_cv.rename(columns={'F1 Score': 'F1 Score(target)'}, inplace=True)

t1 = time.time()

total = 'Time taken: {} seconds'.format(t1-t0)
print(total)

df_tox_cv


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1258747501.py in <cell line: 0>()
      3 t0 = time.time()
      4 
----> 5 df_tox_cv = cv_tf_train_test(balanced_train, 'target', TfidfVectorizer, (1,1))
      6 df_tox_cv.rename(columns={'F1 Score': 'F1 Score(target)'}, inplace=True)
      7 

NameError: name 'balanced_train' is not defined

## === cell 22
X = balanced_train.comment_text
y = balanced_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfv.fit_transform(X_train)  # Convert the X data into a document term matrix dataframe
X_test_fit = tfv.transform(X_test)  # Converts the X_test comments into Vectorized format




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3877808959.py in <cell line: 0>()
----> 1 X = balanced_train.comment_text
      2 y = balanced_train['target']
      3 
      4 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
      5 

NameError: name 'balanced_train' is not defined

## === cell 23
lr=LogisticRegression()
lr.fit(X_train_fit,y_train)

my_ans=[]
for row in range(test.shape[0]):
    comment=[test['comment_text'][row]]
    cmt=tfv.transform(comment)
    my_ans.append(lr.predict_proba(cmt)[:,1])

data={'id':[],
      'prediction':[]
     }
df=pd.DataFrame(data)

df['id']=test['id']

df['prediction']=pd.DataFrame(my_ans)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3111582744.py in <cell line: 0>()
      1 lr=LogisticRegression()
----> 2 lr.fit(X_train_fit,y_train)
      3 
      4 my_ans=[]
      5 for row in range(test.shape[0]):

NameError: name 'X_train_fit' is not defined

## === cell 25
df.to_csv('submission.csv',index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2821879807.py in <cell line: 0>()
----> 1 df.to_csv('submission.csv',index=False)

NameError: name 'df' is not defined
