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

3.11

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.96963

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import re
import numpy as np
import pandas as pd

import missingno as msno # missing value check
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings(action='ignore') 


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import os
file_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_list.append(os.path.join(dirname, filename))
        
import zipfile
for i in range(len(file_list)):
    with zipfile.ZipFile(file_list[i],"r") as z:
        z.extractall(".")
        
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

submission = pd.read_csv('./sample_submission.csv')
test_labels = pd.read_csv('./test_labels.csv')

print("train data length: ", len(train))
print("test data length: ", len(test))


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
BadZipFile                                Traceback (most recent call last)
/tmp/ipykernel_11/1014184875.py in <cell line: 0>()
      7 import zipfile
      8 for i in range(len(file_list)):
----> 9     with zipfile.ZipFile(file_list[i],"r") as z:
     10         z.extractall(".")
     11 

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1311         try:
   1312             if mode == 'r':
-> 1313                 self._RealGetContents()
   1314             elif mode in ('w', 'x'):
   1315                 # set the modified flag so central directory gets written

/usr/lib/python3.11/zipfile.py in _RealGetContents(self)
   1378             raise BadZipFile("File is not a zip file")
   1379         if not endrec:
-> 1380             raise BadZipFile("File is not a zip file")
   1381         if self.debug > 1:
   1382             print(endrec)

BadZipFile: File is not a zip file

## === cell 2
train.info()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3994350071.py in <cell line: 0>()
----> 1 train.info()

NameError: name 'train' is not defined

## === cell 3
train.columns


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2714421424.py in <cell line: 0>()
----> 1 train.columns

NameError: name 'train' is not defined

## === cell 4
target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']


## === cell 5
print(train.isnull().value_counts())
print('-'*30)
print(train.isnull().sum())


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3475426443.py in <cell line: 0>()
      1 # Check missing value
----> 2 print(train.isnull().value_counts())
      3 print('-'*30)
      4 print(train.isnull().sum())

NameError: name 'train' is not defined

## === cell 6
print('size of train : {}'.format(len(train)))
print('size of test : {}'.format(len(test)))
print('-'*20)
print(train[target_cols].sum().sort_values(ascending=False))


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2366396837.py in <cell line: 0>()
      1 # data size
----> 2 print('size of train : {}'.format(len(train)))
      3 print('size of test : {}'.format(len(test)))
      4 print('-'*20)
      5 print(train[target_cols].sum().sort_values(ascending=False))

NameError: name 'train' is not defined

## === cell 7
train['sum_harmful'] = 0
for col in target_cols:
    train['sum_harmful'] += train[col]
train.head()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1211109020.py in <cell line: 0>()
----> 1 train['sum_harmful'] = 0
      2 for col in target_cols:
      3     train['sum_harmful'] += train[col]
      4 train.head()

NameError: name 'train' is not defined

## === cell 8
train['len_of_text'] = train['comment_text'].apply(len)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/994123085.py in <cell line: 0>()
      1 # 텍스트 전처리 전,텍스트 길이 분포 확인
----> 2 train['len_of_text'] = train['comment_text'].apply(len)

NameError: name 'train' is not defined

## === cell 9
sns.set_style("darkgrid")
sns.distplot(train['len_of_text'],kde=True)
plt.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3326397760.py in <cell line: 0>()
      1 # 텍스트 길이 분포 시각화
      2 sns.set_style("darkgrid")
----> 3 sns.distplot(train['len_of_text'],kde=True)
      4 plt.show()

NameError: name 'train' is not defined

## === cell 10
x = train.loc[train['len_of_text'] > 1000,'len_of_text']
sns.distplot(x,kde=True)
plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/215075923.py in <cell line: 0>()
      1 # 대부분이 1000 이하의 텍스트 길이 -> 1000 이상을 한 번 보자
----> 2 x = train.loc[train['len_of_text'] > 1000,'len_of_text']
      3 sns.distplot(x,kde=True)
      4 plt.show()

NameError: name 'train' is not defined

## === cell 11
print(train.loc[train['len_of_text'] > 1000,'sum_harmful'].value_counts())

plt.yscale('log')
display(sns.countplot(train.loc[train['len_of_text'] > 1000,'sum_harmful']))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3670596164.py in <cell line: 0>()
      1 # 길이가 1000이상인 데이터의 target_cols이 1로 분류된 것의 개수의 합
----> 2 print(train.loc[train['len_of_text'] > 1000,'sum_harmful'].value_counts())
      3 
      4 plt.yscale('log')
      5 display(sns.countplot(train.loc[train['len_of_text'] > 1000,'sum_harmful']))

NameError: name 'train' is not defined

## === cell 12
plt.yscale('log')
sns.countplot(train['sum_harmful'])
plt.show()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4270472777.py in <cell line: 0>()
      1 # 1000이상도 기존의 데이터 분포와 비슷해 보인다 -> 이상치 처리 하지 않음
      2 plt.yscale('log')
----> 3 sns.countplot(train['sum_harmful'])
      4 plt.show()

NameError: name 'train' is not defined

## === cell 13
train['sum_harmful'].value_counts()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1120464245.py in <cell line: 0>()
----> 1 train['sum_harmful'].value_counts()

NameError: name 'train' is not defined

## === cell 14
train.describe()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2359111164.py in <cell line: 0>()
----> 1 train.describe()

NameError: name 'train' is not defined

## === cell 15
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


## === cell 16
train['preprocess_text'] = train['comment_text'].apply(clean_text)
train.head(3)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4282235152.py in <cell line: 0>()
----> 1 train['preprocess_text'] = train['comment_text'].apply(clean_text)
      2 train.head(3)

NameError: name 'train' is not defined

## === cell 17
cv = CountVectorizer() # 만약 n-gram으로 설정시 안에 (2-gram까지하는 경우)ngram_range = (1, 3) 를 설정
documents = train.comment_text.tolist()
documents = [' '.join(documents)] 

X = cv.fit_transform(documents).toarray()
freqs = X.transpose().flatten() # 1차원으로, 어레이로 되어있는 인코딩 벡터들을 1차원으로 펴줌
words = cv.get_feature_names() # Array mapping from feature integer indices to feature name.

df_word = pd.DataFrame({'word': words, 'freq': freqs})
df_word = df_word.sort_values(by='freq', ascending=False)

df_word = df_word.reset_index().drop(['index'],axis=1)
df_word[:10]


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1082145760.py in <cell line: 0>()
      1 # 전체 데이터에 대해 Countvectorizer
      2 cv = CountVectorizer() # 만약 n-gram으로 설정시 안에 (2-gram까지하는 경우)ngram_range = (1, 3) 를 설정
----> 3 documents = train.comment_text.tolist()
      4 documents = [' '.join(documents)]
      5 

NameError: name 'train' is not defined

## === cell 18
stopwords_list = df_word.word.tolist()[:70]
print('학습 데이터로 만든 불용어 개수 : ',len(stopwords_list))
print(stopwords_list[:10],'...')

from nltk.corpus import stopwords
stopwords_list+=stopwords.words('english')
stopwords = set(stopwords_list)
print(' ')
print("최종 stopwords 개수 :", len(stopwords))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4116394618.py in <cell line: 0>()
----> 1 stopwords_list = df_word.word.tolist()[:70]
      2 print('학습 데이터로 만든 불용어 개수 : ',len(stopwords_list))
      3 print(stopwords_list[:10],'...')
      4 
      5 # nltk의 stopwords 사용

NameError: name 'df_word' is not defined

## === cell 19
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    stopwords_list = stopwords
    word_tokens = word_tokenize(text)
    result = []
    for w in word_tokens:
        if len(w)>2 and w not in stopwords_list:
            result.append(w)
    return ' '.join(result)

def remove_special(text, lower=True):
    if lower:
        text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(
        text.split()
    )
    return text

def remove_repeat(text, repeat=1):
    text = text.split(' ')
    result = []
    for word in text:
        if result.count(word)<repeat:
            result.append(word)
    return ' '.join(result)


## === cell 20
train['preprocess_text'] = train['preprocess_text'].apply(remove_special)
train['preprocess_text'] = train['preprocess_text'].apply(remove_stopwords)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4004449989.py in <cell line: 0>()
----> 1 train['preprocess_text'] = train['preprocess_text'].apply(remove_special)
      2 train['preprocess_text'] = train['preprocess_text'].apply(remove_stopwords)

NameError: name 'train' is not defined

## === cell 21
test['preprocess_text'] = test['comment_text'].apply(clean_text)
test['preprocess_text'] = test['preprocess_text'].apply(remove_special)
test['preprocess_text'] = test['preprocess_text'].apply(remove_stopwords)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3481498607.py in <cell line: 0>()
----> 1 test['preprocess_text'] = test['comment_text'].apply(clean_text)
      2 test['preprocess_text'] = test['preprocess_text'].apply(remove_special)
      3 test['preprocess_text'] = test['preprocess_text'].apply(remove_stopwords)

NameError: name 'test' is not defined

## === cell 22
def num_of_word(text):
    return len(text.split(' '))

tmp = train['preprocess_text'].apply(num_of_word)
print(tmp.describe())
sns.distplot(tmp)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/392502226.py in <cell line: 0>()
      3     return len(text.split(' '))
      4 
----> 5 tmp = train['preprocess_text'].apply(num_of_word)
      6 print(tmp.describe())
      7 sns.distplot(tmp)

NameError: name 'train' is not defined

## === cell 23
train['num_of_word'] = train['preprocess_text'].apply(num_of_word)
display(train[(train['num_of_word'] > 500)&(train['num_of_word'] < 1000)].head(3))
display(train[train['num_of_word'] > 1000])


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3007450761.py in <cell line: 0>()
----> 1 train['num_of_word'] = train['preprocess_text'].apply(num_of_word)
      2 display(train[(train['num_of_word'] > 500)&(train['num_of_word'] < 1000)].head(3))
      3 display(train[train['num_of_word'] > 1000])

NameError: name 'train' is not defined

## === cell 24
train['remove_repeat'] = train['preprocess_text'].apply(remove_repeat)
test['remove_repeat'] = test['preprocess_text'].apply(remove_repeat)

display(train.head(3))
display(test.head(3))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2883856007.py in <cell line: 0>()
      1 # 중복 제거 여부 성능 비교를 위해서 칼럼을 따로 만들어서 저장
----> 2 train['remove_repeat'] = train['preprocess_text'].apply(remove_repeat)
      3 test['remove_repeat'] = test['preprocess_text'].apply(remove_repeat)
      4 
      5 display(train.head(3))

NameError: name 'train' is not defined

## === cell 25
corr_matrix = train.corr()
sns.heatmap(corr_matrix, cmap='Blues',annot=True)
plt.show()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1054241415.py in <cell line: 0>()
      1 # column들의 상관관계 확인 - heatmap
----> 2 corr_matrix = train.corr()
      3 sns.heatmap(corr_matrix, cmap='Blues',annot=True)
      4 plt.show()

NameError: name 'train' is not defined

## === cell 26
preprocess_text = train.preprocess_text

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(preprocess_text)
y = train[target_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print('X_train len: ', X_train.shape[0])
print('X_valid len:  ',  X_valid.shape[0])


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/207968579.py in <cell line: 0>()
----> 1 preprocess_text = train.preprocess_text
      2 
      3 vectorizer = TfidfVectorizer(max_features=5000)
      4 
      5 X = vectorizer.fit_transform(preprocess_text)

NameError: name 'train' is not defined

## === cell 27
from collections import defaultdict
accuracy_data = defaultdict(list)
for col in target_cols:
    print(' ')
    print('----prediction of {} column----'.format(col))
    print(' ')
    y = y_train[col]
    model = LogisticRegression()
    model.fit(X_train,y)
    y_pred = model.predict(X_valid)
    print('Testing accuracy is {}'.format(round(accuracy_score(y_valid[col], y_pred),5)))
    accuracy_data[col].append(round(accuracy_score(y_valid[col], y_pred),5))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1038898087.py in <cell line: 0>()
      5     print('----prediction of {} column----'.format(col))
      6     print(' ')
----> 7     y = y_train[col]
      8     model = LogisticRegression()
      9     model.fit(X_train,y)

NameError: name 'y_train' is not defined

## === cell 28
remove_repeat_text = train.remove_repeat

vectorizer2 = TfidfVectorizer(max_features=5000)

X2 = vectorizer2.fit_transform(remove_repeat_text)
y = train[target_cols]

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X2, y, test_size=0.2, random_state=0)

print('X_train len: ', X_train2.shape[0])
print('X_valid len:  ',  X_valid2.shape[0])


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/496701025.py in <cell line: 0>()
----> 1 remove_repeat_text = train.remove_repeat
      2 
      3 vectorizer2 = TfidfVectorizer(max_features=5000)
      4 
      5 X2 = vectorizer2.fit_transform(remove_repeat_text)

NameError: name 'train' is not defined

## === cell 29
for col in target_cols:
    print(' ')
    print('----prediction of {} column----'.format(col))
    print(' ')
    y = y_train2[col]
    model2 = LogisticRegression()
    model2.fit(X_train2,y)
    y_pred2 = model.predict(X_valid2)
    print('Testing accuracy is {}'.format(round(accuracy_score(y_valid2[col], y_pred2),5)))
    accuracy_data[col].append(round(accuracy_score(y_valid2[col], y_pred2),5))


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1734791213.py in <cell line: 0>()
      4     print('----prediction of {} column----'.format(col))
      5     print(' ')
----> 6     y = y_train2[col]
      7     model2 = LogisticRegression()
      8     model2.fit(X_train2,y)

NameError: name 'y_train2' is not defined

## === cell 30
tmp = pd.DataFrame(accuracy_data, index=['allow_rep','remove_rep'])
tmp


## === cell 31
pre_test = test.preprocess_text
X_test = vectorizer.transform(pre_test)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3133085202.py in <cell line: 0>()
----> 1 pre_test = test.preprocess_text
      2 X_test = vectorizer.transform(pre_test)

NameError: name 'test' is not defined

## === cell 32
submission_prediction = submission.copy()
preprocess_text = train.preprocess_text

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(preprocess_text)
y = train[target_cols]

for col in target_cols:
    print(' ')
    print('----prediction of {} column----'.format(col))
    y_train = y[col]
    model = LogisticRegression()
    model.fit(X,y_train)
    test_y_prob = model.predict_proba(X_test)[:,1]
    submission_prediction[col] = test_y_prob


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2806220654.py in <cell line: 0>()
      1 # train set 전체 활용
----> 2 submission_prediction = submission.copy()
      3 preprocess_text = train.preprocess_text
      4 
      5 vectorizer = TfidfVectorizer(max_features=5000)

NameError: name 'submission' is not defined

## === cell 33
DATA_PATH = './'
submission_prediction.to_csv(DATA_PATH+'submission.csv',index=False)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4112955701.py in <cell line: 0>()
      1 DATA_PATH = './'
----> 2 submission_prediction.to_csv(DATA_PATH+'submission.csv',index=False)

NameError: name 'submission_prediction' is not defined

## === cell 34
from tensorflow.keras.preprocessing.text import Tokenizer

preprocess_texts = []
for text in train['preprocess_text']:
    preprocess_texts.append(text.split(' '))
    
print(preprocess_texts[:2])


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4247253502.py in <cell line: 0>()
      3 
      4 preprocess_texts = []
----> 5 for text in train['preprocess_text']:
      6     preprocess_texts.append(text.split(' '))
      7 

NameError: name 'train' is not defined

## === cell 35
tokenizer = Tokenizer(oov_token = 'OOV') # OOV 단어는 1로 인덱스

tokenizer.fit_on_texts(preprocess_texts)

word_vocab = tokenizer.word_index
word_vocab[""] = 0

print('총 voca 개수: ', len(word_vocab))


## === cell 36
encoded = tokenizer.texts_to_sequences(preprocess_texts)

MAX_SEQ_LEN = max(len(item) for item in encoded)

print('padding len: ', MAX_SEQ_LEN)
padded = pad_sequences(encoded,maxlen=MAX_SEQ_LEN ,padding='post')
print('train data -> padding shape: {}'.format(padded.shape))


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/216170658.py in <cell line: 0>()
      3 
      4 # 정수 인코딩벡터 길이 맞춰주기
----> 5 MAX_SEQ_LEN = max(len(item) for item in encoded)
      6 
      7 # 길이를 토큰 시퀀스의 최대 길이로 맞춰주고 뒤에 0으로 padding

ValueError: max() arg is an empty sequence

## === cell 37
tmp = train[['id','preprocess_text']]

tmp.to_csv(DATA_PATH+'/preprocess_train.csv')

np.save(open(DATA_PATH + 'encoding_train.npy', 'wb'), padded)

y = train[target_cols]
y.to_csv(DATA_PATH+'label_train.csv',index=False)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/629416724.py in <cell line: 0>()
----> 1 tmp = train[['id','preprocess_text']]
      2 
      3 # save preprocess tests
      4 tmp.to_csv(DATA_PATH+'/preprocess_train.csv')
      5 

NameError: name 'train' is not defined

## === cell 38
preprocess_test = []
for text in test['preprocess_text']:
    preprocess_test.append(text.split(' '))
    
encoded = tokenizer.texts_to_sequences(preprocess_test)

padded_test = pad_sequences(encoded,maxlen=MAX_SEQ_LEN ,padding='post')

tmp2 = test[['id','preprocess_text']]
tmp2.to_csv(DATA_PATH+'/preprocess_test.csv')

np.save(open(DATA_PATH + 'encoding_test.npy', 'wb'), padded_test)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2707336211.py in <cell line: 0>()
      1 preprocess_test = []
----> 2 for text in test['preprocess_text']:
      3     preprocess_test.append(text.split(' '))
      4 
      5 encoded = tokenizer.texts_to_sequences(preprocess_test)

NameError: name 'test' is not defined
