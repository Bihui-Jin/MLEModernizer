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

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

for m in list(sys.modules):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import re
import numpy as np
import pandas as pd

import missingno as msno  # missing value check
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

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

warnings.filterwarnings(action="ignore")


## === cell 1
import os

file_list = []
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        file_list.append(os.path.join(dirname, filename))

import zipfile

for path in file_list:
    if path.lower().endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(".")

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

submission = pd.read_csv("./sample_submission.csv")

if os.path.exists("./test_labels.csv"):
    test_labels = pd.read_csv("./test_labels.csv")
else:
    test_labels = None

print("train data length: ", len(train))
print("test data length: ", len(test))


## === cell 2
train.info()


## === cell 3
train.columns


## === cell 4
target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']


## === cell 5
print(train.isnull().value_counts())
print('-'*30)
print(train.isnull().sum())


## === cell 6
print('size of train : {}'.format(len(train)))
print('size of test : {}'.format(len(test)))
print('-'*20)
print(train[target_cols].sum().sort_values(ascending=False))


## === cell 7
train['sum_harmful'] = 0
for col in target_cols:
    train['sum_harmful'] += train[col]
train.head()


## === cell 8
train['len_of_text'] = train['comment_text'].apply(len)


## === cell 9
sns.set_style("darkgrid")
sns.distplot(train['len_of_text'],kde=True)
plt.show()


## === cell 10
x = train.loc[train['len_of_text'] > 1000,'len_of_text']
sns.distplot(x,kde=True)
plt.show()


## === cell 11
print(train.loc[train["len_of_text"] > 1000, "sum_harmful"].value_counts())

plt.yscale("log")
display(sns.countplot(x=train.loc[train["len_of_text"] > 1000, "sum_harmful"]))


## === cell 12
plt.yscale('log')
sns.countplot(train['sum_harmful'])
plt.show()


## === cell 13
train['sum_harmful'].value_counts()


## === cell 14
train.describe()


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1082145760.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m [0mX[0m [0;34m=[0m [0mcv[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mdocuments[0m[0;34m)[0m[0;34m.[0m[0mtoarray[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0mfreqs[0m [0;34m=[0m [0mX[0m[0;34m.[0m[0mtranspose[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mflatten[0m[0;34m([0m[0;34m)[0m [0;31m# 1차원으로, 어레이로 되어있는 인코딩 벡터들을 1차원으로 펴줌[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0mwords[0m [0;34m=[0m [0mcv[0m[0;34m.[0m[0mget_feature_names[0m[0;34m([0m[0;34m)[0m [0;31m# Array mapping from feature integer indices to feature name.[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m [0mdf_word[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0;34m{[0m[0;34m'word'[0m[0;34m:[0m [0mwords[0m[0;34m,[0m [0;34m'freq'[0m[0;34m:[0m [0mfreqs[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'CountVectorizer' object has no attribute 'get_feature_names'

## === cell 18
stopwords_list = df_word.word.tolist()[:70]
print('학습 데이터로 만든 불용어 개수 : ',len(stopwords_list))
print(stopwords_list[:10],'...')

from nltk.corpus import stopwords
stopwords_list+=stopwords.words('english')
stopwords = set(stopwords_list)
print(' ')
print("최종 stopwords 개수 :", len(stopwords))
