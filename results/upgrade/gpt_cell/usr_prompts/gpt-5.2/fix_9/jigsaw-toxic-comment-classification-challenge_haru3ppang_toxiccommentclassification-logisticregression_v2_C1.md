# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens because `sns.countplot()` in seaborn 0.12+ interprets a passed `Series` as `data` rather than `x`, then tries to access `data[0]`. Since the filtered Series keeps the original (non-0-based) index labels, `data[0]` raises `KeyError: 0`. Passing the Series explicitly as `x=` (or resetting index) avoids seaborn’s internal indexing path and fixes the error deterministically.

Patch summary: In cell 11, change the `sns.countplot(...)` call to use the keyword argument `x=` so seaborn treats the Series as the x variable. No other logic, variables, or plots are changed.

Updated cells: Cell 11 only.

Compatibility notes for cell k+1: This change does not modify `train` or any variables used later; cell 12 continues to work unchanged.

Assumptions: Seaborn version is 0.12.2 as listed, and the goal is to keep the same plot semantics (count of `sum_harmful` for rows with `len_of_text > 1000`).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash occurs because `train.corr()` in pandas 2.2 tries to convert the entire DataFrame to floats, but `train` contains non-numeric string columns like `id` and `comment_text`, causing `ValueError: could not convert string to float`. The intended correlation heatmap should be computed only over numeric columns. Filtering to numeric columns (or setting `numeric_only=True`) preserves the analysis intent and avoids the conversion error.

Patch summary: In cell 25, compute the correlation matrix using only numeric columns via `train.select_dtypes(include=[np.number]).corr()` so string columns are excluded. Keep the heatmap call unchanged aside from referencing the filtered correlation matrix.

Updated cells: Provided below (cell 25 only).

Compatibility notes for cell k+1: This change only affects the local variable `corr_matrix` used for plotting in cell 25; it does not modify `train`, `test`, or any columns used later (e.g., `train.preprocess_text` in cell 26), so downstream cells remain compatible.

Assumptions: `numpy` is already imported as `np` earlier (cell 0), and the goal of the heatmap is to visualize correlations among numeric features/labels.'

# 9. Code solution

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
cv = (
    CountVectorizer()
)  # 만약 n-gram으로 설정시 안에 (2-gram까지하는 경우)ngram_range = (1, 3) 를 설정
documents = train.comment_text.tolist()
documents = [" ".join(documents)]

X = cv.fit_transform(documents).toarray()
freqs = (
    X.transpose().flatten()
)  # 1차원으로, 어레이로 되어있는 인코딩 벡터들을 1차원으로 펴줌

words = (
    cv.get_feature_names_out()
)  # Array mapping from feature integer indices to feature name.

df_word = pd.DataFrame({"word": words, "freq": freqs})
df_word = df_word.sort_values(by="freq", ascending=False)

df_word = df_word.reset_index().drop(["index"], axis=1)
df_word[:10]


## === cell 18
stopwords_list = df_word.word.tolist()[:70]
print('학습 데이터로 만든 불용어 개수 : ',len(stopwords_list))
print(stopwords_list[:10],'...')

from nltk.corpus import stopwords
stopwords_list+=stopwords.words('english')
stopwords = set(stopwords_list)
print(' ')
print("최종 stopwords 개수 :", len(stopwords))


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


## === cell 21
test['preprocess_text'] = test['comment_text'].apply(clean_text)
test['preprocess_text'] = test['preprocess_text'].apply(remove_special)
test['preprocess_text'] = test['preprocess_text'].apply(remove_stopwords)


## === cell 22
def num_of_word(text):
    return len(text.split(' '))

tmp = train['preprocess_text'].apply(num_of_word)
print(tmp.describe())
sns.distplot(tmp)


## === cell 23
train['num_of_word'] = train['preprocess_text'].apply(num_of_word)
display(train[(train['num_of_word'] > 500)&(train['num_of_word'] < 1000)].head(3))
display(train[train['num_of_word'] > 1000])


## === cell 24
train['remove_repeat'] = train['preprocess_text'].apply(remove_repeat)
test['remove_repeat'] = test['preprocess_text'].apply(remove_repeat)

display(train.head(3))
display(test.head(3))


## === cell 25
corr_matrix = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, cmap="Blues", annot=True)
plt.show()


## === cell 26
preprocess_text = train.preprocess_text

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(preprocess_text)
y = train[target_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print('X_train len: ', X_train.shape[0])
print('X_valid len:  ',  X_valid.shape[0])


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


## === cell 28
remove_repeat_text = train.remove_repeat

vectorizer2 = TfidfVectorizer(max_features=5000)

X2 = vectorizer2.fit_transform(remove_repeat_text)
y = train[target_cols]

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X2, y, test_size=0.2, random_state=0)

print('X_train len: ', X_train2.shape[0])
print('X_valid len:  ',  X_valid2.shape[0])


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


## === cell 30
tmp = pd.DataFrame(accuracy_data, index=['allow_rep','remove_rep'])
tmp


## === cell 31
pre_test = test.preprocess_text
X_test = vectorizer.transform(pre_test)


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


## === cell 33
DATA_PATH = './'
submission_prediction.to_csv(DATA_PATH+'submission.csv',index=False)


## === cell 34
from tensorflow.keras.preprocessing.text import Tokenizer

preprocess_texts = []
for text in train['preprocess_text']:
    preprocess_texts.append(text.split(' '))
    
print(preprocess_texts[:2])


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


## === cell 37
tmp = train[['id','preprocess_text']]

tmp.to_csv(DATA_PATH+'/preprocess_train.csv')

np.save(open(DATA_PATH + 'encoding_train.npy', 'wb'), padded)

y = train[target_cols]
y.to_csv(DATA_PATH+'label_train.csv',index=False)


## === cell 38
preprocess_test = []
for text in test['preprocess_text']:
    preprocess_test.append(text.split(' '))
    
encoded = tokenizer.texts_to_sequences(preprocess_test)

padded_test = pad_sequences(encoded,maxlen=MAX_SEQ_LEN ,padding='post')

tmp2 = test[['id','preprocess_text']]
tmp2.to_csv(DATA_PATH+'/preprocess_test.csv')

np.save(open(DATA_PATH + 'encoding_test.npy', 'wb'), padded_test)
