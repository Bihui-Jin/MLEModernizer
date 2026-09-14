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
import os
import re
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
    ".",
]


def _first_existing_file(relname: str):
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, relname)
        if os.path.exists(p):
            return p
    return None


train_path = _first_existing_file("train.csv")
test_path = _first_existing_file("test.csv")
sub_path = _first_existing_file("sample_submission.csv")

if train_path is None or test_path is None or sub_path is None:
    raise FileNotFoundError(
        f"Could not find required files. Found train={train_path}, test={test_path}, sample_submission={sub_path}"
    )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sub_path)

print(
    "train:", train.shape, "test:", test.shape, "sample_submission:", submission.shape
)
print("train columns:", list(train.columns))
print("test columns:", list(test.columns))



## === cell 2
train.info()



## === cell 3
train.columns



## === cell 4
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 5
print(train.isnull().value_counts())
print("-" * 30)
print(train.isnull().sum())



## === cell 6
print("size of train : {}".format(len(train)))
print("size of test : {}".format(len(test)))
print("-" * 20)
print(train[target_cols].sum().sort_values(ascending=False))



## === cell 7
train["sum_harmful"] = 0
for col in target_cols:
    train["sum_harmful"] += train[col]
train.head()



## === cell 8
train["len_of_text"] = train["comment_text"].fillna("").apply(len)



## === cell 12
train["sum_harmful"].value_counts()



## === cell 13
train.describe()




## === cell 14
def clean_text(text):
    text = str(text).lower()
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
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ")
    return text




## === cell 15
train["preprocess_text"] = train["comment_text"].fillna("").apply(clean_text)
train.head(3)



## === cell 16
cv = CountVectorizer()
documents = train.comment_text.fillna("").tolist()
documents = [" ".join(documents)]

X = cv.fit_transform(documents).toarray()
freqs = X.transpose().flatten()
words = cv.get_feature_names_out()

df_word = pd.DataFrame({"word": words, "freq": freqs})
df_word = df_word.sort_values(by="freq", ascending=False).reset_index(drop=True)
df_word[:10]



## === cell 17
import nltk
from nltk.corpus import stopwords as nltk_stopwords

for resource in ["corpora/stopwords", "tokenizers/punkt", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(resource.split("/")[-1], quiet=True)
        except Exception:
            pass

stopwords_list = df_word.word.tolist()[:70]
print("학습 데이터로 만든 불용어 개수 : ", len(stopwords_list))
print(stopwords_list[:10], "...")

stopwords_list += nltk_stopwords.words("english")
stopwords = set(stopwords_list)
print(" ")
print("최종 stopwords 개수 :", len(stopwords))



## === cell 18
from nltk.tokenize import word_tokenize


def remove_stopwords(text):
    stopwords_list = stopwords
    word_tokens = word_tokenize(str(text))
    result = []
    for w in word_tokens:
        if len(w) > 2 and w not in stopwords_list:
            result.append(w)
    return " ".join(result)


def remove_special(text, lower=True):
    text = str(text)
    if lower:
        text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    return text


def remove_repeat(text, repeat=1):
    text = str(text).split(" ")
    result = []
    for word in text:
        if result.count(word) < repeat:
            result.append(word)
    return " ".join(result)




## === cell 19
train["preprocess_text"] = train["preprocess_text"].apply(remove_special)
train["preprocess_text"] = train["preprocess_text"].apply(remove_stopwords)



## === cell 20
test["preprocess_text"] = test["comment_text"].fillna("").apply(clean_text)
test["preprocess_text"] = test["preprocess_text"].apply(remove_special)
test["preprocess_text"] = test["preprocess_text"].apply(remove_stopwords)




## === cell 21
def num_of_word(text):
    return len(str(text).split(" "))


tmp = train["preprocess_text"].apply(num_of_word)
print(tmp.describe())



## === cell 22
train["num_of_word"] = train["preprocess_text"].apply(num_of_word)
train[(train["num_of_word"] > 500) & (train["num_of_word"] < 1000)].head(3)



## === cell 23
train["remove_repeat"] = train["preprocess_text"].apply(remove_repeat)
test["remove_repeat"] = test["preprocess_text"].apply(remove_repeat)

train.head(3)



## === cell 25
preprocess_text = train.preprocess_text

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(preprocess_text)
y = train[target_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("X_train len: ", X_train.shape[0])
print("X_valid len:  ", X_valid.shape[0])



## === cell 26
from collections import defaultdict

accuracy_data = defaultdict(list)
for col in target_cols:
    print(" ")
    print("----prediction of {} column----".format(col))
    print(" ")
    y_col = y_train[col]
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_col)
    y_pred = model.predict(X_valid)
    print(
        "Testing accuracy is {}".format(round(accuracy_score(y_valid[col], y_pred), 5))
    )
    accuracy_data[col].append(round(accuracy_score(y_valid[col], y_pred), 5))



## === cell 27
remove_repeat_text = train.remove_repeat

vectorizer2 = TfidfVectorizer(max_features=5000)

X2 = vectorizer2.fit_transform(remove_repeat_text)
y = train[target_cols]

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
    X2, y, test_size=0.2, random_state=0
)

print("X_train len: ", X_train2.shape[0])
print("X_valid len:  ", X_valid2.shape[0])



## === cell 28
for col in target_cols:
    print(" ")
    print("----prediction of {} column----".format(col))
    print(" ")
    y_col = y_train2[col]
    model2 = LogisticRegression(max_iter=1000, solver="liblinear")
    model2.fit(X_train2, y_col)
    y_pred2 = model2.predict(X_valid2)
    print(
        "Testing accuracy is {}".format(
            round(accuracy_score(y_valid2[col], y_pred2), 5)
        )
    )
    accuracy_data[col].append(round(accuracy_score(y_valid2[col], y_pred2), 5))



## === cell 29
tmp_acc = pd.DataFrame(accuracy_data, index=["allow_rep", "remove_rep"])
tmp_acc



## === cell 30
pre_test = test.preprocess_text
X_test = vectorizer.transform(pre_test)



## === cell 31
submission_prediction = submission.copy()

vectorizer = TfidfVectorizer(max_features=5000)
X_full = vectorizer.fit_transform(train.preprocess_text)
X_test_full = vectorizer.transform(test.preprocess_text)

y_full = train[target_cols]

for col in target_cols:
    print(" ")
    print("----prediction of {} column----".format(col))
    y_col = y_full[col]
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_full, y_col)
    test_y_prob = model.predict_proba(X_test_full)[:, 1]
    submission_prediction[col] = test_y_prob



## === cell 32
required_cols = ["id"] + target_cols
submission_prediction = submission_prediction[required_cols]
submission_prediction.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_prediction.shape)
print(submission_prediction.head())



## === cell 33
preprocess_texts = [
    str(text).split(" ") for text in train["preprocess_text"].fillna("")
]
print(preprocess_texts[:2])



## === cell 34
tokenizer = Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(preprocess_texts)

word_vocab = tokenizer.word_index
word_vocab[""] = 0
print("총 voca 개수: ", len(word_vocab))



## === cell 35
encoded = tokenizer.texts_to_sequences(preprocess_texts)

MAX_SEQ_LEN = max((len(item) for item in encoded), default=1)

print("padding len: ", MAX_SEQ_LEN)
padded = pad_sequences(encoded, maxlen=MAX_SEQ_LEN, padding="post")
print("train data -> padding shape: {}".format(padded.shape))



## === cell 36
DATA_PATH = "./"
tmp3 = train[["id", "preprocess_text"]]
tmp3.to_csv(os.path.join(DATA_PATH, "preprocess_train.csv"), index=False)

np.save(open(os.path.join(DATA_PATH, "encoding_train.npy"), "wb"), padded)

y_save = train[target_cols]
y_save.to_csv(os.path.join(DATA_PATH, "label_train.csv"), index=False)

print("Saved preprocess_train.csv, encoding_train.npy, label_train.csv")



## === cell 37
preprocess_test = [str(text).split(" ") for text in test["preprocess_text"].fillna("")]
encoded_test = tokenizer.texts_to_sequences(preprocess_test)
padded_test = pad_sequences(encoded_test, maxlen=MAX_SEQ_LEN, padding="post")

tmp4 = test[["id", "preprocess_text"]]
tmp4.to_csv(os.path.join(DATA_PATH, "preprocess_test.csv"), index=False)

np.save(open(os.path.join(DATA_PATH, "encoding_test.npy"), "wb"), padded_test)

print("Saved preprocess_test.csv, encoding_test.npy")

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'threat', 'toxic', 'insult', 'identity_hate', 'severe_toxic', 'obscene'}
