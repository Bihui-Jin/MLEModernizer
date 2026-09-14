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

3.6

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4
xgboost==2.0.3

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

0.08791

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21939) has done: 'I fix the pipeline so it runs end-to-end in your Kaggle environment and writes a valid `result.csv` submission with the required columns/order. The main runtime breaks are due to (1) pandas `DataFrame.append()` removal, (2) NLTK stopwords resource not being available, and (3) old Keras 1/2-era imports/APIs (`keras.layers.recurrent`, `nb_epoch`, etc.) that don’t work with Keras 3. I keep the same overall approach (tokenize → pad → simple Embedding+pooling+Dense model trained per label) and only update imports/API calls to their Keras 3 equivalents; I also ensure each label trains from a fresh model to avoid cross-label contamination. Finally, I write the submission with exactly the columns from `sample_submission.csv` and a `.csv` suffix.'
- What this solution (achieved 0.7479) has done: 'I fix the runtime error in the Keras import cell by avoiding the `tf_keras` stack that triggers the protobuf `MessageFactory.GetPrototype` incompatibility, and instead use TensorFlow’s bundled `tensorflow.keras` APIs that are stable in Kaggle. I keep the same core approach (Tokenizer → pad_sequences → Embedding+GlobalAveragePooling1D+Dense(2 softmax), trained separately per label for 2 epochs). I also fix a small logic issue in inference: with a 2-class softmax the toxic probability should be the probability of class “1”, not class “0”, which should improve ROC AUC toward your target. Finally, the script still write a valid `result.csv` with the exact sample submission column order.'
- What this solution (achieved 0.78469) has done: 'I fix the runtime crash in the TensorFlow/Keras import cell that’s caused by an incompatible protobuf stack (the `MessageFactory.GetPrototype` error) by switching to the `tf_keras` package that’s installed in your environment and is compatible here. I keep the exact same tokenization, padding, model architecture (Embedding → GlobalAveragePooling1D → Dense(2 softmax)), and per-label training loop. I also add small safety guards to ensure the vocabulary size is valid and that label/text nulls don’t break tokenization, without changing the modeling approach. The script still write a valid `result.csv` with the exact column order from `sample_submission.csv`.'
- What this solution (achieved 0.74336) has done: 'I fix the runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to the stable `tensorflow.keras` API while keeping the exact same tokenization, padding, model architecture, and per-label training loop. I also add a small safety clamp to keep `nb_words >= 2` so the Embedding layer never receives an invalid vocabulary size (score-neutral). Finally, I ensure the submission is written as `result.csv` with the exact column order from `sample_submission.csv` and that predictions align with test `id`s.'
- What this solution (achieved 0.78148) has done: 'You’re hitting a protobuf/Keras import incompatibility when importing `tensorflow`/`tensorflow.keras`, so the pipeline stops before training/inference and can’t reliably produce a submission. The minimal fix is to switch the deep-learning imports to the already-installed `tf_keras` package (which is compatible in this environment) while keeping the exact same tokenization, padding, model architecture, and per-label training loop. I also keep seeds for determinism and preserve the submission column order exactly as `sample_submission.csv`. These changes are runtime/stability fixes and should keep (or slightly improve) score behavior without altering the core approach.'

# 9. Code solution

## === cell 0
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk import word_tokenize, ngrams
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud, STOPWORDS
import xgboost as xgb

np.random.seed(25)



## === cell 1
BASE_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
]


def _find_file(filename):
    for base in BASE_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    for base in BASE_CANDIDATES:
        p = os.path.join(
            base, "jigsaw-toxic-comment-classification-challenge", filename
        )
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in known Kaggle paths.")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

train.shape, test.shape, sample_sub.shape



## === cell 2
train.head()



## === cell 3
train["comment_text"].iloc[0]



## === cell 4
train["comment_text"].iloc[1]



## === cell 5
train.isnull().sum(axis=0)



## === cell 6
types = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train[types].describe()



## === cell 7
count_list = []
for i in types:
    count_list.append(train[i].sum())

plt.figure(figsize=(10, 4))
sns.barplot(x=types, y=count_list)
plt.xticks(rotation=30)
plt.tight_layout()



## === cell 8
types = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sampled_train1 = train[
    (train["toxic"] == 0)
    & (train["severe_toxic"] == 0)
    & (train["obscene"] == 0)
    & (train["threat"] == 0)
    & (train["insult"] == 0)
    & (train["identity_hate"] == 0)
]

sampled_train2 = train[
    (train["toxic"] != 0)
    | (train["severe_toxic"] != 0)
    | (train["obscene"] != 0)
    | (train["threat"] != 0)
    | (train["insult"] != 0)
    | (train["identity_hate"] != 0)
]

sampled_train2.head()



## === cell 9
sampled_train = pd.concat(
    [sampled_train2, sampled_train1.iloc[:16223]], axis=0, ignore_index=True
)
sampled_train = sampled_train.sample(frac=1.0, random_state=25).reset_index(drop=True)
sampled_train.head()



## === cell 10
import string
import itertools
import re
from nltk.stem import WordNetLemmatizer
from string import punctuation

try:
    _ = stopwords.words("english")
except LookupError:
    try:
        nltk.download("stopwords")
    except Exception:
        pass

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()


def cleanData(
    text, lowercase=False, remove_stops=False, stemming=False, lemmatization=False
):
    txt = str(text)

    txt = txt.replace("isn't", "is not")
    txt = txt.replace("aren't", "are not")
    txt = txt.replace("ain't", "am not")
    txt = txt.replace("won't", "will not")
    txt = txt.replace("didn't", "did not")
    txt = txt.replace("shan't", "shall not")
    txt = txt.replace("haven't", "have not")
    txt = txt.replace("hadn't", "had not")
    txt = txt.replace("hasn't", "has not")
    txt = txt.replace("don't", "do not")
    txt = txt.replace("wasn't", "was not")
    txt = txt.replace("weren't", "were not")
    txt = txt.replace("doesn't", "does not")
    txt = txt.replace("'s", " is")
    txt = txt.replace("'re", " are")
    txt = txt.replace("'m", " am")
    txt = txt.replace("'d", " would")
    txt = txt.replace("'ll", " will")
    txt = txt.replace("--th", " ")

    txt = re.sub(r"alot", "a lot", txt)
    txt = re.sub(r"what's", "", txt)
    txt = re.sub(r"What's", "", txt)
    txt = re.sub(r"\'s", " ", txt)
    txt = txt.replace("pic", "picture")
    txt = re.sub(r"\'ve", " have ", txt)
    txt = re.sub(r"can't", "cannot ", txt)
    txt = re.sub(r"n't", " not ", txt)
    txt = re.sub(r"I'm", "I am", txt)
    txt = re.sub(r" m ", " am ", txt)
    txt = re.sub(r"\'re", " are ", txt)
    txt = re.sub(r"\'d", " would ", txt)
    txt = re.sub(r"\'ll", " will ", txt)
    txt = re.sub(r"60k", " 60000 ", txt)
    txt = re.sub(r" e g ", " eg ", txt)
    txt = re.sub(r" b g ", " bg ", txt)
    txt = re.sub(r"\0s", "0", txt)
    txt = re.sub(r" 9 11 ", "911", txt)
    txt = re.sub(r"e-mail", "email", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = re.sub(r"quikly", "quickly", txt)
    txt = re.sub(r"imrovement", "improvement", txt)
    txt = re.sub(r"intially", "initially", txt)
    txt = re.sub(r"quora", "Quora", txt)
    txt = re.sub(r" dms ", "direct messages ", txt)
    txt = re.sub(r"demonitization", "demonetization", txt)
    txt = re.sub(r"actived", "active", txt)
    txt = re.sub(r"kms", " kilometers ", txt)
    txt = re.sub(r"KMs", " kilometers ", txt)
    txt = re.sub(r" cs ", " computer science ", txt)
    txt = re.sub(r" upvotes ", " up votes ", txt)
    txt = re.sub(r" iPhone ", " phone ", txt)
    txt = re.sub(r"\0rs ", " rs ", txt)
    txt = re.sub(r"calender", "calendar", txt)
    txt = re.sub(r"ios", "operating system", txt)
    txt = re.sub(r"gps", "GPS", txt)
    txt = re.sub(r"gst", "GST", txt)
    txt = re.sub(r"programing", "programming", txt)
    txt = re.sub(r"bestfriend", "best friend", txt)
    txt = re.sub(r"dna", "DNA", txt)
    txt = re.sub(r"III", "3", txt)
    txt = re.sub(r"the US", "America", txt)
    txt = re.sub(r"Astrology", "astrology", txt)
    txt = re.sub(r"Method", "method", txt)
    txt = re.sub(r"Find", "find", txt)
    txt = re.sub(r"banglore", "Banglore", txt)
    txt = re.sub(r" J K ", " JK ", txt)
    txt = re.sub(r"comfy", "comfortable", txt)
    txt = re.sub(r"colour", "color", txt)
    txt = re.sub(r"travellers", "travelers", txt)

    txt = re.sub(r"^https?:\/\/.*[\r\n]*", " ", txt, flags=re.MULTILINE)
    txt = re.sub(r"[\w\.-]+@[\w\.-]+", " ", txt, flags=re.MULTILINE)

    txt = "".join("".join(s)[:2] for _, s in itertools.groupby(txt))
    txt = "".join([c for c in txt if c not in punctuation])

    txt = re.sub(r"[^A-Za-z\s]", r" ", txt)
    txt = re.sub(r"\n", r" ", txt)

    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])

    if remove_stops and stop_words:
        txt = " ".join([w for w in txt.split() if w not in stop_words])

    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos="v") for w in txt.split()])

    return txt




## === cell 11
sampled_train["comment_text"] = sampled_train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

sampled_train["comment_text"] = sampled_train["comment_text"].map(
    lambda x: cleanData(
        x, lowercase=True, remove_stops=True, stemming=True, lemmatization=True
    )
)
test["comment_text"] = test["comment_text"].map(
    lambda x: cleanData(
        x, lowercase=True, remove_stops=True, stemming=True, lemmatization=True
    )
)

sampled_train[["comment_text"] + types].head()



## === cell 12
train.columns



## === cell 13
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

np.random.seed(25)
random.seed(25)
try:
    keras.utils.set_random_seed(25)
except Exception:
    pass



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
MAX_SEQUENCE_LENGTH = 256
MAX_NB_WORDS = 200000  # keep same value as original intent (cap is optional here)

tokenizer = Tokenizer(lower=False, filters="")
tokenizer.fit_on_texts(sampled_train["comment_text"].tolist())

sequences = tokenizer.texts_to_sequences(sampled_train["comment_text"].tolist())
test_sequences = tokenizer.texts_to_sequences(test["comment_text"].tolist())

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of train data tensor:", train_data.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of test data tensor:", test_data.shape)

nb_words = int(min(MAX_NB_WORDS, len(tokenizer.word_index) + 1))
nb_words = max(nb_words, 2)  # safety guard: Embedding input_dim must be >= 2
nb_words




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433752090.py in <cell line: 0>()
      2 MAX_NB_WORDS = 200000  # keep same value as original intent (cap is optional here)
      3 
----> 4 tokenizer = Tokenizer(lower=False, filters="")
      5 tokenizer.fit_on_texts(sampled_train["comment_text"].tolist())
      6 

NameError: name 'Tokenizer' is not defined

## === cell 15
def build_model(nb_words, max_len):
    m = Sequential()
    m.add(Embedding(nb_words, 20, input_length=max_len))
    m.add(GlobalAveragePooling1D())
    m.add(Dense(2, activation="softmax"))
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m




## === cell 16
li = []
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for lab in labels:
    y = np.array(sampled_train[lab].astype(int).values)
    model = build_model(nb_words, MAX_SEQUENCE_LENGTH)

    model.fit(
        train_data,
        to_categorical(y, num_classes=2),
        validation_split=0.2,
        epochs=2,
        batch_size=128,
        verbose=2,
    )

    li.append(model.predict(test_data, batch_size=1024, verbose=0)[:, 1])

len(li), [arr.shape for arr in li[:1]]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/52591369.py in <cell line: 0>()
      4 for lab in labels:
      5     y = np.array(sampled_train[lab].astype(int).values)
----> 6     model = build_model(nb_words, MAX_SEQUENCE_LENGTH)
      7 
      8     model.fit(

NameError: name 'nb_words' is not defined

## === cell 17
result = pd.DataFrame({"id": test["id"].values})

for idx, col in enumerate(types):
    result[col] = li[idx]

result = result[sample_sub.columns.tolist()]

out_path = "result.csv"
result.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", result.shape)
result.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3302501198.py in <cell line: 0>()
      2 
      3 for idx, col in enumerate(types):
----> 4     result[col] = li[idx]
      5 
      6 # Ensure exact required column order (id first, then the six labels)

IndexError: list index out of range
