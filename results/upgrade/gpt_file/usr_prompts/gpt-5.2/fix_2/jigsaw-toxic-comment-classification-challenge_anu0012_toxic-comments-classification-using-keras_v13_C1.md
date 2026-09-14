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

0.69469

# 6. Current score

0.91382

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.91382) has done: 'I fix the pipeline so it runs end-to-end under modern pandas and Keras in this Kaggle environment, while keeping the same overall approach (downsampling non-toxic, heavy text cleaning, Tokenizer+pad_sequences, small CNN, then write submission). The main runtime issues are (1) `DataFrame.append` removal in pandas, (2) NLTK stopwords resource availability, (3) legacy/invalid Keras imports (and an old protobuf-related crash), and (4) using an incompatible softmax/categorical loss for multi-label targets. I make minimal, directly-relevant fixes: use `pd.concat`, add safe NLTK downloads, switch to `tf_keras` imports, and compile with sigmoid + binary crossentropy (matching the multi-label AUC metric). Finally, I ensure the submission is written as `result.csv` with the required columns in the correct order and aligned row count.'

# 9. Code solution

## === cell 0
import os
import sys
import re
import string
import itertools
import numpy as np
import pandas as pd

np.random.seed(25)



## === cell 1
BASE_INPUT = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input"

train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train.shape, test.shape)
print(train.columns.tolist())



## === cell 2
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

sampled_train = pd.concat(
    [sampled_train2, sampled_train1.iloc[:16223]], axis=0, ignore_index=True
)
sampled_train = sampled_train.sample(frac=1, random_state=25).reset_index(drop=True)

print(sampled_train.shape)
print(sampled_train[types].sum())



## === cell 3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))


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

    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stop_words])

    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos="v") for w in txt.split()])

    return txt




## === cell 4
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

print(sampled_train["comment_text"].head())



## === cell 5
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
)

MAX_SEQUENCE_LENGTH = 256
MAX_NB_WORDS = 200000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, filters="")
tokenizer.fit_on_texts(sampled_train["comment_text"])

sequences = tokenizer.texts_to_sequences(sampled_train["comment_text"])
test_sequences = tokenizer.texts_to_sequences(test["comment_text"])

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index) + 1)

print("Shape of train data tensor:", train_data.shape)
print("Shape of test data tensor:", test_data.shape)
print("nb_words:", nb_words)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
model = Sequential()
model.add(Embedding(nb_words, 32, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(6, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()



## === cell 7
labels = types
y = sampled_train[labels].values.astype(np.float32)

history = model.fit(
    train_data, y, validation_split=0.2, epochs=2, batch_size=16, verbose=2
)



## === cell 8
pred = model.predict(test_data, batch_size=1024, verbose=1)
pred = np.clip(pred, 0.0, 1.0)
print(pred.shape)
print(pred[:2])



## === cell 9
sample_submission = pd.read_csv(sub_path)

sample_submission[labels] = pred
sample_submission = sample_submission[["id"] + labels]

out_path = "result.csv"
sample_submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", sample_submission.shape)
print(sample_submission.head())
