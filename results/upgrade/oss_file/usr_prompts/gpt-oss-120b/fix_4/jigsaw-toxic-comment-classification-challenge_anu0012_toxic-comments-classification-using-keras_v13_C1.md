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

0.92858

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.92542) has done: 'I fixed the DataFrame concatenation, added missing imports for the newer Keras API, corrected tokenisation, switched the model to a sigmoid output with binary‑crossentropy (appropriate for multi‑label), updated the training call to use `epochs`, and ensured the pipeline creates the `result.csv` submission file with the correct columns.'
- What this solution (achieved 0.92858) has done: 'I fixed the import errors caused by the newer Keras API by switching to TensorFlow’s `tf.keras` implementation, which resolves the `MessageFactory` protobuf issue and properly defines `Tokenizer`, `Embedding`, and other layers. The rest of the pipeline remains unchanged, so the model trains and a `result.csv` file with the correct columns is created.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re, itertools, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

np.random.seed(25)




## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




## === cell 2
types = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sampled_train1 = train[(train[types] == 0).all(axis=1)]
sampled_train2 = train[(train[types] != 0).any(axis=1)]

sampled_train = pd.concat(
    [sampled_train2, sampled_train1.iloc[:16223]], ignore_index=True
)
sampled_train = sampled_train.sample(frac=1, random_state=25).reset_index(drop=True)




## === cell 3
stop_words = set(stopwords.words("english"))


def cleanData(
    text, lowercase=False, remove_stops=False, stemming=False, lemmatization=False
):
    txt = str(text)
    txt = txt.replace("isn't", "is not").replace("aren't", "are not")
    txt = txt.replace("ain't", "am not").replace("won't", "will not")
    txt = txt.replace("didn't", "did not").replace("shan't", "shall not")
    txt = txt.replace("haven't", "have not").replace("hadn't", "had not")
    txt = txt.replace("hasn't", "has not").replace("don't", "do not")
    txt = txt.replace("wasn't", "was not").replace("weren't", "were not")
    txt = txt.replace("doesn't", "does not").replace("'s", " is")
    txt = txt.replace("'re", " are").replace("'m", " am").replace("'d", " would")
    txt = txt.replace("'ll", " will").replace("--th", " ")
    txt = re.sub(r"alot", "a lot", txt)
    txt = re.sub(r"what's", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = re.sub(r"^https?:\/\/.*$", " ", txt, flags=re.MULTILINE)
    txt = re.sub(r"[\w\.-]+@[\w\.-]+", " ", txt, flags=re.MULTILINE)
    txt = re.sub(r"[^A-Za-z\s]", " ", txt)
    txt = re.sub(r"\n", " ", txt)

    if lowercase:
        txt = txt.lower()
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stop_words])
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        txt = " ".join([lemmatizer.lemmatize(w, pos="v") for w in txt.split()])

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




## === cell 5
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
MAX_SEQUENCE_LENGTH = 256
MAX_NB_WORDS = 200000




## === cell 7
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, filters="")
tokenizer.fit_on_texts(sampled_train["comment_text"])

train_sequences = tokenizer.texts_to_sequences(sampled_train["comment_text"])
test_sequences = tokenizer.texts_to_sequences(test["comment_text"])

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index) + 1)




## === cell 8
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = sampled_train[labels].values.astype("float32")




## === cell 9
model = Sequential()
model.add(Embedding(nb_words, 32, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(6, activation="sigmoid"))  # sigmoid for multi‑label
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()




## === cell 10
model.fit(train_data, y, validation_split=0.2, epochs=2, batch_size=16, verbose=2)




## === cell 11
pred = model.predict(test_data, batch_size=32)




## === cell 12
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[labels] = pred
sample_submission.to_csv("result.csv", index=False)




## === cell 13
sample_submission.head()
