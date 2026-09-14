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

3.9

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

0.97403

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 2
import zipfile

z = zipfile.ZipFile(
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"
)
z.extractall()




## === cell 3
train = pd.read_csv(
    "../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
)
test = pd.read_csv(
    "../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
)




## === cell 4
cols_target = ["obscene", "insult", "toxic", "severe_toxic", "identity_hate", "threat"]




## === cell 5
train.head()




## === cell 6
train.isnull().sum()




## === cell 7
train.describe()




## === cell 8
train.drop(["id"], axis=1, inplace=True)




## === cell 9
train.head()




## === cell 10
train.shape




## === cell 11
import re




## === cell 12
train["comment_text"] = train["comment_text"].apply(lambda x: x.lower())




## === cell 13
train["comment_text"] = train["comment_text"].apply(lambda x: re.sub("\w*\d\w*", "", x))




## === cell 14
test.shape




## === cell 15
import string

train["comment_text"] = train["comment_text"].apply(
    lambda x: re.sub("[%s]" % re.escape(string.punctuation), "", x)
)




## === cell 16
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    return text


train["comment_text"] = train["comment_text"].apply(clean_text)
train["comment_text"]




## === cell 17
train




## === cell 18
train.to_csv("train.csv", index=False)




## === cell 19
train.head()




## === cell 20
X = train.comment_text




## === cell 21
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=5000, stop_words="english")
vect




## === cell 22
type(X)




## === cell 23
test.shape




## === cell 24
test_df = test.comment_text




## === cell 25
X_train = vect.fit_transform(X.astype("U"))
X_test = vect.transform(test_df.astype("U"))




## === cell 26
X_test




## === cell 27
X_train




## === cell 28
import sklearn




## === cell 29
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=24.0, max_iter=1000)




## === cell 30
submission_binary1 = pd.read_csv("./sample_submission.csv")
submission_binary1.shape




## === cell 31
from sklearn.metrics import classification_report

for label in cols_target:
    print("... Processing {}".format(label))
    y = train[label]
    logreg.fit(X_train, y)
    y_pred_X = logreg.predict(X_train)
    print("Training accuracy is {}".format(accuracy_score(y, y_pred_X)))
    print(classification_report(y, y_pred_X))
    test_y_prob = logreg.predict_proba(X_test)[:, 1]
    print(test_y_prob.shape)
    submission_binary1[label] = test_y_prob




## === cell 32
train




## === cell 33
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train["none"] = 1 - train[label_cols].max(axis=1)




## === cell 34
train




## === cell 35
max_features = 20000
maxlen = 200




## === cell 36
list_sentences_train = train["comment_text"].values
list_sentences_test = test["comment_text"].values




## === cell 37
y = train[label_cols].values




## === cell 38
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LSTM,
    Bidirectional,
    GlobalMaxPool1D,
    Dropout,
)
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 39
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)




## === cell 40
output_bias = None




## === cell 41
def get_model(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    embed_size = 128
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid", bias_initializer=output_bias)(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


model = get_model()
print(model.summary())
batch_size = 32
epochs = 2




## === cell 42
submission_binary1.to_csv("baseline.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the column: id
