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

- What this solution (achieved 0.5) has done: 'I fix the runtime errors and guarantee a proper submission file.  
- Read the sample submission from the correct extracted location so the `id` column is present.  
- Wrap the TensorFlow import in a safe try/except; if TensorFlow isn’t available the neural‑network code is skipped, keeping the original logistic‑regression pipeline functional.  
- Guard model creation to run only when TensorFlow loads successfully.  
- Finally, write the submission CSV using the populated DataFrame that now definitely contains the `id` column.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
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
train.drop(["id"], axis=1, inplace=True)




## === cell 6
import re
import string




## === cell 7
train["comment_text"] = train["comment_text"].apply(lambda x: x.lower())
train["comment_text"] = train["comment_text"].apply(
    lambda x: re.sub(r"\w*\d\w*", "", x)
)
train["comment_text"] = train["comment_text"].apply(
    lambda x: re.sub(r"[%s]" % re.escape(string.punctuation), "", x)
)




## === cell 8
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    return text


train["comment_text"] = train["comment_text"].apply(clean_text)




## === cell 9
train.to_csv("train_clean.csv", index=False)




## === cell 10
X = train.comment_text




## === cell 11
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=5000, stop_words="english")




## === cell 12
test_df = test.comment_text.astype("U")




## === cell 13
X_train = vect.fit_transform(X.astype("U"))
X_test = vect.transform(test_df)




## === cell 14
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

logreg = LogisticRegression(
    C=24.0,
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced",
)




## === cell 15
submission = pd.read_csv(
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)




## === cell 16
for label in cols_target:
    print(f"... Processing {label}")
    y = train[label]
    logreg.fit(X_train, y)
    y_pred_train = logreg.predict(X_train)
    print("Training accuracy:", accuracy_score(y, y_pred_train))
    print(classification_report(y, y_pred_train))
    test_prob = logreg.predict_proba(X_test)[:, 1]
    submission[label] = test_prob




## === cell 17
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None
    print("TensorFlow not available – skipping neural‑network part.")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
if tf is not None:
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

    max_features = 20000
    maxlen = 200

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train["comment_text"]))
    X_seq_train = sequence.pad_sequences(
        tokenizer.texts_to_sequences(train["comment_text"]), maxlen=maxlen
    )
    X_seq_test = sequence.pad_sequences(
        tokenizer.texts_to_sequences(test["comment_text"]), maxlen=maxlen
    )

    def get_model():
        embed_size = 128
        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size)(inp)
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        out = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    model = get_model()
    print(model.summary())
    model.fit(
        X_seq_train,
        train[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].values,
        batch_size=32,
        epochs=2,
        validation_split=0.1,
        verbose=1,
    )
    nn_pred = model.predict(X_seq_test)




## === cell 19
submission.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the column: id
