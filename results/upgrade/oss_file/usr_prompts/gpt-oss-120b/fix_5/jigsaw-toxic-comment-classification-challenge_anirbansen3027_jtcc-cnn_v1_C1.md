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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.82254

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the non‑code text cell, replace the TensorFlow‑specific Keras imports with the native keras imports (fixing the protobuf error), guard the unzip step so it never crashes, and clean up stray backticks that caused a syntax error. I also increase training epochs modestly from 1 to 2 to boost AUC toward the target while keeping the core CNN architecture unchanged. The script now runs end‑to‑end and writes a proper submission.csv file.'
- What this solution (achieved 0.5) has done: 'The fix replaces the TensorFlow‑based Keras imports (which caused a protobuf `MessageFactory` error) with the standalone keras package that works in the environment, and rewrites the submission step to build a DataFrame with the required columns explicitly, guaranteeing that the CSV contains all six label columns plus the `id`. These minimal changes resolve the import crash and the “missing columns” submission error while keeping the original CNN architecture and training settings unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPool1D, Dense
from keras.metrics import AUC

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2
EPOCHS = 3  # a small increase to move AUC toward the target




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
zip_paths = glob.glob(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/*.zip"
)
for zp in zip_paths:
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall("/kaggle/working")
    except Exception:
        pass  # ignore if already extracted or any extraction issue




## === cell 2
def load_csv(filename):
    wk_path = os.path.join("/kaggle/working", filename)
    return pd.read_csv(wk_path if os.path.exists(wk_path) else filename)


df_train = load_csv("train.csv")
df_test = load_csv("test.csv")

train_texts = df_train["comment_text"].astype(str).values
test_texts = df_test["comment_text"].astype(str).values

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_labels = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2686129395.py in <cell line: 0>()
     10 test_texts = df_test["comment_text"].astype(str).values
     11 
---> 12 tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
     13 tokenizer.fit_on_texts(train_texts)
     14 

NameError: name 'Tokenizer' is not defined

## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    train_data,
    train_labels,
    test_size=VALIDATION_SPLIT,
    random_state=123,
    shuffle=True,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2825769088.py in <cell line: 0>()
      1 X_train, X_val, y_train, y_val = train_test_split(
----> 2     train_data,
      3     train_labels,
      4     test_size=VALIDATION_SPLIT,
      5     random_state=123,

NameError: name 'train_data' is not defined

## === cell 4
cnn_model = Sequential(
    [
        Embedding(
            input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH
        ),
        Conv1D(128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=5),
        Conv1D(128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=5),
        Conv1D(128, kernel_size=5, activation="relu"),
        GlobalMaxPool1D(),
        Dense(128, activation="relu"),
        Dense(6, activation="sigmoid"),
    ]
)

cnn_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[AUC()],
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/714678763.py in <cell line: 0>()
----> 1 cnn_model = Sequential(
      2     [
      3         Embedding(
      4             input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH
      5         ),

NameError: name 'Sequential' is not defined

## === cell 5
history = cnn_model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/869297696.py in <cell line: 0>()
----> 1 history = cnn_model.fit(
      2     X_train,
      3     y_train,
      4     batch_size=128,
      5     epochs=EPOCHS,

NameError: name 'cnn_model' is not defined

## === cell 6
y_preds = cnn_model.predict(test_data, batch_size=128, verbose=0)

submission = pd.DataFrame(
    y_preds,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", df_test["id"].values)

submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1682113089.py in <cell line: 0>()
----> 1 y_preds = cnn_model.predict(test_data, batch_size=128, verbose=0)
      2 
      3 submission = pd.DataFrame(
      4     y_preds,
      5     columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],

NameError: name 'cnn_model' is not defined
