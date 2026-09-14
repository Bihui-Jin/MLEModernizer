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

3.8

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.96744

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.47338) has done: 'Diagnosis: The crash happens because this environment uses Keras 3 (`keras==3.8.0`), where the legacy `keras.preprocessing.text` module is no longer shipped. However, `tf.keras` (TensorFlow 2.18) still provides the legacy preprocessing utilities under `tensorflow.keras.preprocessing`. The fix is to switch the `pad_sequences` and `Tokenizer` imports to `tensorflow.keras.preprocessing` while keeping the rest of the model/training code unchanged. This is a minimal import-level patch that restores the same preprocessing behavior without altering core logic.

Patch summary: Replace `from keras.preprocessing...` imports with `from tensorflow.keras.preprocessing...` to resolve `ModuleNotFoundError` in Keras 3.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: No variables/functions used by later cells are changed; the same names `pad_sequences` and `Tokenizer` remain available with identical expected behavior.

Assumptions: TensorFlow 2.18’s bundled `tf.keras.preprocessing` is available (it is, given installed packages), and later cells rely on these preprocessing utilities rather than KerasNLP alternatives.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sys

if "google.protobuf" not in sys.modules:
    try:
        import google.protobuf as _pb

        _ver = getattr(_pb, "__version__", "0")
    except Exception:
        _ver = "0"
    major = (
        int(str(_ver).split(".", 1)[0]) if str(_ver).split(".", 1)[0].isdigit() else 0
    )
    if major >= 5:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
        )

import matplotlib.pyplot as plt
import re
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    from keras.utils import plot_model
except Exception:
    from tensorflow.keras.utils import plot_model

from sklearn.metrics import roc_auc_score

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



## === cell 2
import os, zipfile

train_zip = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_zip = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"

if os.path.exists(train_zip) and not os.path.exists("./train.csv"):
    with zipfile.ZipFile(train_zip, "r") as z:
        z.extractall(".")
if os.path.exists(test_zip) and not os.path.exists("./test.csv"):
    with zipfile.ZipFile(test_zip, "r") as z:
        z.extractall(".")



## === cell 3
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
else:
    df_train = pd.read_csv("./train.csv")

print("Shape=>", df_train.shape)
df_train.head()



## === cell 4
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
if os.path.exists(test_path):
    df_test = pd.read_csv(test_path)
else:
    df_test = pd.read_csv("./test.csv")

print("Shape=>", df_test.shape)
df_test.head()



## === cell 5
for i in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    print(df_train[i].value_counts(normalize=True) * 100)



## === cell 6
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for ax, class_name in zip(
    axes.flatten(),
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
):
    pd.value_counts(df_train[class_name], sort=True).plot(kind="bar", rot=0, ax=ax)
    ax.set_title("{} Distribution".format(class_name))
    ax.set_xticks([0, 1])
    ax.set_xlabel("Labels")
    ax.set_ylabel("Frequency")

plt.show()




## === cell 7
def cleaner(text):
    text = text.lower()
    text = re.sub("[^a-z]+", " ", text)
    text = re.sub("[ ]+", " ", text)

    return text




## === cell 8
df_train["cleaned"] = df_train["comment_text"].apply(cleaner)



## === cell 9
df_train["comment_text"][:2].values



## === cell 10
df_train["cleaned"][:2].values



## === cell 11
df_test["cleaned"] = df_test["comment_text"].apply(cleaner)



## === cell 12
df_train.head()



## === cell 13
df_test.head()



## === cell 14
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["cleaned"])



## === cell 15
print("Vocabulary Size=>", len(tokenizer.word_index))



## === cell 16
train_seq = tokenizer.texts_to_sequences(df_train["cleaned"])
test_seq = tokenizer.texts_to_sequences(df_test["cleaned"])



## === cell 17
train_seq = pad_sequences(train_seq, maxlen=100, padding="post")
test_seq = pad_sequences(test_seq, maxlen=100, padding="post")



## === cell 18
vocabulary = len(tokenizer.word_index) + 1
print("Vocabulary Size=>", vocabulary)



## === cell 19
print("Shape of train_sequence=>", train_seq.shape)
print("Shape of test_sequence=>", test_seq.shape)



## === cell 20
y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values
print("Shape of Training Labels=>", y_train.shape)



## === cell 21
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "iterative-stratification"]
)



## === cell 22
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



## === cell 23
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)



## === cell 24
for train_index, val_index in msss.split(train_seq, y_train):
    x_train_split, y_train_split = train_seq[train_index], y_train[train_index]
    x_valid_split, y_valid_split = train_seq[val_index], y_train[val_index]



## === cell 25
print("Shape of Train Split=>", x_train_split.shape, y_train_split.shape)
print("Shape of Validation Split=>", x_valid_split.shape, y_valid_split.shape)



## === cell 26
print("Class Distribution of Train Split in Percentage")
for i, v in enumerate(
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
):
    print(v)
    print(pd.Series(y_train_split[:, i]).value_counts(normalize=True) * 100)



## === cell 27
print("Class Distribution of Validation Split in Percentage")
for i, v in enumerate(
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
):
    print(v)
    print(pd.Series(y_valid_split[:, i]).value_counts(normalize=True) * 100)



## === cell 28
input_1 = Input(shape=(100,))
embedding_1 = Embedding(vocabulary, 100)(input_1)
conv_1 = Conv1D(filters=352, kernel_size=7, padding="same")(embedding_1)
dropout_1 = Dropout(0.06675)(conv_1)
pool_1 = GlobalMaxPooling1D()(dropout_1)

dense = Dense(128, activation="relu")(pool_1)
output = Dense(6, activation="sigmoid")(dense)

model = Model(inputs=[input_1], outputs=output)

model.summary()



## === cell 29
plot_model(model, to_file="model.png", show_shapes=True)



## === cell 30
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 31
es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=5, min_delta=1e-5
)
mc = ModelCheckpoint(
    "/kaggle/working/model.keras",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    mode="min",
)



## === cell 32
model.fit(
    x_train_split,
    y_train_split,
    batch_size=512,
    epochs=100,
    verbose=1,
    validation_data=(x_valid_split, y_valid_split),
    callbacks=[es, mc],
)



## === cell 33
best_model = tf.keras.models.load_model("/kaggle/working/model.keras")



## === cell 34
train_pred = best_model.predict(x_train_split, verbose=0)
print("In-sample Evaluation ROC-AUC Score:\n", roc_auc_score(y_train_split, train_pred))



## === cell 35
valid_pred = best_model.predict(x_valid_split, verbose=0)
print("In-sample Evaluation ROC-AUC Score:\n", roc_auc_score(y_valid_split, valid_pred))



## === cell 36
final_pred = best_model.predict(test_seq, verbose=0)



## === cell 37
prob = pd.DataFrame(
    columns=[
        "id",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
)
prob["id"] = df_test["id"]
for index, value in enumerate(
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
):
    prob[value] = final_pred[:, index]



## === cell 38
prob



## === cell 39
prob.to_csv("submission-CNN-final.csv", index=False)
print("Wrote submission-CNN-final.csv with shape:", prob.shape)
