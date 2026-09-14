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

3.8

# 3. Installed packages

geopandas==0.14.4
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

0.9598777576795992

# 6. Current score

0.6578

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.6578) has done: 'I fixed the file paths, removed the missing `local_sumb` load, corrected the weight‑file extension, expanded the training subset a bit for a better model, and added proper prediction and submission writing so that a valid `submission.csv` is produced.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 2
import numpy as np

np.random.seed(42)
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


import warnings

warnings.filterwarnings("ignore")

import os

os.environ["OMP_NUM_THREADS"] = "4"




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
subm = pd.read_csv(sample_sub_path)  # sample submission template




## === cell 4
print(train.shape, test.shape, subm.shape)




## === cell 5
train.head()




## === cell 6
subm.head()




## === cell 7
train.head()




## === cell 8
text = train["comment_text"]




## === cell 9
text.iloc[0]




## === cell 10
train["comment_text"].iloc[0]




## === cell 11
lens = train.comment_text.str.len()
print(lens.mean(), lens.std(), lens.max())




## === cell 12
lens = test.comment_text.str.len()
print(lens.mean(), lens.std(), lens.max())




## === cell 13
lens.hist()




## === cell 14
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train["none"] = 1 - train[label_cols].max(axis=1)
train.describe()




## === cell 15
print(len(train), len(test))




## === cell 16
COMMENT = "comment_text"
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)




## === cell 17
import re, string

re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")


def tokenize(s):
    return re_tok.sub(r" \1 ", s).split()


def clean(s):
    return re_tok.sub(r" \1 ", s)




## === cell 18
words = []
for t in text:
    words.extend(tokenize(t))
print(words[:100])
vocab = list(set(words))
print(len(words), len(vocab))




## === cell 19
train["comment_text"].iloc[0]




## === cell 20
clean(train["comment_text"].iloc[0])




## === cell 21
def full_one_hot_word_embedding(train_df, test_df, vocab_size=10000, maxlen=500):
    """
    Encode texts with Keras one_hot, pad to `maxlen` and return
    numpy arrays suitable for the model.
    """
    txt_train = [clean(txt) for txt in train_df[COMMENT]]
    txt_test = [clean(txt) for txt in test_df[COMMENT]]

    encoded_train = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train]
    encoded_test = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test]

    pad_train = keras.preprocessing.sequence.pad_sequences(
        encoded_train, padding="post", maxlen=maxlen
    )
    pad_test = keras.preprocessing.sequence.pad_sequences(
        encoded_test, padding="post", maxlen=maxlen
    )
    return pad_train, train_df[label_cols], pad_test




## === cell 22
max_features = 30000
maxlen = 500  # reduced length to keep memory reasonable
embed_size = 300
vocab_size = 10000




## === cell 23
def get_model():
    inp = keras.layers.Input(shape=(maxlen,))
    x = keras.layers.Embedding(vocab_size, 16)(inp)
    x = keras.layers.SpatialDropout1D(0.2)(x)
    x = keras.layers.GRU(80, return_sequences=True)(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    conc = keras.layers.concatenate([avg_pool, max_pool])
    outp = keras.layers.Dense(6, activation="sigmoid")(conc)

    model = keras.models.Model(inputs=inp, outputs=outp)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model




## === cell 24
train_subset = train.sample(n=5000, random_state=42).reset_index(drop=True)
x_train, y_train, x_test = full_one_hot_word_embedding(
    train_subset, test, vocab_size=vocab_size, maxlen=maxlen
)




## === cell 25
print(x_train.shape, y_train.shape, x_test.shape)




## === cell 26
model1 = get_model()




## === cell 27
batch_size = 32
epochs = 2  # a small number of epochs keeps runtime low but gives a bit more learning

X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, train_size=0.9, random_state=233
)

X_tra = X_tra.astype(np.float32)
X_val = X_val.astype(np.float32)
y_tra = y_tra.values.astype(np.float32)
y_val = y_val.values.astype(np.float32)




## === cell 28
hist = model1.fit(
    X_tra,
    y_tra,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1,
)




## === cell 29
model_json = model1.to_json()
with open("my_model1.json", "w") as json_file:
    json_file.write(model_json)

model1.save_weights("my_model1.weights.h5")




## === cell 30
with open("my_model1.json", "r") as json_file:
    loaded_model_json = json_file.read()
model1 = tf.keras.models.model_from_json(loaded_model_json)
model1.load_weights("my_model1.weights.h5")
print("Loaded model from disk")




## === cell 31
def column(matrix, i):
    return [row[i] for row in matrix]




## === cell 32
test_pred = model1.predict(x_test, batch_size=256, verbose=0)

submission = subm.copy()
submission[label_cols] = test_pred
submission.head()




## === cell 33
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
