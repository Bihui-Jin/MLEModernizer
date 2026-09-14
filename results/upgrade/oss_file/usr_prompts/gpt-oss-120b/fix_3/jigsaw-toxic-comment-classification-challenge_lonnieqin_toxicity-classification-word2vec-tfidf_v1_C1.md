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

3.10

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
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

0.9556

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # if protobuf is unavailable, let the import fail later (will be caught)

import pandas as pd
import tensorflow as tf
import pathlib
import random
import string
import re
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import rankdata
import json



## === cell 1
pass



## === cell 2
train = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
)
train.head()




## === cell 3
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "", " ")
    text = tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )
    text = tf.strings.regex_replace(text, f"[0-9]+", " ")
    text = tf.strings.regex_replace(text, f"[ ]+", " ")
    text = tf.strings.strip(text)
    return text




## === cell 4
tfidf_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_mode="tf-idf",
    ngrams=2,
)
with tf.device("CPU"):
    tfidf_vectorizer.adapt(list(train["comment_text"]))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/364426083.py in <cell line: 0>()
      1 tfidf_vectorizer = layers.TextVectorization(
      2     standardize=custom_standardization,
----> 3     max_tokens=config.vocab_size,
      4     output_mode="tf-idf",
      5     ngrams=2,

NameError: name 'config' is not defined

## === cell 5
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_sequence_length=config.sequence_length,
)
with tf.device("CPU"):
    word2vec_vectorizer.adapt(train["comment_text"])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1958016034.py in <cell line: 0>()
      1 word2vec_vectorizer = layers.TextVectorization(
      2     standardize=custom_standardization,
----> 3     max_tokens=config.vocab_size,
      4     output_sequence_length=config.sequence_length,
      5 )

NameError: name 'config' is not defined

## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    train["comment_text"],
    train[config.labels],
    test_size=config.validation_split,
    random_state=42,
    stratify=train[config.labels].idxmax(axis=1),  # simple stratification
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3816724645.py in <cell line: 0>()
      1 X_train, X_val, y_train, y_val = train_test_split(
      2     train["comment_text"],
----> 3     train[config.labels],
      4     test_size=config.validation_split,
      5     random_state=42,

NameError: name 'config' is not defined

## === cell 7
X_train.shape, y_train.shape, X_val.shape, y_val.shape




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1159612320.py in <cell line: 0>()
----> 1 X_train.shape, y_train.shape, X_val.shape, y_val.shape
      2 
      3 

NameError: name 'X_train' is not defined

## === cell 8
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
        dataset = dataset.shuffle(256, seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE).repeat(1)
    return dataset




## === cell 9
train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/908744859.py in <cell line: 0>()
----> 1 train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
      2 valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")
      3 

NameError: name 'X_train' is not defined

## === cell 10
for batch in train_ds.take(1):
    print(batch)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2610647300.py in <cell line: 0>()
----> 1 for batch in train_ds.take(1):
      2     print(batch)
      3 
      4 

NameError: name 'train_ds' is not defined

## === cell 11
class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, dropout_rate=0.1, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        inp_complex = tf.cast(inputs, tf.complex64)
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        layer_norm = self.layernorm_2(proj_input + proj_output)
        return layer_norm




## === cell 12
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions




## === cell 13
def get_word2vec_model(config, inputs):
    x = word2vec_vectorizer(inputs)
    x = PositionalEmbedding(
        config.sequence_length, config.vocab_size, config.embed_dim
    )(x)
    x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    for _ in range(3):
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    return x




## === cell 14
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x




## === cell 15
def get_model(config):
    inputs = keras.Input(shape=(), dtype="string", name="inputs")  # scalar string input
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="model")
    return model




## === cell 16
model = get_model(config)
model.summary()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2500331970.py in <cell line: 0>()
----> 1 model = get_model(config)
      2 model.summary()
      3 

NameError: name 'config' is not defined

## === cell 17
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/121849565.py in <cell line: 0>()
----> 1 model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
      2 

NameError: name 'model' is not defined

## === cell 18
acc_checkpoint = keras.callbacks.ModelCheckpoint(
    config.best_acc_model_path,
    monitor="val_binary_accuracy",
    save_weights_only=True,
    save_best_only=True,
    mode="max",
    verbose=1,
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_binary_accuracy",
    patience=5,
    factor=0.5,
    min_delta=1e-4,
    min_lr=1e-6,
    mode="max",
    verbose=1,
)
model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, reduce_lr],
)
model.save_weights(config.lastest_model_path)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3799621516.py in <cell line: 0>()
      1 acc_checkpoint = keras.callbacks.ModelCheckpoint(
----> 2     config.best_acc_model_path,
      3     monitor="val_binary_accuracy",
      4     save_weights_only=True,
      5     save_best_only=True,

NameError: name 'config' is not defined

## === cell 19
from sklearn.metrics import classification_report

y_pred = (model.predict(valid_ds) > 0.5).astype(int)
cls_report = classification_report(y_val, y_pred, zero_division=0)
print(cls_report)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/167808319.py in <cell line: 0>()
      1 from sklearn.metrics import classification_report
      2 
----> 3 y_pred = (model.predict(valid_ds) > 0.5).astype(int)
      4 cls_report = classification_report(y_val, y_pred, zero_division=0)
      5 print(cls_report)

NameError: name 'model' is not defined

## === cell 20
test = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
)
test.head()



## === cell 21
sample_submission = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
sample_submission.head()



## === cell 22
test_ds = (
    tf.data.Dataset.from_tensor_slices(test["comment_text"])
    .batch(config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
score = model.predict(test_ds)
sample_submission[config.labels] = score
submission_path = "submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
sample_submission.head()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3290474773.py in <cell line: 0>()
      1 test_ds = (
      2     tf.data.Dataset.from_tensor_slices(test["comment_text"])
----> 3     .batch(config.batch_size)
      4     .cache()
      5     .prefetch(tf.data.AUTOTUNE)

NameError: name 'config' is not defined
