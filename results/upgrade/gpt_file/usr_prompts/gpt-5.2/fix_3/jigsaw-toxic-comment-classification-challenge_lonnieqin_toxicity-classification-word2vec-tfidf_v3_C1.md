# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
class Config:
    vocab_size = 15000  # Vocabulary Size
    tfidf_vocab_size = 40000
    sequence_length = 100  # Length of sequence
    batch_size = 1024
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    epochs = 10  # Number of Epochs to train

    best_auc_model_path = "model_best_auc.weights.h5"
    best_acc_model_path = "model_best_acc.weights.h5"
    lastest_model_path = "model_latest.weights.h5"

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


config = Config()



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
    except Exception as e:
        print("WARNING: protobuf compatibility step skipped/failed:", repr(e))


_ensure_protobuf_compat()

import pandas as pd
import tensorflow as tf
import pathlib
import random
import string
import re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import rankdata
import json

print("TF:", tf.__version__)



## === cell 2
for fn_zip in [
    "../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip",
    "../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip",
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip",
]:
    if os.path.exists(fn_zip):
        os.system(f"unzip -o -q {fn_zip} -d /kaggle/working")



## === cell 3
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
if not os.path.exists(train_path):
    train_path = "/kaggle/working/train.csv"
train = pd.read_csv(train_path)
train.head()




## === cell 4
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




## === cell 5
tfidf_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.tfidf_vocab_size,
    output_mode="tf-idf",
    ngrams=2,
)
with tf.device("CPU"):
    tfidf_vectorizer.adapt(list(train["comment_text"].astype(str)))



## === cell 6
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=config.vocab_size,
    output_sequence_length=config.sequence_length,
)
with tf.device("CPU"):
    word2vec_vectorizer.adapt(train["comment_text"].astype(str))



## === cell 7
X_train, X_val, y_train, y_val = train_test_split(
    train["comment_text"].astype(str),
    train[config.labels],
    test_size=config.validation_split,
    random_state=42,
)
X_train.shape, y_train.shape, X_val.shape, y_val.shape




## === cell 8
def make_dataset(X, y, batch_size, mode):
    X = tf.convert_to_tensor(X.values if hasattr(X, "values") else X, dtype=tf.string)
    y = tf.convert_to_tensor(y.values if hasattr(y, "values") else y, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def _fix_shape(x, yy):
        x = tf.expand_dims(x, axis=-1)  # ( ) -> (1,)
        return x, yy

    dataset = dataset.map(_fix_shape, num_parallel_calls=tf.data.AUTOTUNE)
    if mode == "train":
        dataset = dataset.shuffle(256, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset




## === cell 9
train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")



## === cell 10
for batch in train_ds.take(1):
    print("X batch shape:", batch[0].shape, "dtype:", batch[0].dtype)
    print("y batch shape:", batch[1].shape, "dtype:", batch[1].dtype)




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

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)




## === cell 13
def get_word2vec_model(config, inputs):
    x = word2vec_vectorizer(inputs)
    x = PositionalEmbedding(
        config.sequence_length, config.vocab_size, config.embed_dim
    )(x)
    x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    for i in range(3):
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
    inputs = keras.Input(shape=(None,), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="model")
    return model




## === cell 16
model = get_model(config)
model.summary()



## === cell 17
try:
    keras.utils.plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 18
model.compile(
    "adam",
    loss="binary_crossentropy",
    metrics=["categorical_accuracy", keras.metrics.AUC(name="auc")],
)



## === cell 19
acc_checkpoint = keras.callbacks.ModelCheckpoint(
    config.best_acc_model_path,
    monitor="val_categorical_accuracy",
    save_weights_only=True,
    save_best_only=True,
)
auc_checkpoint = keras.callbacks.ModelCheckpoint(
    config.best_auc_model_path,
    monitor="val_auc",
    save_weights_only=True,
    save_best_only=True,
)
early_stopping = keras.callbacks.EarlyStopping(patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)

model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr],
)
model.save_weights(config.lastest_model_path)



## === cell 20
from sklearn.metrics import classification_report

y_pred = np.array(model.predict(valid_ds, verbose=0) > 0.5, dtype=int)
cls_report = classification_report(y_val, y_pred, zero_division=0)
print(cls_report)



## === cell 21
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/working/test.csv"
test = pd.read_csv(test_path)
test.head()



## === cell 22
sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)
if not os.path.exists(sub_path):
    sub_path = "/kaggle/working/sample_submission.csv"
sample_submission = pd.read_csv(sub_path)
sample_submission.head()



## === cell 23
test_text = tf.convert_to_tensor(
    test["comment_text"].astype(str).values, dtype=tf.string
)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_text)
    .map(lambda x: tf.expand_dims(x, axis=-1), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

scores = []
for path in [
    config.best_acc_model_path,
    config.best_auc_model_path,
    config.lastest_model_path,
]:
    if os.path.exists(path):
        model.load_weights(path)
        pred = model.predict(test_ds, verbose=0)
        scores.append(pred)
    else:
        print(f"WARNING: weights not found, skipping: {path}")

if len(scores) == 0:
    score = model.predict(test_ds, verbose=0)
else:
    score = np.mean(scores, axis=0)

print("Test prediction shape:", score.shape)



## === cell 24
pred_df = pd.DataFrame(score, columns=config.labels)
pred_df.insert(0, "id", test["id"].values)

sample_submission = sample_submission[["id"] + config.labels].copy()
sample_submission = sample_submission.merge(
    pred_df, on="id", how="left", suffixes=("", "_pred")
)

for c in config.labels:
    sample_submission[c] = sample_submission[f"{c}_pred"].astype(np.float32)
    sample_submission.drop(columns=[f"{c}_pred"], inplace=True)

sample_submission[config.labels] = sample_submission[config.labels].fillna(0.5)

sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
sample_submission.head()
