# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
class Config:
    vocab_size = 15000 # Vocabulary Size
    tfidf_vocab_size = 40000
    sequence_length = 100 # Length of sequence
    batch_size = 1024
    validation_split = 0.15
    embed_dim = 256
    latent_dim = 256
    epochs = 10 # Number of Epochs to train
    best_auc_model_path = "model_best_auc.tf"
    best_acc_model_path = "model_best_acc.tf"
    lastest_model_path = "model_latest.tf"
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult",	"identity_hate"]
config = Config()


## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import pathlib
import random
import string
import re
import sys
import numpy as np
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from scipy.stats import rankdata
import json

from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        if hasattr(_message_factory, "GetMessageClass"):
            return _message_factory.GetMessageClass(descriptor)
        raise AttributeError(
            "No compatible GetPrototype/GetMessageClass available in this protobuf version."
        )

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


## === cell 2
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip


## === cell 3
train = pd.read_csv("/kaggle/working/train.csv")
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
    ngrams=2
)
with tf.device("CPU"):
    tfidf_vectorizer.adapt(list(train["comment_text"]))


## === cell 6
word2vec_vectorizer = layers.TextVectorization(
    standardize=custom_standardization, 
    max_tokens=config.vocab_size, 
    output_sequence_length=config.sequence_length
)
with tf.device("CPU"):
    word2vec_vectorizer.adapt(train["comment_text"])


## === cell 7
X_train, X_val, y_train, y_val = train_test_split(train["comment_text"], train[config.labels], test_size=config.validation_split)


## === cell 8
X_train.shape, y_train.shape, X_val.shape, y_val.shape


## === cell 9
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
       dataset = dataset.shuffle(256) 
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(16).repeat(1)
    return dataset


## === cell 10
train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")


## === cell 11
for batch in train_ds.take(1):
    print(batch)


## === cell 12
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


## === cell 13
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
        return tf.math.not_equal(inputs, 0)


## === cell 14
def get_word2vec_model(config, inputs):
    x = word2vec_vectorizer(inputs)
    x = PositionalEmbedding(config.sequence_length, config.vocab_size, config.embed_dim)(x)
    x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    for i in range(3):
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    return x


## === cell 15
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x


## === cell 16
def get_model(config):
    inputs = keras.Input(shape=(None, ), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="model")
    return model


## === cell 17
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


## === cell 18
model = get_model(config)

keras.utils.plot_model(model, show_shapes=True)


## === cell 19
model.compile(
    "adam", loss="binary_crossentropy", metrics=["categorical_accuracy", keras.metrics.AUC()]
)


## === cell 20
acc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_acc_model_path, monitor="val_categorical_accuracy",save_weights_only=True, save_best_only=True)
auc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_auc_model_path, monitor="val_auc",save_weights_only=True, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
model.fit(train_ds, epochs=config.epochs, validation_data=valid_ds, callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr])
model.save_weights(config.lastest_model_path)


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1952810906.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0macc_checkpoint[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m[0;34m.[0m[0mModelCheckpoint[0m[0;34m([0m[0mconfig[0m[0;34m.[0m[0mbest_acc_model_path[0m[0;34m,[0m [0mmonitor[0m[0;34m=[0m[0;34m"val_categorical_accuracy"[0m[0;34m,[0m[0msave_weights_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0msave_best_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mauc_checkpoint[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m[0;34m.[0m[0mModelCheckpoint[0m[0;34m([0m[0mconfig[0m[0;34m.[0m[0mbest_auc_model_path[0m[0;34m,[0m [0mmonitor[0m[0;34m=[0m[0;34m"val_auc"[0m[0;34m,[0m[0msave_weights_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0msave_best_only[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mearly_stopping[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m[0;34m.[0m[0mEarlyStopping[0m[0;34m([0m[0mpatience[0m[0;34m=[0m[0;36m10[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mreduce_lr[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m[0;34m.[0m[0mReduceLROnPlateau[0m[0;34m([0m[0mpatience[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0mmin_delta[0m[0;34m=[0m[0;36m1e-4[0m[0;34m,[0m [0mmin_lr[0m[0;34m=[0m[0;36m1e-6[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mtrain_ds[0m[0;34m,[0m [0mepochs[0m[0;34m=[0m[0mconfig[0m[0;34m.[0m[0mepochs[0m[0;34m,[0m [0mvalidation_data[0m[0;34m=[0m[0mvalid_ds[0m[0;34m,[0m [0mcallbacks[0m[0;34m=[0m[0;34m[[0m[0macc_checkpoint[0m[0;34m,[0m [0mauc_checkpoint[0m[0;34m,[0m [0mreduce_lr[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py[0m in [0;36m__init__[0;34m(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)[0m
[1;32m    182[0m         [0;32mif[0m [0msave_weights_only[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mfilepath[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m".weights.h5"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 184[0;31m                 raise ValueError(
[0m[1;32m    185[0m                     [0;34m"When using `save_weights_only=True` in `ModelCheckpoint`"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    186[0m                     [0;34m", the filepath provided must end in `.weights.h5` "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=model_best_acc.tf

## === cell 21
from sklearn.metrics import classification_report
y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
cls_report = classification_report(y_val, y_pred)
print(cls_report)
