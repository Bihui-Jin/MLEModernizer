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

0.95569

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import os
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import rankdata
import json


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
model = get_model(config)
model.summary()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4066080755.py in <cell line: 0>()
----> 1 model = get_model(config)
      2 model.summary()

/tmp/ipykernel_11/4215442096.py in get_model(config)
      1 def get_model(config):
      2     inputs = keras.Input(shape=(None, ), dtype="string", name="inputs")
----> 3     word2vec_x = get_word2vec_model(config, inputs)
      4     tfidf_x = get_tfidf_model(config, inputs)
      5     x = layers.Concatenate()([word2vec_x, tfidf_x])

/tmp/ipykernel_11/1027164558.py in get_word2vec_model(config, inputs)
      1 def get_word2vec_model(config, inputs):
      2     x = word2vec_vectorizer(inputs)
----> 3     x = PositionalEmbedding(config.sequence_length, config.vocab_size, config.embed_dim)(x)
      4     x = FNetEncoder(config.embed_dim, config.latent_dim)(x)
      5     x = layers.GlobalAveragePooling1D()(x)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1748745899.py in compute_mask(self, inputs, mask)
     20 
     21     def compute_mask(self, inputs, mask=None):
---> 22         return tf.math.not_equal(inputs, 0)

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 18
keras.utils.plot_model(model, show_shapes=True)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/191346248.py in <cell line: 0>()
----> 1 keras.utils.plot_model(model, show_shapes=True)

NameError: name 'model' is not defined

## === cell 19
model.compile(
    "adam", loss="binary_crossentropy", metrics=["categorical_accuracy", keras.metrics.AUC()]
)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/146384657.py in <cell line: 0>()
----> 1 model.compile(
      2     "adam", loss="binary_crossentropy", metrics=["categorical_accuracy", keras.metrics.AUC()]
      3 )

NameError: name 'model' is not defined

## === cell 20
acc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_acc_model_path, monitor="val_categorical_accuracy",save_weights_only=True, save_best_only=True)
auc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_auc_model_path, monitor="val_auc",save_weights_only=True, save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
model.fit(train_ds, epochs=config.epochs, validation_data=valid_ds, callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr])
model.save_weights(config.lastest_model_path)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1952810906.py in <cell line: 0>()
----> 1 acc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_acc_model_path, monitor="val_categorical_accuracy",save_weights_only=True, save_best_only=True)
      2 auc_checkpoint = keras.callbacks.ModelCheckpoint(config.best_auc_model_path, monitor="val_auc",save_weights_only=True, save_best_only=True)
      3 early_stopping = keras.callbacks.EarlyStopping(patience=10)
      4 reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
      5 model.fit(train_ds, epochs=config.epochs, validation_data=valid_ds, callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr])

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=model_best_acc.tf

## === cell 21
from sklearn.metrics import classification_report
y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
cls_report = classification_report(y_val, y_pred)
print(cls_report)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4252127927.py in <cell line: 0>()
      1 from sklearn.metrics import classification_report
----> 2 y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
      3 cls_report = classification_report(y_val, y_pred)
      4 print(cls_report)

NameError: name 'model' is not defined

## === cell 22
test = pd.read_csv("/kaggle/working/test.csv")
test.head()


## === cell 23
sample_submission = pd.read_csv("/kaggle/working/sample_submission.csv")
sample_submission.head()


## === cell 24
scores = []
test_ds = tf.data.Dataset.from_tensor_slices((test["comment_text"])).batch(config.batch_size).cache().prefetch(1)
for path in [config.best_acc_model_path, config.lastest_model_path]:
    model.load_weights("/kaggle/working/" + path)
    score = model.predict(test_ds)
    scores.append(score)
score = np.mean(scores, axis=0)
print(score.shape)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1347804692.py in <cell line: 0>()
      3 #for path in [config.best_acc_model_path, config.best_auc_model_path, config.lastest_model_path]:
      4 for path in [config.best_acc_model_path, config.lastest_model_path]:
----> 5     model.load_weights("/kaggle/working/" + path)
      6     score = model.predict(test_ds)
      7     scores.append(score)

NameError: name 'model' is not defined

## === cell 25
sample_submission[config.labels] = score
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4263208379.py in <cell line: 0>()
----> 1 sample_submission[config.labels] = score
      2 sample_submission.to_csv("submission.csv", index=False)
      3 sample_submission.head()

NameError: name 'score' is not defined
