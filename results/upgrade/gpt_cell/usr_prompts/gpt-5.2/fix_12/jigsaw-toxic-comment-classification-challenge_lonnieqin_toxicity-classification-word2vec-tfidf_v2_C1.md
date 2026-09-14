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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during imports in cell 1, before any model/data logic runs. With `protobuf==6.33.0`, some TensorFlow/Keras import paths can indirectly hit older generated-proto code that still expects `google.protobuf.message_factory.MessageFactory.GetPrototype`, which was removed in protobuf 6. This manifests as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when importing TensorFlow. The minimal fix is to force protobuf to use the pure-Python implementation (which maintains compatibility for this call path) before importing TensorFlow.

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2) via `os.environ` *before* importing `tensorflow`. Keep all existing imports and downstream semantics unchanged.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1, and the traceback (`'MessageFactory' object has no attribute 'GetPrototype'`) indicates an incompatibility between the installed `protobuf==6.33.0` runtime and TensorFlow 2.18’s expected protobuf API. The environment variable forcing the pure-Python protobuf implementation does not resolve this API mismatch, so TensorFlow import fails immediately.  

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` back to the default C++ implementation (by removing/overriding the forced `"python"` setting) before importing TensorFlow. This avoids the missing `GetPrototype` attribute path and lets TensorFlow load normally with the installed protobuf. No other logic is changed.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: All imported symbols (`pd`, `tf`, `keras`, `layers`, etc.) remain available with the same names, so subsequent cells run unchanged.

Assumptions: This environment includes the default protobuf C++ implementation (standard in typical Kaggle/TF builds), and TensorFlow 2.18 works with it here.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during imports in cell 1, before any model code runs. With `protobuf==6.33.0`, some TensorFlow/Keras submodules (or transitively imported dependencies) still expect older protobuf APIs (e.g., `MessageFactory.GetPrototype`), causing the `AttributeError`. The standard fix is to force protobuf to use the pure-Python implementation, which preserves the expected API surface for these older call sites. We set the required environment variables *before* importing TensorFlow (and anything that might import protobuf), instead of removing them.

Patch summary: In cell 1, replace the two `os.environ.pop(...)` lines with `os.environ.setdefault(...)` to enforce `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) prior to importing TensorFlow/Keras. This is a minimal, localized change that only affects protobuf runtime selection and unblocks the notebook.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: No variable names, imports, or interfaces used by later cells are changed; cell 2 and beyond can run as-is once imports succeed.

Assumptions: The environment allows setting `os.environ` before TensorFlow import (standard in notebooks/scripts), and using the pure-Python protobuf implementation is acceptable for this workload.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during imports in cell 1 because `protobuf==6.33.0` is incompatible with some installed TensorFlow-related packages (commonly TF/TF-Hub/TF-Text) that still expect the older protobuf API `MessageFactory.GetPrototype`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not fix that API removal in protobuf 6. The minimal deterministic fix is to force protobuf to use the legacy pure-Python implementation behavior at runtime by pinning the protobuf module to the compatible major version via a safe fallback: importing TensorFlow only after ensuring protobuf uses the expected API surface. We do this by importing `google.protobuf` first and, if the newer API is detected, raise a clear error with guidance to downgrade; but since we cannot change environment packages, the practical in-notebook workaround is to avoid importing TensorFlow before setting the env vars and to import TensorFlow in a guarded way that falls back to importing only the non-TF modules needed later in the notebook.

Patch summary: Update cell 1 to (1) set protobuf env vars before any protobuf/TensorFlow import, (2) import protobuf and detect the incompatible API early, and (3) import TensorFlow only after that check to prevent the immediate AttributeError crash.

Updated cells / Compatibility notes for cell k+1 / Assumptions: Cell 2 only unzips data and does not depend on TensorFlow symbols, so this change preserves compatibility. Assumption: later cells can proceed once TensorFlow successfully imports; if the environment truly cannot support TF with protobuf 6, the early check provide an actionable error instead of an opaque AttributeError.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 24 builds `test_ds` from a 1D string tensor shaped `(batch,)`, but the model was built with `Input(shape=(None,), dtype="string")`, which expects each example to be rank-1 (`(None,)`) and therefore each batch to be rank-2 (`(batch, None)`). During training you fixed this by expanding dims in `make_dataset` (cell 20), but the test dataset pipeline in cell 24 doesn’t apply the same rank fix, causing Keras to raise an “Invalid input shape” error at `model.predict`.  
Patch summary: Update only cell 24 to expand the per-example string tensor rank (scalar -> `(1,)`) before batching, matching the training/validation input structure. Keep the rest of the inference logic (weights loading, ensembling, mean) unchanged.  
Updated cells: Only cell 24 is changed.  
Compatibility notes for cell k+1: `score` remains a NumPy array of shape `(len(test), 6)` and dtype compatible with assigning into `sample_submission[config.labels]` in cell 25.  
Assumptions: The model expects a single string tensor input; expanding dims to `(1,)` per example is consistent with the training dataset fix in cell 20 and does not change prediction semantics.'

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
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def _fix_input_rank(x, y_):
        x = tf.expand_dims(x, axis=-1)  # () -> (1,) per example, (batch,) -> (batch, 1)
        return x, y_

    dataset = dataset.map(_fix_input_rank, num_parallel_calls=tf.data.AUTOTUNE)
    if mode == "train":
        dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(16).repeat(1)
    return dataset


train_ds = make_dataset(X_train, y_train, batch_size=config.batch_size, mode="train")
valid_ds = make_dataset(X_val, y_val, batch_size=config.batch_size, mode="valid")

best_acc_path = config.best_acc_model_path
best_auc_path = config.best_auc_model_path
latest_path = config.lastest_model_path

if not best_acc_path.endswith(".weights.h5"):
    best_acc_path = best_acc_path.rsplit(".", 1)[0] + ".weights.h5"
if not best_auc_path.endswith(".weights.h5"):
    best_auc_path = best_auc_path.rsplit(".", 1)[0] + ".weights.h5"
if not latest_path.endswith(".weights.h5"):
    latest_path = latest_path.rsplit(".", 1)[0] + ".weights.h5"

acc_checkpoint = keras.callbacks.ModelCheckpoint(
    best_acc_path,
    monitor="val_categorical_accuracy",
    save_weights_only=True,
    save_best_only=True,
)
auc_checkpoint = keras.callbacks.ModelCheckpoint(
    best_auc_path, monitor="val_auc", save_weights_only=True, save_best_only=True
)
early_stopping = keras.callbacks.EarlyStopping(patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, auc_checkpoint, reduce_lr],
)
model.save_weights(latest_path)


## === cell 21
from sklearn.metrics import classification_report
y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
cls_report = classification_report(y_val, y_pred)
print(cls_report)


## === cell 22
test = pd.read_csv("/kaggle/working/test.csv")
test.head()


## === cell 23
sample_submission = pd.read_csv("/kaggle/working/sample_submission.csv")
sample_submission.head()


## === cell 24
scores = []

test_ds = tf.data.Dataset.from_tensor_slices(test["comment_text"])


def _fix_input_rank(x):
    return tf.expand_dims(x, axis=-1)  # () -> (1,) so batched input becomes (batch, 1)


test_ds = (
    test_ds.map(_fix_input_rank, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(config.batch_size)
    .cache()
    .prefetch(1)
)

for path in [config.best_acc_model_path, config.lastest_model_path]:
    load_path = path
    if not load_path.endswith(".weights.h5"):
        load_path = load_path.rsplit(".", 1)[0] + ".weights.h5"
    model.load_weights("/kaggle/working/" + load_path)
    score = model.predict(test_ds)
    scores.append(score)

score = np.mean(scores, axis=0)
print(score.shape)


## === cell 25
sample_submission[config.labels] = score
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()
