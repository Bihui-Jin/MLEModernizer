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

0.9556

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during imports in cell 1 due to an incompatibility between the installed `protobuf==6.33.0` and some TensorFlow-related libraries (directly or indirectly imported via `tensorflow`), where older code paths still expect `google.protobuf.message_factory.MessageFactory.GetPrototype`, which was removed in newer protobuf. This is a common runtime break when protobuf is upgraded beyond what the TensorFlow stack expects. The smallest safe fix is to force protobuf to use the pure-Python implementation at import time, which restores the expected behavior without changing your model/training logic.  

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) before importing TensorFlow, keeping all other imports and variables unchanged to preserve downstream behavior.  

Updated cells:  

Compatibility notes for cell k+1: Cell 2 only unzips files and does not depend on any changed symbols; all imports from cell 1 remain available with the same names.  

Assumptions: The environment permits setting `os.environ` prior to importing TensorFlow in the notebook kernel, and using the Python protobuf backend is acceptable for this workflow (it is deterministic and fixes the missing attribute).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1 because the environment has `protobuf==6.33.0`, which is incompatible with the TensorFlow 2.18 protobuf API expectations, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not fix this specific incompatibility. The minimal deterministic fix is to downgrade protobuf to a TensorFlow-compatible version before importing TensorFlow.

Patch summary: In cell 1, install a compatible protobuf version (4.25.x) at runtime, then restart the protobuf import path by re-importing TensorFlow after the install. Keep all existing imports and logic intact, only inserting the minimal pip install + import order fix.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: No variables/interfaces consumed by cell 2 change; the fix only ensures TensorFlow can import successfully so the notebook proceeds.

Assumptions: Runtime allows `pip` installs (typical Kaggle-like environment) and installing `protobuf==4.25.3` is sufficient for TensorFlow 2.18 to import cleanly.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 19 crashes because `model` is referenced before it is created. The function `get_model(config)` is defined in cell 17, but no call is made to instantiate and assign it to the variable `model` prior to plotting. Therefore, `keras.utils.plot_model(model, ...)` raises a `NameError`.

Patch summary: In cell 19, create the Keras model by calling `get_model(config)` and assigning it to `model` before calling `plot_model`. This is the minimal change to ensure `model` exists for both plotting in cell 19 and compilation in cell 20.

Updated cells: Only cell 19 is changed.

Compatibility notes for cell k+1: Cell 20 expects `model` to exist; after this patch, `model` be defined and compatible with `model.compile(...)` as written.

Assumptions: `get_model` and `config` are successfully defined in earlier cells (as shown), and the environment supports `keras.utils.plot_model` (if graphviz/pydot are missing it may error later, but this patch fixes the current `NameError`).'
- What this solution (achieved 0.5) has done: 'Diagnosis: The crash happens when constructing `keras.callbacks.ModelCheckpoint` with `save_weights_only=True` while the filepath ends with `.tf`. In Keras 3 / TF 2.18, weight-only checkpoints must use the Keras weights H5 suffix `.weights.h5`, otherwise `ModelCheckpoint` raises a `ValueError` before training starts. Additionally, the callback is monitoring `val_accuracy`, but the model was compiled with `categorical_accuracy`, so the correct metric key emitted during validation is `val_categorical_accuracy`.

Patch summary: In cell 21 only, change the checkpoint filepath to end with `.weights.h5` (keeping the same base name) and update `monitor` to `val_categorical_accuracy` to match the compiled metric name. Leave the training call and weight saving logic unchanged, only adjusting these parameters to satisfy Keras’ API requirements and ensure the monitored metric exists.

Updated cells:'
- What this solution (achieved 0.5) has done: 'Diagnosis: The model input is defined as `keras.Input(shape=(None,), dtype="string")`, i.e., each example should be a 1D string tensor (rank 1) so batched it becomes `(batch, None)`. But the `tf.data.Dataset.from_tensor_slices` in `make_dataset()` is slicing a Pandas Series and producing scalar string tensors per example; after batching this yields `(batch,)`, which Keras rejects as incompatible with `(batch, None)`.  
Patch summary: In cell 21 only, wrap each string in an extra dimension inside the dataset pipeline right before fitting so each example becomes shape `(1,)` and batched becomes `(batch, 1)` / compatible with `(batch, None)`. This preserves the model and training logic while fixing the input-rank mismatch.  
Updated cells: Only cell 21 is changed.  
Compatibility notes for cell k+1: `model` remains the same object; `model.predict(valid_ds)` still works because `valid_ds` keeps the same element structure `(x, y)` and only adjusts `x` rank.  
Assumptions: The dataset yields `x` as a scalar `tf.string` per example and `y` already has shape `(6,)` per example (from the DataFrame), so only `x` needs reshaping.'
- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 21 crashes because `model.save_weights()` in Keras 3 enforces that weight files must end with `.weights.h5` (or another supported weights format), but `config.lastest_model_path` is `"model_latest.tf"`. This path comes from `Config` and is used directly, triggering a `ValueError`. The training itself succeeds; only the final save step fails due to the filename suffix requirement.

Patch summary: In cell 21 only, keep the same behavior (saving the latest weights) but adjust the save path at call time to ensure it ends with `.weights.h5`. This avoids modifying earlier cells (including `Config`) and preserves the rest of the training/evaluation logic unchanged.

Updated cells: See updated cell 21 below.

Compatibility notes for cell k+1: Cell 22 uses the in-memory `model` object for prediction and does not depend on the saved weights filename, so this change is fully compatible.

Assumptions: It is acceptable to save the “latest” weights to a filename derived from `config.lastest_model_path` but with the required `.weights.h5` suffix, without changing any later code that might reference the config value.'
- What this solution (achieved 0.5) has done: 'Diagnosis: The model was built with an input shape of `(None, None)` (because `keras.Input(shape=(None,), dtype="string")` expects rank-2: `(batch, timesteps)`), but `test_ds` yields a rank-1 tensor of strings `(batch,)`. During training you fixed this by `tf.expand_dims(x, axis=-1)`, but that expansion is missing for the test dataset, so `model.predict(test_ds)` fails with an incompatible input rank.  
Patch summary: Update cell 25 to mirror the training/validation preprocessing by expanding the last dimension of each batch in `test_ds` to produce `(batch, 1)` strings, matching the model’s expected input shape. Keep the rest of the submission logic unchanged.  
Updated cells: Only cell 25 is modified.  
Compatibility notes for cell k+1: No interface changes; `score` remains a NumPy array of shape `(len(test), 6)` and `sample_submission[config.labels] = score` continues to work.  
Assumptions: The intended model input for inference is identical to training (string tensors with an added last dimension), and no additional preprocessing beyond this rank fix is required.'

# 9. Code solution

## === cell 0
class Config:
    vocab_size = 15000 # Vocabulary Size
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

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

import pandas as pd
import tensorflow as tf
import pathlib
import random
import string
import re
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
    max_tokens=config.vocab_size, 
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
"""
class_weight =  1 / train["label"].value_counts(normalize=True)
class_weight = dict(class_weight / class_weight.sum())
class_weight
"""


## === cell 13
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


## === cell 14
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


## === cell 15
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


## === cell 16
def get_tfidf_model(config, inputs):
    x = tfidf_vectorizer(inputs)
    x = layers.Dense(256, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer="l2")(x)
    return x


## === cell 17
def get_model(config):
    inputs = keras.Input(shape=(None, ), dtype="string", name="inputs")
    word2vec_x = get_word2vec_model(config, inputs)
    tfidf_x = get_tfidf_model(config, inputs)
    x = layers.Concatenate()([word2vec_x, tfidf_x])
    output = layers.Dense(6, activation="sigmoid")(x)
    model = keras.Model(inputs, output, name="model")
    return model


## === cell 18
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


## === cell 19
model = get_model(config)

keras.utils.plot_model(model, show_shapes=True)


## === cell 20
model.compile(
    "adam", loss="binary_crossentropy", metrics=["categorical_accuracy"]
)


## === cell 21
train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
valid_ds = valid_ds.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))

acc_checkpoint = keras.callbacks.ModelCheckpoint(
    "model_best_acc.weights.h5",
    monitor="val_categorical_accuracy",
    save_weights_only=True,
    save_best_only=True,
)
early_stopping = keras.callbacks.EarlyStopping(patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6)
model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=valid_ds,
    callbacks=[acc_checkpoint, reduce_lr],
)

latest_weights_path = config.lastest_model_path
if not str(latest_weights_path).endswith(".weights.h5"):
    latest_weights_path = str(latest_weights_path).rsplit(".", 1)[0] + ".weights.h5"
model.save_weights(latest_weights_path)


## === cell 22
from sklearn.metrics import classification_report
y_pred = np.array(model.predict(valid_ds) > 0.5, dtype=int)
cls_report = classification_report(y_val, y_pred)
print(cls_report)


## === cell 23
test = pd.read_csv("/kaggle/working/test.csv")
test.head()


## === cell 24
sample_submission = pd.read_csv("/kaggle/working/sample_submission.csv")
sample_submission.head()


## === cell 25
test_ds = (
    tf.data.Dataset.from_tensor_slices(test["comment_text"])
    .batch(config.batch_size)
    .map(lambda x: tf.expand_dims(x, axis=-1))
    .cache()
    .prefetch(1)
)

score = model.predict(test_ds)
sample_submission[config.labels] = score
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()
