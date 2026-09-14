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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

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
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.45013

# 6. Current score

0.50485

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.11505) has done: 'Diagnosis: Cell 4 fails because `TransformerBlock.call(self, inputs, training)` declares `training` as a required positional argument. In Keras 3 / TF 2.18, when a layer is invoked inside a Functional model (`x = transformer_block(x)`), Keras not pass `training` at model-construction time, so the layer’s `call` must accept `training=None` (or have a default). This is an API/compatibility issue, not a modeling issue.  
Patch summary: In cell 4, pass an explicit `training` argument when calling `transformer_block` to satisfy the required signature, using `training=False` so graph construction succeeds deterministically. No architecture, layers, or loss/training loop changes are introduced.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: `model` remains a valid compiled Keras Model with identical input/output shapes; cell 5 can compile/fit it unchanged.  
Assumptions: Using `training=False` at build-time is acceptable for model construction; Keras still run training with `training=True` at fit-time for subsequent layers (dropout outside the TransformerBlock and elsewhere) as usual.'
- What this solution (achieved 0.50559) has done: 'Diagnosis: The crash happens in cell 5 because Keras 3 optimizers no longer accept the legacy keyword argument `lr`; they require `learning_rate`. The `ValueError: Argument(s) not recognized: {'lr': 5e-05}` is raised before training starts, so no model/history objects are produced. Replacing `Adam(lr=5e-5)` with `Adam(learning_rate=5e-5)` preserves the same optimizer and learning rate while matching the current API.

Patch summary: Update only the Adam optimizer initialization in cell 5 to use `learning_rate` instead of `lr`. No other training logic, callbacks, or model code is changed.

Updated cells: Only cell 5 is modified below.

Compatibility notes for cell k+1: This fix allows `model.fit(...)` to run and creates `history` as expected, so cell 6 (which reads `history.history[...]`) remains compatible and unchanged.

Assumptions: TensorFlow/Keras versions are as listed (TF 2.18.0 with Keras 3.8.0), so the `lr` alias is not supported and `learning_rate` is required.'
- What this solution (achieved 0.53377) has done: 'Your current score (0.50559, lower-is-better) is worse than the target (0.45013), so we should make a small, low-risk improvement without changing the model architecture or training loop. The biggest issue is that class↔column mapping is currently non-deterministic (`set(...)`) and the submission columns are built from `label_mapper.keys()`; this can silently misalign predicted probabilities with `EAP/HPL/MWS`, inflating log loss. I make the label mapping deterministic and explicitly aligned to the required column order, and I ensure the submission uses that same fixed order. These are minimal changes that preserve the model and training, but should move log loss down toward the target.'
- What this solution (achieved 0.49619) has done: 'Your current logloss (0.53377) is worse than the target (0.45013), so we should make a small, low-risk improvement without changing the model architecture or training loop. The biggest remaining correctness issue is that the TransformerBlock is forced to run with `training=False`, which disables dropout inside the transformer during training and can hurt generalization; changing the layer signature to accept `training=None` restores Keras’ normal behavior at fit-time while keeping the same block/ops. To move logloss down a bit further with minimal risk, we also set `restore_best_weights=True` so the final model corresponds to the best validation loss reached under the same early-stopping regime (no new approximations; just using the best epoch you already evaluated). Submission formatting and class-column alignment are kept deterministic and unchanged.'
- What this solution (achieved 0.50485) has done: 'Your current logloss (0.49619, lower-is-better) is worse than the target (0.45013), so we make one small, low-risk improvement that preserves the exact model architecture and training loop: use a stratified train/validation split so the author class proportions match between train and validation. This typically stabilizes validation loss and helps EarlyStopping pick a better checkpoint, improving generalization a bit without changing layers, optimizer, loss, epochs, or batching. We keep the fixed class order and submission column alignment unchanged, and we add determinism seeds to reduce run-to-run variance (same semantics, just more stable). These changes are minimal and should nudge logloss downward toward the target band.'

# 9. Code solution

## === cell 0
"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _major(_pb_ver) is not None and _major(_pb_ver) >= 6:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf<5"]
    )

    import importlib

    importlib.invalidate_caches()
    for _m in list(sys.modules):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.utils import to_categorical

from tensorflow.keras.preprocessing import text, sequence

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

tf.keras.utils.set_random_seed(42)

print("Tensorflow Version: ", tf.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

notebookstart = time.time()




## === cell 1
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




## === cell 2
MAXLEN = 250
VOCABLEN = 10000
NCHANNEL = 3
BATCHSIZE = 32
EPOCHS = 60

TEXTCOL = "text"
TARGETCOL = "author"
CHARS_TO_REMOVE = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—"



## === cell 3
_base_candidates = [
    "/kaggle/input/spooky-author-identification",
    "/kaggle/data/spooky-author-identification",
    "/kaggle/input",
    "/kaggle/data",
]
_base = next((p for p in _base_candidates if os.path.exists(p)), None)
if _base is None:
    raise FileNotFoundError(
        f"Could not find dataset directory. Tried: {_base_candidates}"
    )

train_csv = os.path.join(_base, "train.csv")
test_csv = os.path.join(_base, "test.csv")
sub_csv = os.path.join(_base, "sample_submission.csv")

if os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(sub_csv):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    submission = pd.read_csv(sub_csv)
else:
    train = pd.read_csv(f"zip://{_base}/train.zip::train.csv")
    test = pd.read_csv(f"zip://{_base}/test.zip::test.csv")
    submission = pd.read_csv(
        f"zip://{_base}/sample_submission.zip::sample_submission.csv"
    )

testdex = test.id
sub_cols = submission.columns

xtrain = train[TEXTCOL].astype(str)
xtest = test[TEXTCOL].astype(str)

class_order = [c for c in ["EAP", "HPL", "MWS"] if c in train[TARGETCOL].unique()]
if len(class_order) != NCHANNEL:
    class_order = sorted(train[TARGETCOL].unique().tolist())

label_mapper = {name: i for i, name in enumerate(class_order)}
num_label = np.vectorize(label_mapper.get)(train[TARGETCOL].values)
y_train = to_categorical(num_label, num_classes=NCHANNEL)

tokenizer = text.Tokenizer(
    filters=CHARS_TO_REMOVE,
    lower=False,
    num_words=VOCABLEN,
)
tokenizer.fit_on_texts(list(xtrain) + list(xtest))

xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = sequence.pad_sequences(xtrain, maxlen=MAXLEN)
xtest = sequence.pad_sequences(xtest, maxlen=MAXLEN)

print("X Shape: {}".format(xtrain.shape))
print("y Shape: {}".format(y_train.shape))

X_train, X_val, y_train, y_val = train_test_split(
    xtrain,
    y_train,
    test_size=0.33,
    random_state=42,
    stratify=np.argmax(y_train, axis=1),
)

print("X_train Shape: {}".format(X_train.shape))
print("y_train Shape: {}".format(y_train.shape))

del train, test, xtrain



## === cell 4
embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MAXLEN,))
embedding_layer = TokenAndPositionEmbedding(MAXLEN, VOCABLEN, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NCHANNEL, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)



## === cell 5
"""
## Train and Evaluate
"""
es = callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=4,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

model.compile(
    Adam(learning_rate=5e-5), "categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[es],
)



## === cell 6
plot_metrics = ["loss"]

f, ax = plt.subplots(1, figsize=[7, 4])
for p_i, metric in enumerate(plot_metrics):
    ax.plot(history.history[metric], label="Train " + metric)
    ax.plot(history.history["val_" + metric], label="Val " + metric)
    ax.set_title("Loss Curve - {}".format(metric))
    ax.legend()
plt.show()



## === cell 7
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


val_pred = model.predict(X_val, verbose=0)
cnf_matrix = confusion_matrix(np.argmax(y_val, axis=1), np.argmax(val_pred, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(4, 4))
plot_confusion_matrix(
    cnf_matrix,
    classes=["EAP", "HPL", "MWS"],
    title="Confusion matrix, without normalization",
)
plt.show()



## === cell 8
test_pred = model.predict(xtest, verbose=0)
test_pred.shape



## === cell 9
proba_df = pd.DataFrame(test_pred, columns=class_order)
proba_df["id"] = testdex

for c in ["EAP", "HPL", "MWS"]:
    if c not in proba_df.columns:
        proba_df[c] = 0.0

submission = proba_df[sub_cols]
submission.to_csv("submission_transfomer_keras.csv", index=False)
print(submission.shape)
print("Submission columns:", submission.columns.tolist())



## === cell 10
import subprocess, sys

subprocess.run(["head", "submission_transfomer_keras.csv"], check=False)



## === cell 11
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))
