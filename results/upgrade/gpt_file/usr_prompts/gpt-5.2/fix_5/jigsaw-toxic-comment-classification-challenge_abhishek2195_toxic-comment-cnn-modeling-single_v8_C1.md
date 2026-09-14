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

No external packages required in the script and installed.

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

0.965819443845645

# 6. Current score

0.4792

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53537) has done: 'I fix the import/runtime crash by using `tf.keras` consistently (avoids the protobuf/MessageFactory `GetPrototype` issue caused by mixed `keras`/`tensorflow` stacks) and remove the non-essential `plot_model` call that can fail due to missing graphviz/pydot. I also fix the dataset I/O by loading the actual competition `train.csv` and `test.csv` from `/kaggle/input/jigsaw-toxic-comment-classification-challenge/` (with a safe fallback to the mirrored `/kaggle/data/...` path), and map the expected `cleaned` column to `comment_text` so the rest of the pipeline stays intact. Finally, I keep the same CNN architecture/training semantics, ensure all variables exist in order, and write a valid `submission.csv` with the required header/column order.'
- What this solution (achieved 0.4792) has done: 'The timeout is dominated by (1) slow Python-loop tokenization/padding over ~1.1M texts, (2) building an enormous 200k-word vocabulary and embedding matrix, and (3) heavy disk I/O from per-epoch ModelCheckpoint saves. The changes below keep the same model, loss, training loop, and evaluation semantics, but speed up preprocessing by using TensorFlow’s fast `Tokenizer.texts_to_sequences` + `pad_sequences` (vectorized C/TF ops), right-size the embedding vocabulary to the actually used capped vocabulary, and reduce checkpoint overhead by saving weights only. These optimizations are equivalent in outputs (up to negligible floating point differences) and are directly targeted at the timeout bottlenecks.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        shown += 1
        if shown >= 30:
            break
    if shown >= 30:
        break



## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
tf.random.set_seed(42)

tf.config.run_functions_eagerly(False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 4
BASE1 = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
BASE2 = "/kaggle/data/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(BASE1, "train.csv")
test_path = os.path.join(BASE1, "test.csv")
sample_path = os.path.join(BASE1, "sample_submission.csv")

if not os.path.exists(train_path):
    train_path = os.path.join(BASE2, "train.csv")
    test_path = os.path.join(BASE2, "test.csv")
    sample_path = os.path.join(BASE2, "sample_submission.csv")

print("Train path:", train_path)
print("Test path:", test_path)
print("Sample path:", sample_path)



## === cell 5
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
usecols_train = ["id", "comment_text"] + LABELS
dtype_train = {c: "int8" for c in LABELS}
dtype_train.update({"id": "string", "comment_text": "string"})

df_train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
print("Shape=>", df_train.shape)
df_train.head()



## === cell 6
usecols_test = ["id", "comment_text"]
dtype_test = {"id": "string", "comment_text": "string"}

df_test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)
print("Shape=>", df_test.shape)
df_test.head()



## === cell 7
train_text = df_train["comment_text"].fillna("").astype(str).to_numpy()
test_text = df_test["comment_text"].fillna("").astype(str).to_numpy()



## === cell 8
MAX_FEATURES = 200000
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
tokenizer.fit_on_texts(train_text)



## === cell 9
vocab_size_effective = min(MAX_FEATURES, len(tokenizer.word_index) + 1)
print("Vocabulary Size (capped)=>", vocab_size_effective)



## === cell 10
MAXLEN = 100



## === cell 11
train_seq = pad_sequences(
    tokenizer.texts_to_sequences(train_text),
    maxlen=MAXLEN,
    padding="post",
    truncating="post",
    value=0,
).astype(np.int32)

test_seq = pad_sequences(
    tokenizer.texts_to_sequences(test_text),
    maxlen=MAXLEN,
    padding="post",
    truncating="post",
    value=0,
).astype(np.int32)



## === cell 12
vocabulary = vocab_size_effective
print("Embedding Vocabulary Size=>", vocabulary)



## === cell 13
print("Shape of train_sequence=>", train_seq.shape, train_seq.dtype)
print("Shape of test_sequence=>", test_seq.shape, test_seq.dtype)



## === cell 14
y_train = df_train[LABELS].to_numpy(dtype=np.float32)
print(y_train.shape, y_train.dtype)



## === cell 15
N = train_seq.shape[0]
val_size = int(0.1 * N)
train_size = N - val_size

x_tr, x_val = train_seq[:train_size], train_seq[train_size:]
y_tr, y_val = y_train[:train_size], y_train[train_size:]

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    .shuffle(2048, seed=42, reshuffle_each_iteration=True)
    .batch(64, drop_remainder=False)
    .prefetch(AUTO)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(64, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = tf.data.Dataset.from_tensor_slices(test_seq).batch(256).prefetch(AUTO)



## === cell 16
print(train_dataset)
print(val_dataset)
print(test_dataset)



## === cell 17
with strategy.scope():
    input_1 = Input(shape=(MAXLEN,))
    embedding_1 = Embedding(vocabulary, 100)(input_1)
    conv_1 = Conv1D(filters=64, kernel_size=3, padding="same")(embedding_1)
    dropout_1 = Dropout(0.2)(conv_1)
    pool_1 = GlobalMaxPooling1D()(dropout_1)

    dense = Dense(128, activation="relu")(pool_1)
    output = Dense(6, activation="sigmoid")(dense)

    model = Model(inputs=[input_1], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

model.summary()



## === cell 18
es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=5,
    min_delta=1e-5,
    restore_best_weights=True,
)

mc = ModelCheckpoint(
    "/kaggle/working/model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)

model.fit(
    train_dataset,
    epochs=100,
    verbose=1,
    validation_data=val_dataset,
    callbacks=[es, mc],
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2388563081.py in <cell line: 0>()
     10 # Speedup: avoid heavy full-model serialization each epoch; weights-only checkpoint is much faster I/O.
     11 # Correctness: restore_best_weights=True already ensures best weights are used; checkpoint remains best-only.
---> 12 mc = ModelCheckpoint(
     13     "/kaggle/working/model.keras",
     14     monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=/kaggle/working/model.keras

## === cell 19
train_pred = model.predict(train_seq, batch_size=256, verbose=1)
print("In-sample Evaluation ROC-AUC Score:\n", roc_auc_score(y_train, train_pred))



## === cell 20
final_pred = model.predict(test_dataset, verbose=1)



## === cell 21
prob = pd.DataFrame(final_pred, columns=LABELS)
prob.insert(0, "id", df_test["id"].to_numpy())
prob.head()



## === cell 22
out_path = "/kaggle/working/submission.csv"
prob.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(prob.shape)
