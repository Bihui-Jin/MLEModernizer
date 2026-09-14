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

0.9659288367986162

# 6. Current score

0.44509

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.44509) has done: 'I replace the failing imports, load the provided CSV files instead of missing JSON files, use the existing “comment_text” column (there is no “cleaned” column), and ensure all Keras symbols are imported correctly. The model architecture stays unchanged; only the data‑handling and small utility fixes are applied so the notebook runs end‑to‑end and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import os, warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
from sklearn.metrics import roc_auc_score



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 4
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)



## === cell 5
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["comment_text"].astype(str))

print("Vocabulary Size =>", len(tokenizer.word_index))



## === cell 6
MAX_LEN = 100
train_seq = tokenizer.texts_to_sequences(df_train["comment_text"].astype(str))
test_seq = tokenizer.texts_to_sequences(df_test["comment_text"].astype(str))

train_seq = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post")
test_seq = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post")

print("train_seq shape:", train_seq.shape)
print("test_seq  shape:", test_seq.shape)



## === cell 7
y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values
print("y_train shape:", y_train.shape)



## === cell 8
vocabulary = len(tokenizer.word_index) + 1


def generate_model(filters, dropout):
    inp = Input(shape=(MAX_LEN,))
    emb = Embedding(vocabulary, 100)(inp)
    conv = Conv1D(filters=int(round(filters)), kernel_size=7, padding="same")(emb)
    drop = Dropout(dropout)(conv)
    pool = GlobalMaxPooling1D()(drop)
    dense = Dense(128, activation="relu")(pool)
    out = Dense(6, activation="sigmoid")(dense)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model




## === cell 9
with strategy.scope():
    model = generate_model(filters=352, dropout=0.06675)

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=5, verbose=1, min_delta=1e-5
)
ckpt_path = "/kaggle/working/model.hdf5"
mc = ModelCheckpoint(
    ckpt_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

model.fit(
    train_seq,
    y_train,
    batch_size=512,
    epochs=100,
    validation_split=0.2,
    callbacks=[es, mc],
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3091659995.py in <cell line: 0>()
      6 )
      7 ckpt_path = "/kaggle/working/model.hdf5"
----> 8 mc = ModelCheckpoint(
      9     ckpt_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
     10 )

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/model.hdf5

## === cell 10
train_pred = model.predict(train_seq, batch_size=TOTAL_BATCH_SIZE)
train_auc = roc_auc_score(y_train, train_pred, average="macro")
print("In‑sample ROC‑AUC (macro):", train_auc)



## === cell 11
test_pred = model.predict(test_seq, batch_size=TOTAL_BATCH_SIZE)



## === cell 12
submission = pd.DataFrame(
    {
        "id": df_test["id"],
        "toxic": test_pred[:, 0],
        "severe_toxic": test_pred[:, 1],
        "obscene": test_pred[:, 2],
        "threat": test_pred[:, 3],
        "insult": test_pred[:, 4],
        "identity_hate": test_pred[:, 5],
    }
)
submission.head()



## === cell 13
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
