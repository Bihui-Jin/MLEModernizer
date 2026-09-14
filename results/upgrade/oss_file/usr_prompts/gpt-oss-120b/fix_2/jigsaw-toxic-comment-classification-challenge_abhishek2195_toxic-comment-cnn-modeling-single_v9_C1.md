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

0.963559276192663

# 6. Current score

0.50588

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50588) has done: 'I replace the broken imports, load the correct CSV files, use the existing “comment_text” column (renamed to “cleaned”), fix the tokenizer and sequence steps, simplify the data pipeline, and ensure all Keras objects are imported from tf.keras. The script now builds, trains, evaluates the CNN model and writes a proper submission CSV with the required columns.'

# 9. Code solution

## === cell 0
import os, warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    Dropout,
    GlobalMaxPooling1D,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
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
df_train["cleaned"] = df_train["comment_text"].astype(str).str.lower()
df_test["cleaned"] = df_test["comment_text"].astype(str).str.lower()



## === cell 6
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["cleaned"])
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size =>", vocab_size)



## === cell 7
MAX_LEN = 100
train_seq = tokenizer.texts_to_sequences(df_train["cleaned"])
test_seq = tokenizer.texts_to_sequences(df_test["cleaned"])

train_seq = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post")
test_seq = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post")
print("train_seq shape:", train_seq.shape)
print("test_seq  shape:", test_seq.shape)



## === cell 8
y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values
print("y_train shape:", y_train.shape)



## === cell 9
with strategy.scope():
    input_ = Input(shape=(MAX_LEN,))
    embed = Embedding(input_dim=vocab_size, output_dim=100, input_length=MAX_LEN)(
        input_
    )
    conv = Conv1D(filters=100, kernel_size=3, padding="same", activation="relu")(embed)
    drop = Dropout(0.2)(conv)
    pool = GlobalMaxPooling1D()(drop)
    dense = Dense(128, activation="relu")(pool)
    output = Dense(6, activation="sigmoid")(dense)

    model = Model(inputs=input_, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

model.summary()
plot_model(model, to_file="model.png", show_shapes=True)



## === cell 10
es = EarlyStopping(
    monitor="val_loss", mode="min", patience=3, verbose=1, min_delta=1e-5
)
mc = ModelCheckpoint(
    "/kaggle/working/best_model.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)

history = model.fit(
    train_seq,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=6,
    validation_split=0.1,
    callbacks=[es, mc],
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2759822229.py in <cell line: 0>()
      2     monitor="val_loss", mode="min", patience=3, verbose=1, min_delta=1e-5
      3 )
----> 4 mc = ModelCheckpoint(
      5     "/kaggle/working/best_model.hdf5",
      6     monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/best_model.hdf5

## === cell 11
model.load_weights("/kaggle/working/best_model.hdf5")
train_pred = model.predict(train_seq, batch_size=BATCH_SIZE)
roc = roc_auc_score(y_train, train_pred, average="macro")
print("In‑sample ROC‑AUC (macro):", roc)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1479801348.py in <cell line: 0>()
      1 # Load best weights and evaluate on the training split
----> 2 model.load_weights("/kaggle/working/best_model.hdf5")
      3 train_pred = model.predict(train_seq, batch_size=BATCH_SIZE)
      4 roc = roc_auc_score(y_train, train_pred, average="macro")
      5 print("In‑sample ROC‑AUC (macro):", roc)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/working/best_model.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 12
final_pred = model.predict(test_seq, batch_size=BATCH_SIZE)



## === cell 13
submission = pd.DataFrame(
    {
        "id": df_test["id"],
        "toxic": final_pred[:, 0],
        "severe_toxic": final_pred[:, 1],
        "obscene": final_pred[:, 2],
        "threat": final_pred[:, 3],
        "insult": final_pred[:, 4],
        "identity_hate": final_pred[:, 5],
    }
)
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
