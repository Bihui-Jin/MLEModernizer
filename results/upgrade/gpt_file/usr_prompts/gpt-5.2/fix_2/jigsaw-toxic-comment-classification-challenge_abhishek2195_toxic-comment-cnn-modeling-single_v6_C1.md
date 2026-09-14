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

0.9644613541468936

# 6. Current score

0.50911

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50911) has done: 'I fix the environment-breaking import error by using `tf.keras` consistently (avoiding the protobuf/keras incompatibility that triggers `MessageFactory.GetPrototype`). I also fix the missing input files by loading the correct Kaggle competition CSVs (`/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv` and `test.csv`) and create the expected `cleaned` text field from `comment_text` with minimal preprocessing. Then I keep your same CNN architecture, sequence length (150), embedding size (50), and training approach, but ensure all variables are defined in-order and that a valid submission CSV with the required columns is written. Finally, I make the script robust to path differences by falling back to `/kaggle/input/train.csv`/`test.csv` if needed, without changing core modeling logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import random

random.seed(42)
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



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
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

print("TensorFlow:", tf.__version__)



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
base1 = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
train_path = os.path.join(base1, "train.csv")
test_path = os.path.join(base1, "test.csv")
sample_path = os.path.join(base1, "sample_submission.csv")

if not os.path.exists(train_path):
    train_path = "/kaggle/input/train.csv"
    test_path = "/kaggle/input/test.csv"
    sample_path = "/kaggle/input/sample_submission.csv"

print("Using train:", train_path)
print("Using test :", test_path)
print("Using samp :", sample_path)



## === cell 5
df_train = pd.read_csv(train_path)
print("Shape=>", df_train.shape)
df_train.head()



## === cell 6
df_test = pd.read_csv(test_path)
print("Shape=>", df_test.shape)
df_test.head()




## === cell 7
def basic_clean(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s


df_train["cleaned"] = basic_clean(df_train["comment_text"])
df_test["cleaned"] = basic_clean(df_test["comment_text"])

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
assert all(
    c in df_train.columns for c in label_cols
), "Missing expected target columns in train.csv"



## === cell 8
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["cleaned"])



## === cell 9
print("Vocabulary Size=>", len(tokenizer.word_index))



## === cell 10
train_seq = tokenizer.texts_to_sequences(df_train["cleaned"])
test_seq = tokenizer.texts_to_sequences(df_test["cleaned"])



## === cell 11
MAXLEN = 150



## === cell 12
train_seq = pad_sequences(train_seq, maxlen=MAXLEN, padding="post")
test_seq = pad_sequences(test_seq, maxlen=MAXLEN, padding="post")



## === cell 13
vocabulary = len(tokenizer.word_index) + 1
print("Vocabulary Size=>", vocabulary)



## === cell 14
print("Shape of train_sequence=>", train_seq.shape)
print("Shape of test_sequence=>", test_seq.shape)



## === cell 15
y_train = df_train[label_cols].values.astype(np.float32)
print("y_train:", y_train.shape, y_train.dtype)



## === cell 16
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_seq, y_train))
    .shuffle(2048, seed=42, reshuffle_each_iteration=True)
    .batch(TOTAL_BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_seq).batch(TOTAL_BATCH_SIZE).prefetch(AUTO)
)



## === cell 17
print(train_dataset)
print(test_dataset)



## === cell 18
tf.random.set_seed(42)



## === cell 19
with strategy.scope():
    input_1 = Input(shape=(MAXLEN,))
    embedding_1 = Embedding(vocabulary, 50)(input_1)
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

try:
    plot_model(model, to_file="model.png", show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 20
es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=5,
    min_delta=1e-5,
    restore_best_weights=False,
)
mc = ModelCheckpoint(
    "/kaggle/working/model.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)

history = model.fit(
    train_seq,
    y_train,
    batch_size=64,
    epochs=100,
    verbose=1,
    validation_split=0.1,
    callbacks=[es, mc],
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2134116424.py in <cell line: 0>()
      8     restore_best_weights=False,
      9 )
---> 10 mc = ModelCheckpoint(
     11     "/kaggle/working/model.hdf5",
     12     monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/model.hdf5

## === cell 21
train_pred = model.predict(train_seq, batch_size=256, verbose=1)
print("In-sample Evaluation ROC-AUC Score:\n", roc_auc_score(y_train, train_pred))



## === cell 22
final_pred = model.predict(test_seq, batch_size=256, verbose=1)
print("final_pred shape:", final_pred.shape)



## === cell 23
sub = pd.read_csv(sample_path)
sub = sub[["id"] + label_cols].copy()

pred_df = pd.DataFrame(final_pred, columns=label_cols)
pred_df.insert(0, "id", df_test["id"].values)

sub = sub[["id"]].merge(pred_df, on="id", how="left")

sub[label_cols] = sub[label_cols].fillna(0.5).clip(0.0, 1.0)

print(sub.head())
print(sub.shape)



## === cell 24
out_path = "/kaggle/working/submission-CNN-single-2-150.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(sub.columns))
print("Any nulls:", sub.isnull().any().any())
