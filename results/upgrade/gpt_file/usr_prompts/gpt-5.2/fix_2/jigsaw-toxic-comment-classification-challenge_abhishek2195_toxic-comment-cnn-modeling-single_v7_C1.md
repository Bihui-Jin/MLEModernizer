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

0.9615426202729558

# 6. Current score

0.48777

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.48777) has done: 'I fix the TensorFlow/Keras import crash by using `tf.keras` consistently (the current protobuf-related `MessageFactory.GetPrototype` error typically comes from mixing standalone `keras` with the bundled TF stack). I also fix the broken data loading paths by reading the provided `train.csv`/`test.csv` (and create the expected `cleaned` column from `comment_text` with minimal preprocessing so the existing tokenizer logic still works). Then I ensure all downstream variables exist, the model trains, and a valid `submission.csv` with the required column order is written to `/kaggle/working`. These changes keep the same CNN architecture, loss, and training approach, but make the notebook run end-to-end.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

max_print = 50
count = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if count < max_print:
            print(os.path.join(dirname, filename))
            count += 1



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

print("TF version:", tf.__version__)



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
DATA_ROOT = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(train_path):
    DATA_ROOT = "/kaggle/input"
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using paths:")
print("train:", train_path)
print("test :", test_path)
print("sub  :", sub_path)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)
print("Sample submission shape:", df_sub.shape)

df_train.head()




## === cell 5
def minimal_clean(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


df_train["cleaned"] = minimal_clean(df_train["comment_text"])
df_test["cleaned"] = minimal_clean(df_test["comment_text"])

df_train[["id", "cleaned"]].head()



## === cell 6
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["cleaned"])

print("Vocabulary Size=>", len(tokenizer.word_index))



## === cell 7
train_seq = tokenizer.texts_to_sequences(df_train["cleaned"])
test_seq = tokenizer.texts_to_sequences(df_test["cleaned"])

train_seq = pad_sequences(train_seq, maxlen=125, padding="post")
test_seq = pad_sequences(test_seq, maxlen=125, padding="post")

vocabulary = len(tokenizer.word_index) + 1
print("Vocabulary Size=>", vocabulary)
print("Shape of train_sequence=>", train_seq.shape)
print("Shape of test_sequence=>", test_seq.shape)



## === cell 8
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = df_train[label_cols].values.astype(np.float32)
print("y_train shape:", y_train.shape)



## === cell 9
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_seq, y_train))
    .repeat()
    .shuffle(
        buffer_size=min(len(train_seq), 10000), seed=42, reshuffle_each_iteration=True
    )
    .batch(TOTAL_BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_seq).batch(TOTAL_BATCH_SIZE).prefetch(AUTO)
)

print(train_dataset)
print(test_dataset)



## === cell 10
with strategy.scope():
    input_1 = Input(shape=(125,))
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
    print("Saved model plot to model.png")
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 11
es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=5, min_delta=1e-5
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1791287415.py in <cell line: 0>()
      3     monitor="val_loss", mode="min", verbose=1, patience=5, min_delta=1e-5
      4 )
----> 5 mc = ModelCheckpoint(
      6     "/kaggle/working/model.hdf5",
      7     monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/kaggle/working/model.hdf5

## === cell 12
train_pred = model.predict(train_seq, batch_size=256, verbose=1)
print("In-sample Evaluation ROC-AUC Score:\n", roc_auc_score(y_train, train_pred))



## === cell 13
final_pred = model.predict(test_seq, batch_size=256, verbose=1)
print("final_pred shape:", final_pred.shape)



## === cell 14
prob = pd.DataFrame(columns=["id"] + label_cols, index=df_test.index)
prob["id"] = df_test["id"].values
for index, col in enumerate(label_cols):
    prob[col] = final_pred[:, index]

prob = prob[["id"] + label_cols].fillna(0.5)

prob.head()



## === cell 15
out_path = "/kaggle/working/submission-CNN-single-2-125.csv"
prob.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print("Submission shape:", prob.shape)
print("Columns:", list(prob.columns))
