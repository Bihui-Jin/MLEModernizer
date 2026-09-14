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

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.82254

# 6. Current score

0.42193

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.42193) has done: 'I fix the import/runtime breakage caused by the new Kaggle environment’s Keras/Protobuf stack by switching your Keras imports to `tf_keras`, which provides the expected `Tokenizer`, `pad_sequences`, and `Sequential` APIs. I also fix data paths so the CSVs load reliably from `/kaggle/input/...` (no dependency on unzip), and I ensure padding uses a consistent `maxlen=MAX_SEQUENCE_LENGTH` to match your intended setup and avoid shape issues. Finally, I write a proper submission file with a `.csv` suffix (not overwriting the provided `sample_submission.csv`) and with columns in the required order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, Embedding
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

df_train.shape, df_test.shape, sample_submission.shape



## === cell 2
train_texts = df_train["comment_text"].fillna("").values
test_texts = df_test["comment_text"].fillna("").values

train_labels = df_train[LABEL_COLS].astype("float32")

train_texts[:1], train_labels.head(1)



## === cell 3
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
len(word_index)



## === cell 4
trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

trainvalid_data.shape, test_data.shape



## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    trainvalid_data,
    train_labels.values,
    test_size=VALIDATION_SPLIT,
    shuffle=True,
    random_state=SEED,
)

X_train.shape, y_train.shape, X_val.shape, y_val.shape



## === cell 6
cnn_model = Sequential()
cnn_model.add(Embedding(MAX_NUM_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
cnn_model.add(Conv1D(128, 5, activation="relu"))
cnn_model.add(MaxPooling1D(5))
cnn_model.add(Conv1D(128, 5, activation="relu"))
cnn_model.add(MaxPooling1D(5))
cnn_model.add(Conv1D(128, 5, activation="relu"))
cnn_model.add(GlobalMaxPool1D())
cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dense(6, activation="sigmoid"))

cnn_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

cnn_model.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_10/3733993976.py in <cell line: 0>()
     11 
     12 # Keep core compile semantics; metric is for monitoring only
---> 13 cnn_model.compile(
     14     loss="binary_crossentropy",
     15     optimizer="adam",

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/distribute_lib.py in variable_created_in_scope(self, v)
   4019 
   4020   def variable_created_in_scope(self, v):
-> 4021     return v._distribute_strategy is None  # pylint: disable=protected-access
   4022 
   4023   def _experimental_distribute_dataset(self, dataset, options):

AttributeError: 'Variable' object has no attribute '_distribute_strategy'

## === cell 7
history = cnn_model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=1,
    validation_data=(X_val, y_val),
    verbose=2,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_10/2380441243.py in <cell line: 0>()
----> 1 history = cnn_model.fit(
      2     X_train,
      3     y_train,
      4     batch_size=128,
      5     epochs=1,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py in _assert_compile_was_called(self)
   3976         # (i.e. whether the model is built and its inputs/outputs are set).
   3977         if not self._is_compiled:
-> 3978             raise RuntimeError(
   3979                 "You must compile your model before "
   3980                 "training/testing. "

RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.

## === cell 8
y_preds = cnn_model.predict(test_data, batch_size=512, verbose=1)
y_preds.shape, y_preds[:1]



## === cell 9
submission = pd.DataFrame(y_preds, columns=LABEL_COLS)
submission.insert(0, "id", df_test["id"].values)

submission = submission[["id"] + LABEL_COLS]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

out_path, submission.head()
