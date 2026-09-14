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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.1475958055409032

# 6. Current score

0.18869

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18869) has done: 'Your notebook fails because it depends on external Kaggle dataset files (`tokenizer.pickle` and `model.h5`) that are not present in your environment, and it also hits a TensorFlow/protobuf compatibility issue during import. I remove the unavailable file dependencies by training the same core BiGRU-style Keras model inside the notebook using the provided `train.csv`, and I build a tokenizer from the training text so `compute_sequences()` works. I also make the TensorFlow import more robust by avoiding the protobuf-triggering `load_model` path and by using standard `tqdm` instead of `tqdm_notebook`. Finally, I ensure the submission uses the exact column order from `sample_submission.csv`, clips predictions to `[0,1]`, and writes `submission.csv` successfully.'
- What this solution (achieved 0.18869) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing it (a common Kaggle runtime issue behind `MessageFactory.GetPrototype`). I also make the data path resolution more robust for both `/kaggle/input/...` and the alternative `/kaggle/data/...` layout you listed, without changing the training/prediction logic. Finally, I ensure the submission columns exactly match `sample_submission.csv` order, clip predictions to `[0,1]`, and always write a `submission.csv` in the working directory.'
- What this solution (achieved 0.18869) has done: 'I fix the TensorFlow/protobuf import crash by switching to `tf.keras` via the standalone `keras` package if TensorFlow fails to import, while keeping the exact same model architecture/training loop/prediction flow. I also add deterministic seeding for the backend actually used (TF or Keras) so results are stable. Finally, I keep the submission formatting identical but make the file-path detection include your `/kaggle/data/...` and `/kaggle/input/...` layouts and ensure the output is always a valid `submission.csv` with the sample’s column order.'

# 9. Code solution

## === cell 0
import os
import json
import pickle
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

USING_TF = False
try:
    import tensorflow as tf  # noqa: F401

    USING_TF = True
except Exception as e:
    USING_TF = False
    _tf_import_error = repr(e)

if USING_TF:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import (
        Input,
        Embedding,
        Bidirectional,
        GRU,
        GlobalMaxPooling1D,
        Dense,
        Dropout,
        Concatenate,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(SEED)
else:
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.utils import pad_sequences
    from keras.layers import (
        Input,
        Embedding,
        Bidirectional,
        GRU,
        GlobalMaxPooling1D,
        Dense,
        Dropout,
        Concatenate,
    )
    from keras.models import Model
    from keras.optimizers import Adam

    try:
        keras.utils.set_random_seed(SEED)
    except Exception:
        pass

    print("WARNING: TensorFlow import failed; falling back to standalone Keras.")
    print("TF import error was:", _tf_import_error)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
CANDIDATE_ROOTS = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "/kaggle/data/google-quest-challenge/google-quest-challenge",
]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


TRAIN_PATH = _first_existing(*[os.path.join(r, "train.csv") for r in CANDIDATE_ROOTS])
TEST_PATH = _first_existing(*[os.path.join(r, "test.csv") for r in CANDIDATE_ROOTS])
SAMPLE_PATH = _first_existing(
    *[os.path.join(r, "sample_submission.csv") for r in CANDIDATE_ROOTS]
)

if TRAIN_PATH is None or TEST_PATH is None or SAMPLE_PATH is None:
    raise FileNotFoundError(
        f"Could not locate competition files. Found: TRAIN={TRAIN_PATH}, TEST={TEST_PATH}, SAMPLE={SAMPLE_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_PATH)

print(
    "train:", train.shape, "test:", test.shape, "sample_submission:", submission.shape
)

TARGET_COLS = submission.columns.tolist()[1:]  # exclude qa_id
assert len(TARGET_COLS) == 30, f"Expected 30 targets, got {len(TARGET_COLS)}"



## === cell 3
TEXT_COLS = ["question_title", "question_body", "answer"]

for c in TEXT_COLS:
    train[c] = train[c].fillna("").astype(str)
    test[c] = test[c].fillna("").astype(str)

y = train[TARGET_COLS].values.astype(np.float32)



## === cell 4
MAX_FEATURES = 50000
tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True, oov_token="<OOV>")

all_text = pd.concat(
    [train["question_title"], train["question_body"], train["answer"]],
    axis=0,
    ignore_index=True,
).values

tokenizer.fit_on_texts(all_text)

with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)




## === cell 5
def compute_sequences(cols, tokenizer, maxlens):
    sequences = []
    for texts, maxlen in zip(cols, maxlens):
        seq = tokenizer.texts_to_sequences(texts.values)
        seq = pad_sequences(seq, maxlen=maxlen)
        sequences.append(seq)
    return sequences


MAXLENS = [30, 300, 300]

train_data = compute_sequences(
    [train.question_title, train.question_body, train.answer], tokenizer, MAXLENS
)

test_data = compute_sequences(
    [test.question_title, test.question_body, test.answer], tokenizer, MAXLENS
)

for i, arr in enumerate(train_data):
    print(f"train_data[{i}]:", arr.shape, arr.dtype)



## === cell 6
VOCAB_SIZE = min(MAX_FEATURES, len(tokenizer.word_index) + 1)
EMB_DIM = 128
RNN_UNITS = 64


def build_branch(name, maxlen):
    inp = Input(shape=(maxlen,), name=f"{name}_input")
    x = Embedding(VOCAB_SIZE, EMB_DIM, name=f"{name}_emb")(inp)
    x = Bidirectional(GRU(RNN_UNITS, return_sequences=True), name=f"{name}_bigru")(x)
    x = GlobalMaxPooling1D(name=f"{name}_gmp")(x)
    return inp, x


inp_title, feat_title = build_branch("title", MAXLENS[0])
inp_body, feat_body = build_branch("body", MAXLENS[1])
inp_ans, feat_ans = build_branch("answer", MAXLENS[2])

x = Concatenate(name="concat")([feat_title, feat_body, feat_ans])
x = Dense(256, activation="relu", name="dense1")(x)
x = Dropout(0.2, name="dropout1")(x)
out = Dense(len(TARGET_COLS), activation="sigmoid", name="targets")(x)

model = Model(inputs=[inp_title, inp_body, inp_ans], outputs=out)
model.compile(optimizer=Adam(learning_rate=2e-3), loss="binary_crossentropy")
model.summary()



## === cell 7
n = len(train)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(n * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

x_trn = [arr[trn_idx] for arr in train_data]
y_trn = y[trn_idx]
x_val = [arr[val_idx] for arr in train_data]
y_val = y[val_idx]

EPOCHS = 2
BATCH_SIZE = 256

history = model.fit(
    x_trn,
    y_trn,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 8
test_pred = model.predict(test_data, batch_size=512, verbose=1)
test_pred = np.clip(test_pred, 0.0, 1.0)

sub = submission.copy()
sub = sub[["qa_id"] + TARGET_COLS]

sub[TARGET_COLS] = test_pred

if "qa_id" in test.columns and len(test) == len(sub):
    sub["qa_id"] = test["qa_id"].values

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
