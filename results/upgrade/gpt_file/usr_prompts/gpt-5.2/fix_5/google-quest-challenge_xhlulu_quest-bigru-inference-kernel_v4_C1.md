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

0.1873609852665528

# 6. Current score

0.21483

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12167) has done: 'Your notebook fails because it tries to load a tokenizer and pre-trained model from a Kaggle dataset path that isn’t available, and it also hits a TensorFlow/protobuf import issue. I remove the unavailable external-file dependency by training the same BiGRU-style model inside this notebook using the provided `train.csv`, then use it to predict on `test.csv`. I also replace `tqdm_notebook` (which can misbehave in non-notebook runs) with plain `tqdm`, and add a safe TensorFlow import/config to avoid the protobuf `MessageFactory` error. Finally, the script always write a valid `submission.csv` with the exact sample submission columns and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.11266) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing a compatible pure-Python protobuf implementation *before* importing TensorFlow, which is the root runtime blocker in your environment. Then I keep the same BiGRU multi-input architecture and training loop, but make a minimal metric-aligned change: switch the loss from `binary_crossentropy` to `mse`, which better matches the continuous [0,1] regression targets used for Spearman correlation in this competition and should move the score upward toward your target. I also add a small validation split (no architectural change) so training behavior is stable/observable, while still training on almost all data. Finally, I ensure the script always writes a correct `submission.csv` with the exact sample submission columns and row count.'
- What this solution (achieved 0.21483) has done: 'I fix the TensorFlow/protobuf crash by forcing a protobuf version that TensorFlow expects (and falling back to an alternate safe setting) *before* importing TensorFlow, which is the root runtime blocker. Then I keep your exact tokenizer + 3-input BiGRU architecture and the MSE regression objective, but increase training slightly (more epochs) to move the score upward toward your target while staying within typical Kaggle CPU/GPU time limits. I also ensure the script always writes a correctly-shaped `submission.csv` matching `sample_submission.csv` columns and test row order. All other logic (data paths, features, model structure) remains unchanged.'
- What this solution (achieved 0.21483) has done: 'We fix the TensorFlow/protobuf crash by setting the additional environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early and importing `google.protobuf` before importing TensorFlow (this is the most reliable ordering fix for the `MessageFactory.GetPrototype` error). The rest of the pipeline (tokenizer, 3-input BiGRU model, MSE loss, training loop, and submission formatting) stays the same to avoid unnecessary score shifts since your current score is already above the target band. I also add a small fallback path resolver so the code works whether the data lives in `/kaggle/input/google-quest-challenge` or `/kaggle/input/google-quest-challenge/google-quest-challenge`. Finally, the script always write a valid `submission.csv` with the exact sample submission columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
]
DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = DATA_DIR_CANDIDATES[0]  # keep original path as default

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

print("Using DATA_DIR:", DATA_DIR)
print(
    "Files exist:",
    os.path.exists(TRAIN_PATH),
    os.path.exists(TEST_PATH),
    os.path.exists(SAMPLE_SUB_PATH),
)



## === cell 1
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
except Exception as e:
    print("Warning: could not import google.protobuf early:", e)

import tensorflow as tf

try:
    tf.random.set_seed(SEED)
except Exception as e:
    print("Warning: could not set TF seed:", e)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    GRU,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

print(
    "train:", train.shape, "test:", test.shape, "sample_submission:", submission.shape
)

TARGET_COLS = submission.columns.tolist()[1:]  # 30 targets
TEXT_COLS = ["question_title", "question_body", "answer"]

for c in TEXT_COLS:
    train[c] = train[c].fillna("")
    test[c] = test[c].fillna("")

y = train[TARGET_COLS].astype("float32").values
print("Targets shape:", y.shape)



## === cell 3
all_text = (
    pd.concat(
        [
            train["question_title"],
            train["question_body"],
            train["answer"],
            test["question_title"],
            test["question_body"],
            test["answer"],
        ],
        axis=0,
        ignore_index=True,
    )
    .astype(str)
    .values
)

MAX_FEATURES = 50000  # keep runtime reasonable
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
tokenizer.fit_on_texts(all_text)

print("Tokenizer vocab size (raw):", len(tokenizer.word_index))




## === cell 4
def compute_sequences(cols, tokenizer, maxlens):
    sequences = []
    for texts, maxlen in zip(cols, maxlens):
        seq = tokenizer.texts_to_sequences(texts.values)
        seq = pad_sequences(seq, maxlen=maxlen, padding="pre", truncating="pre")
        sequences.append(seq)
    return sequences


MAXLENS = [30, 300, 300]

train_data = compute_sequences(
    [train.question_title, train.question_body, train.answer], tokenizer, MAXLENS
)
test_data = compute_sequences(
    [test.question_title, test.question_body, test.answer], tokenizer, MAXLENS
)

print([a.shape for a in train_data], [a.shape for a in test_data])




## === cell 5
def build_model(
    max_features, maxlens, embed_dim=128, rnn_units=64, dropout=0.2, lr=2e-3
):
    inputs = []
    pooled = []

    for i, maxlen in enumerate(maxlens):
        inp = Input(shape=(maxlen,), name=f"inp_{i}")
        x = Embedding(
            input_dim=max_features + 1, output_dim=embed_dim, name=f"emb_{i}"
        )(inp)
        x = Bidirectional(GRU(rnn_units, return_sequences=True), name=f"bigru_{i}")(x)
        avg = GlobalAveragePooling1D(name=f"avg_{i}")(x)
        mx = GlobalMaxPooling1D(name=f"max_{i}")(x)
        x = Concatenate(name=f"concat_pool_{i}")([avg, mx])
        pooled.append(x)
        inputs.append(inp)

    x = Concatenate(name="concat_all")(pooled) if len(pooled) > 1 else pooled[0]
    x = Dropout(dropout, name="dropout_0")(x)
    x = Dense(128, activation="relu", name="dense_0")(x)
    x = Dropout(dropout, name="dropout_1")(x)
    out = Dense(len(TARGET_COLS), activation="sigmoid", name="out")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


max_features_effective = min(
    MAX_FEATURES,
    tokenizer.num_words if tokenizer.num_words is not None else MAX_FEATURES,
)
model = build_model(max_features=max_features_effective, maxlens=MAXLENS)
model.summary()



## === cell 6
BATCH_SIZE = 512
EPOCHS = 5

history = model.fit(
    train_data,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    shuffle=True,
    validation_split=0.05,
)



## === cell 7
test_pred = model.predict(test_data, batch_size=512, verbose=1)
test_pred = np.clip(test_pred, 0.0, 1.0)

sub = submission.copy()
sub[TARGET_COLS] = test_pred.astype("float32")

assert (
    sub.shape[0] == test.shape[0]
), f"Submission rows {sub.shape[0]} != test rows {test.shape[0]}"
assert list(sub.columns) == list(
    submission.columns
), "Submission columns do not match sample submission."

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
