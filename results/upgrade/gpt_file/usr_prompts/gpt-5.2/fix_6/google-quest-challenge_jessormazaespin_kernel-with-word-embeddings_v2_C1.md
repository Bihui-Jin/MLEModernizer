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

0.1138529599004864

# 6. Current score

0.12773

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12817) has done: 'I fix the runtime import issue causing the `MessageFactory.GetPrototype` error by removing unused protobuf-triggering imports/usages and switching `tqdm_notebook` to a safe default. Since the external model/tokenizer files aren’t available at the referenced `/kaggle/input/kernel...` path, I keep the same “tokenize → pad → Keras model predict” core approach but train the same multi-input Keras model inside this notebook using the provided `train.csv`. I also ensure padding uses the same `maxlen` across train/test (derived from train) so shapes match at inference. Finally, I generate a valid `submission.csv` with the exact columns from `sample_submission.csv` and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.12773) has done: 'The protobuf `MessageFactory.GetPrototype` crash is happening during TensorFlow import due to an incompatible `protobuf` version in the Kaggle runtime; the most stable minimal fix is to force the pure-Python protobuf implementation *before* importing TensorFlow. I also add a safe fallback to load data from either `/kaggle/input/google-quest-challenge/` or `/kaggle/data/google-quest-challenge/` so it runs in both directory layouts without changing I/O intent. To move the score down slightly toward your target (your current 0.12817 is above 0.11385), I keep the same model/training loop but make training a bit more regularized by slightly increasing dropout, which typically reduces correlation performance modestly without changing the overall approach. Everything else (tokenization → padding → 3-branch BiGRU → sigmoid outputs → submission columns) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.12773) has done: 'You’re hitting the `MessageFactory.GetPrototype` crash during TensorFlow import, so I harden the protobuf fix by forcing the pure-Python protobuf backend early and also disabling the C++ implementation via env vars that commonly trigger this exact error in Kaggle runtimes. I keep the same tokenization → padding → 3-branch BiGRU → sigmoid multi-target training/inference core logic unchanged, only making the TensorFlow import more robust. I also keep the same data path fallback logic and ensure the script always writes `submission.csv` with the exact `sample_submission.csv` columns and `[0,1]` clipped predictions. No score-tuning changes are introduced beyond restoring the ability to run end-to-end (your current score is already within the ±10% band of the target).'
- What this solution (achieved 0.12773) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime and disabling the C++ implementation *before any TensorFlow-related import*, and by clearing any conflicting protobuf C-extension settings. This is a runtime-stability change only and keeps your model/tokenization/training logic identical. I also add a robust fallback that uses `tf.keras` preprocessing if `tensorflow.keras.preprocessing` isn’t available in this runtime, without changing behavior. Finally, I keep the submission writing exactly the same but ensure the script always reaches it and produces `submission.csv` with the correct columns and `[0,1]` clipped predictions.'
- What this solution (achieved 0.12773) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables correctly and *not* undoing them before importing TensorFlow. This is a runtime-stability fix only and keeps your exact tokenization → padding → 3-branch BiGRU → sigmoid multi-target training/inference approach intact. I also add a safe fallback that uses Keras’ legacy preprocessing if needed (some Kaggle TF builds move these utilities), without changing behavior. No score-tuning changes are introduced since your current score is already within ±10% of the target band; the goal is to get a valid `submission.csv` reliably.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_USE_CPROTOBUF"] = "0"

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CANDIDATE_PATHS = [
    "/kaggle/input/google-quest-challenge/",
    "/kaggle/data/google-quest-challenge/",
    "/kaggle/input/google-quest-challenge/google-quest-challenge/",
    "/kaggle/data/google-quest-challenge/google-quest-challenge/",
]
PATH = None
for p in CANDIDATE_PATHS:
    if os.path.exists(os.path.join(p, "train.csv")):
        PATH = p
        break
if PATH is None:
    raise FileNotFoundError(
        "Could not find train.csv under expected Kaggle paths. "
        f"Tried: {CANDIDATE_PATHS}"
    )

train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")
sample_path = os.path.join(PATH, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

target_cols = [c for c in sample_submission.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 target columns, got {len(target_cols)}"

text_cols = ["question_title", "question_body", "answer"]
for c in text_cols:
    df_train[c] = df_train[c].fillna("").astype(str)
    df_test[c] = df_test[c].fillna("").astype(str)

y = df_train[target_cols].values.astype("float32")

print("Using PATH:", PATH)
print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
print("Targets:", len(target_cols))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MAX_FEATURES = 50000
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
tokenizer.fit_on_texts(
    (
        df_train["question_title"].tolist()
        + df_train["question_body"].tolist()
        + df_train["answer"].tolist()
    )
)


def to_padded(texts, maxlen=None):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")


train_title_seq = tokenizer.texts_to_sequences(df_train["question_title"].tolist())
train_body_seq = tokenizer.texts_to_sequences(df_train["question_body"].tolist())
train_ans_seq = tokenizer.texts_to_sequences(df_train["answer"].tolist())


def percentile_maxlen(seqs, pct=95, min_len=20, cap=300):
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    L = int(np.percentile(lens, pct))
    L = max(L, min_len)
    L = min(L, cap)
    return L


MAXLEN_TITLE = percentile_maxlen(train_title_seq, pct=95, min_len=10, cap=60)
MAXLEN_BODY = percentile_maxlen(train_body_seq, pct=95, min_len=30, cap=300)
MAXLEN_ANS = percentile_maxlen(train_ans_seq, pct=95, min_len=30, cap=300)

X_title = pad_sequences(
    train_title_seq, maxlen=MAXLEN_TITLE, padding="post", truncating="post"
)
X_body = pad_sequences(
    train_body_seq, maxlen=MAXLEN_BODY, padding="post", truncating="post"
)
X_ans = pad_sequences(
    train_ans_seq, maxlen=MAXLEN_ANS, padding="post", truncating="post"
)

X_title = X_title.astype("int32")
X_body = X_body.astype("int32")
X_ans = X_ans.astype("int32")

print("Maxlens:", MAXLEN_TITLE, MAXLEN_BODY, MAXLEN_ANS)
print("Padded shapes:", X_title.shape, X_body.shape, X_ans.shape, "y:", y.shape)



## === cell 2
VOCAB_SIZE = min(MAX_FEATURES, len(tokenizer.word_index) + 1)
EMB_DIM = 64


def make_branch(input_len, name_prefix):
    inp = keras.Input(shape=(input_len,), name=f"{name_prefix}_input")
    x = layers.Embedding(VOCAB_SIZE, EMB_DIM, name=f"{name_prefix}_emb")(inp)
    x = layers.SpatialDropout1D(0.30, name=f"{name_prefix}_sdrop")(x)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False), name=f"{name_prefix}_bigru"
    )(x)
    x = layers.Dense(64, activation="relu", name=f"{name_prefix}_dense")(x)
    return inp, x


inp_title, br_title = make_branch(MAXLEN_TITLE, "title")
inp_body, br_body = make_branch(MAXLEN_BODY, "body")
inp_ans, br_ans = make_branch(MAXLEN_ANS, "ans")

x = layers.Concatenate(name="concat")([br_title, br_body, br_ans])
x = layers.Dropout(0.40, name="drop")(x)
x = layers.Dense(128, activation="relu", name="fc1")(x)
out = layers.Dense(len(target_cols), activation="sigmoid", name="out")(x)

model = keras.Model(inputs=[inp_title, inp_body, inp_ans], outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-3), loss="binary_crossentropy"
)
model.summary()



## === cell 3
N = len(df_train)
idx = np.arange(N)
np.random.shuffle(idx)
split = int(N * 0.9)
tr_idx, va_idx = idx[:split], idx[split:]

Xtr = [X_title[tr_idx], X_body[tr_idx], X_ans[tr_idx]]
ytr = y[tr_idx]
Xva = [X_title[va_idx], X_body[va_idx], X_ans[va_idx]]
yva = y[va_idx]

BATCH_SIZE = 256
EPOCHS = 2  # keep training loop/approach unchanged

history = model.fit(
    Xtr,
    ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 4
test_title = to_padded(df_test["question_title"].tolist(), maxlen=MAXLEN_TITLE).astype(
    "int32"
)
test_body = to_padded(df_test["question_body"].tolist(), maxlen=MAXLEN_BODY).astype(
    "int32"
)
test_ans = to_padded(df_test["answer"].tolist(), maxlen=MAXLEN_ANS).astype("int32")

pred = model.predict([test_title, test_body, test_ans], batch_size=512, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

print("Pred shape:", pred.shape)



## === cell 5
sub = pd.DataFrame(pred, columns=target_cols)
sub.insert(0, "qa_id", df_test["qa_id"].values)

assert sub.shape[0] == df_test.shape[0]
assert list(sub.columns) == list(sample_submission.columns)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
