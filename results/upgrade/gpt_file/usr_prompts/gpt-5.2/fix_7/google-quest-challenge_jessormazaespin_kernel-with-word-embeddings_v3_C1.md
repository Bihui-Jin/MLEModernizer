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

0.2265280693542634

# 6. Current score

0.17442

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18739) has done: 'I fix the environment/runtime issues by removing the broken `tqdm_notebook` import and avoiding the unavailable external model/tokenizer files, which currently prevent any submission from being created. To preserve the core “tokenize text → pad sequences → neural net predicts 30 targets” logic, I train the same style Keras model directly from the provided `train.csv` and then run inference on `test.csv`. I also make padding length deterministic (based on training sequence lengths) so train/test shapes match, and I write the submission using `sample_submission.csv` to guarantee correct column names/order and a valid `.csv` file. This should run end-to-end within the Kaggle environment and yield a reasonable score toward the target.'
- What this solution (achieved 0.19136) has done: 'I fix the protobuf/TensorFlow crash that happens at import time by forcing TensorFlow to use the pure‑Python protobuf implementation (a common Kaggle notebook workaround for the `MessageFactory.GetPrototype` error). Then I keep your existing tokenization, padding, and multi-branch BiGRU model intact, but switch the loss from `binary_crossentropy` to `mean_squared_error`, which better matches the continuous [0,1] targets and typically improves Spearman correlation with minimal semantic change. Finally, I ensure the script still writes a correctly formatted `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved 0.17442) has done: 'The crash happens before any CSVs are read because the TensorFlow/protobuf combo in this environment is still trying to call a removed `MessageFactory.GetPrototype`. I fix this by forcing the pure‑Python protobuf implementation *and* disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and by ensuring these env vars are set before TensorFlow (or anything that might import protobuf) loads. I also add a safe fallback that sets `TF_USE_LEGACY_KERAS=1` (harmless if unused) to improve TF/Keras import stability in Kaggle runtimes. No changes are made to tokenization, padding, model architecture, training loop, or submission formatting—this is runtime-stability only so you can generate a valid `submission.csv` and then iterate on score.'
- What this solution (achieved 0.17442) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the environment variables are set before any protobuf/TensorFlow-related imports and by importing protobuf early to force the pure-Python implementation. Then I keep your tokenization, padding, model architecture, and training loop intact, only adding a deterministic Spearman-friendly post-processing step: rank-based normalization per target column (a monotonic transform) to better match the evaluation metric without changing the model itself. Finally, I keep submission formatting anchored to `sample_submission.csv` column order and write a valid `submission.csv`.'
- What this solution (achieved 0.17442) has done: 'You’re failing before any data is read due to a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) that the current env-var workaround isn’t reliably preventing. I make the protobuf fallback more robust by forcing the pure-Python protobuf implementation **and** uninstalling/avoiding the C++ backend at runtime when possible, and I switch imports to `tf.keras` only after protobuf is guaranteed initialized. Then I keep your existing tokenization, padding, BiGRU architecture, and training loop intact, and keep the rank-normalization post-processing (monotonic, Spearman-friendly) to move the score upward toward the target. Finally, I ensure the submission is written with correct column order and `qa_id` alignment.'
- What this solution (achieved 0.17442) has done: 'The current failure happens before any data is loaded due to a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) that is not reliably avoided by environment variables alone in this runtime. I fix this by ensuring the pure-Python protobuf implementation is forced *and* by pinning protobuf to the Python implementation before TensorFlow is imported (including an early import of `google.protobuf`), plus a safe fallback to import `tensorflow.compat.v1` first if needed. I keep your tokenization, padding, BiGRU architecture, training loop, loss, and rank-normalization post-processing unchanged to preserve semantics while restoring end-to-end execution. The script still write a valid `submission.csv` with the exact `sample_submission.csv` column order and correct `qa_id` alignment.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import google.protobuf  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import protobuf early: {e}")

import pandas as pd

try:
    import tensorflow as tf
except AttributeError:
    import tensorflow.compat.v1 as tf  # type: ignore

    tf.disable_v2_behavior()  # noqa: E402

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import text as ktext
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    tf.random.set_seed(SEED)
except Exception:
    pass

PATH = "../input/google-quest-challenge/"

train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")
sample_path = os.path.join(PATH, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(df_train.shape, df_test.shape, sample_sub.shape)
print("First columns:", df_train.columns[:12].tolist())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
text_cols = ["question_title", "question_body", "answer"]

target_cols = sample_sub.columns.tolist()
target_cols.remove("qa_id")
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

for c in text_cols:
    df_train[c] = df_train[c].fillna("").astype(str)
    df_test[c] = df_test[c].fillna("").astype(str)

y = df_train[target_cols].astype(np.float32).values

tokenizer = ktext.Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(
    (
        df_train["question_title"].tolist()
        + df_train["question_body"].tolist()
        + df_train["answer"].tolist()
    )
)

train_title = tokenizer.texts_to_sequences(df_train["question_title"].tolist())
train_question = tokenizer.texts_to_sequences(df_train["question_body"].tolist())
train_answer = tokenizer.texts_to_sequences(df_train["answer"].tolist())

test_title = tokenizer.texts_to_sequences(df_test["question_title"].tolist())
test_question = tokenizer.texts_to_sequences(df_test["question_body"].tolist())
test_answer = tokenizer.texts_to_sequences(df_test["answer"].tolist())


def percentile_len(seqs, p=95):
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    return int(np.percentile(lens, p))


maxlen_title = max(10, percentile_len(train_title, 95))
maxlen_question = max(50, percentile_len(train_question, 95))
maxlen_answer = max(50, percentile_len(train_answer, 95))

print("Max lengths:", maxlen_title, maxlen_question, maxlen_answer)

train_title = pad_sequences(
    train_title, maxlen=maxlen_title, padding="post", truncating="post"
)
train_question = pad_sequences(
    train_question, maxlen=maxlen_question, padding="post", truncating="post"
)
train_answer = pad_sequences(
    train_answer, maxlen=maxlen_answer, padding="post", truncating="post"
)

test_title = pad_sequences(
    test_title, maxlen=maxlen_title, padding="post", truncating="post"
)
test_question = pad_sequences(
    test_question, maxlen=maxlen_question, padding="post", truncating="post"
)
test_answer = pad_sequences(
    test_answer, maxlen=maxlen_answer, padding="post", truncating="post"
)

vocab_size = min(50000, len(tokenizer.word_index) + 1)
print("Vocab size:", vocab_size)




## === cell 2
def build_model(vocab_size, maxlen_title, maxlen_question, maxlen_answer, n_targets=30):
    emb_dim = 64

    in_title = layers.Input(shape=(maxlen_title,), name="title")
    in_question = layers.Input(shape=(maxlen_question,), name="question")
    in_answer = layers.Input(shape=(maxlen_answer,), name="answer")

    emb = layers.Embedding(
        input_dim=vocab_size, output_dim=emb_dim, mask_zero=True, name="emb"
    )

    def branch(inp, name_prefix):
        x = emb(inp)
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=False), name=f"{name_prefix}_bigru"
        )(x)
        x = layers.Dense(64, activation="relu", name=f"{name_prefix}_dense")(x)
        return x

    b1 = branch(in_title, "title")
    b2 = branch(in_question, "question")
    b3 = branch(in_answer, "answer")

    x = layers.Concatenate(name="concat")([b1, b2, b3])
    x = layers.Dense(128, activation="relu", name="dense_merged")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(n_targets, activation="sigmoid", name="targets")(x)

    model = models.Model(inputs=[in_title, in_question, in_answer], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss="mean_squared_error",
    )
    return model


model = build_model(
    vocab_size, maxlen_title, maxlen_question, maxlen_answer, n_targets=len(target_cols)
)
model.summary()



## === cell 3
n = len(df_train)
idx = np.arange(n)
np.random.shuffle(idx)

val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

X_trn = [train_title[trn_idx], train_question[trn_idx], train_answer[trn_idx]]
y_trn = y[trn_idx]
X_val = [train_title[val_idx], train_question[val_idx], train_answer[val_idx]]
y_val = y[val_idx]

history = model.fit(
    X_trn, y_trn, validation_data=(X_val, y_val), epochs=2, batch_size=128, verbose=2
)



## === cell 4
test_pred = model.predict(
    [test_title, test_question, test_answer], batch_size=256, verbose=1
)
test_pred = np.clip(test_pred, 0.0, 1.0)


def rank_normalize_cols(arr):
    arr = np.asarray(arr)
    n_rows, n_cols = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    denom = max(1, n_rows - 1)
    for j in range(n_cols):
        order = np.argsort(arr[:, j], kind="mergesort")
        ranks = np.empty(n_rows, dtype=np.int32)
        ranks[order] = np.arange(n_rows, dtype=np.int32)
        out[:, j] = ranks.astype(np.float32) / float(denom)
    return out


test_pred = rank_normalize_cols(test_pred)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample_sub.copy()
submission["qa_id"] = df_test["qa_id"].values
for i, c in enumerate(target_cols):
    submission[c] = test_pred[:, i]

submission = submission[["qa_id"] + target_cols]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert submission.shape[0] == df_test.shape[0]
assert submission.shape[1] == 31
assert out_path.endswith(".csv")
