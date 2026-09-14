# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3475238241448931

# 6. Current score

0.20006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09599) has done: 'I fix the runtime crash coming from the `transformers`/protobuf incompatibility by removing the dependency on HuggingFace BERT downloads (which are unavailable offline here) and switching to a lightweight, fully-local text vectorization + dense neural network that still produces 30 sigmoid outputs in `[0,1]`. I also ensure the code reads the competition CSVs from the provided Kaggle paths, builds features from the same core text fields (title/body/answer), and writes a valid `submission.csv` with the exact sample submission columns. Finally, I add basic determinism and shape checks so the pipeline runs end-to-end reliably within the time limit.'
- What this solution (achieved 0.22583) has done: 'I fix the protobuf-related crash by ensuring TensorFlow is imported in a way that avoids the problematic C++ protobuf implementation in this environment (forcing the Python protobuf backend before importing TF). Then I keep the same lightweight TextVectorization + Embedding + pooling + dense-sigmoid architecture, but adjust only training hyperparameters that directly impact Spearman correlation (use MSE for continuous targets and a slightly higher Adam LR) without changing the core model. I also correct unused/contradictory hyperparameters (batch_size/learning_rate) so the script does what it declares, and keep deterministic seeding and strict submission shape/column alignment. The result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.22228) has done: 'The crash is happening before training because TensorFlow’s import triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this environment. I fix it by forcing the pure-Python protobuf backend **and** ensuring the runtime uses a protobuf version that TensorFlow expects (via a safe, offline pip install from prebuilt wheels if needed), then import TensorFlow. After that, I keep your exact TextVectorization → Embedding → pooling → Dense → sigmoid model and training loop unchanged, so behavior and score are only affected by the fact that the code can now run end-to-end. Finally, I keep the submission writing logic but add a small safety check to always align `qa_id` ordering to the sample submission.'
- What this solution (achieved 0.22307) has done: 'I fix the TensorFlow/protobuf import crash by forcing a protobuf 3.x/4.x version that matches TF expectations, doing so *before* importing TensorFlow and restarting the interpreter so the new protobuf is actually used. I keep your exact TextVectorization → Embedding → pooling → Dense → sigmoid model and the same train/val loop, only adding a tiny, score-helping but still “core-logic-preserving” tweak: concatenate `category` and `host` to the text input (they’re already in the provided CSVs and often help correlation with minimal disruption). Finally, I keep the submission formatting checks and ensure a `submission.csv` is always written with the correct columns and row order.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by avoiding the `pip install protobuf<5` path (which is triggering the `MessageFactory.GetPrototype` mismatch) and instead forcing the pure-Python protobuf backend while importing TensorFlow cleanly. Then I keep your exact TextVectorization→Embedding→pooling→Dense→sigmoid architecture and the same train/val training loop, only adding a minimal, metric-aligned post-processing step: per-target rank-based (CDF) calibration using training-label ranks, which typically improves mean Spearman correlation without changing the model itself. Finally, I keep the same submission-writing logic and ensure the output is a valid `submission.csv` with correct columns/order.'
- What this solution (achieved 0.20968) has done: 'You’re currently not getting a Kaggle score, so the first priority is to ensure a stable end-to-end run that always writes a valid `submission.csv`. I keep your exact model architecture and training loop, but make two minimal, score-relevant fixes: (1) remove the forced protobuf pip-install/restart behavior (which can intermittently break or loop in offline Kaggle), relying instead on the safe “python protobuf backend” env vars you already set; and (2) fix the rank-calibration to use a proper empirical CDF mapping (avoids edge-case `interp` artifacts and reduces chances of NaNs), while keeping predictions in `[0,1]`. These changes preserve evaluation semantics (still sigmoid `[0,1]` outputs, still Spearman-friendly monotonic calibration) and are designed to reliably yield a valid submission and improve score versus a failed run.'
- What this solution (achieved 0.21135) has done: 'We need to fix the immediate runtime crash on importing TensorFlow caused by a protobuf API mismatch (`MessageFactory.GetPrototype`). The most reliable minimal fix in Kaggle is to pin protobuf to a TF-compatible version *before* importing TensorFlow, then proceed with your exact same model/training/prediction/calibration logic. I also add a small guard to avoid re-install loops and keep everything deterministic, while leaving the core architecture, loss (MSE), and rank-based calibration unchanged to preserve evaluation semantics and nudge score upward only via enabling the code to actually run. The script still write a valid `submission.csv` with the exact sample submission columns.'
- What this solution (achieved 0.11829) has done: 'Your current score (0.21135) is well below the target (0.3475), so we should make a small, low-risk improvement that tends to increase mean Spearman without changing the model architecture or training loop. The biggest score-relevant issue here is that the TextVectorization vocabulary/sequence length are likely truncating too aggressively for this task; increasing `SEQ_LEN` and slightly increasing embedding width typically improves rank correlation while keeping the exact same model structure. I also make the rank-based calibration more stable by using the *training predictions’* rank mapping (monotonic, Spearman-aligned) instead of mapping raw test ranks directly to label quantiles, which can overfit distributional assumptions and suppress correlation. All other core logic (TextVectorization→Embedding→pool→Dense→sigmoid, MSE loss, same fit loop, same output formatting) stays intact, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.20006) has done: 'Your current score (0.118) is far below the target (0.348), so we should make a small, low-risk change that improves mean Spearman without changing your model or training loop. The largest issue is the current rank-calibration uses `np.interp` on potentially heavily-tied `x` values, which can make the mapping unstable/degenerate and hurt correlation. I replace it with a strictly monotonic, tie-robust empirical CDF (“rank”) mapping: map each test prediction to its percentile within the train predictions, then map that percentile to the corresponding quantile of the train labels. This keeps predictions in `[0,1]`, preserves your architecture/loss/fit procedure, and usually improves Spearman because it’s explicitly rank-aligned.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    need_pin = (pb_ver is None) or (_major(pb_ver) is not None and _major(pb_ver) >= 5)

    if need_pin and os.environ.get("PROTOBUF_PINNED_ONCE", "0") != "1":
        os.environ["PROTOBUF_PINNED_ONCE"] = "1"
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )

        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]


_ensure_compatible_protobuf()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Python seed set to:", SEED)



## === cell 1
max_sequence_length = 380  # retained for compatibility; not used
n_epoch = 5
learning_rate = 2e-3
n_fold = 5  # retained for compatibility; not used
batch_size = 256
dropout_rate = 0.1

DATA_PATH_CANDIDATES = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "/kaggle/data/google-quest-challenge/google-quest-challenge",
    "../input/google-quest-challenge",
    "../input/google-quest-challenge/google-quest-challenge",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_PATH = next(
    (p for p in DATA_PATH_CANDIDATES if os.path.exists(os.path.join(p, "train.csv"))),
    None,
)
if DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {DATA_PATH_CANDIDATES}"
    )

print("Using DATA_PATH:", DATA_PATH)



## === cell 2
df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
df_sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

input_columns = ["question_title", "question_body", "answer", "category", "host"]
output_labels = list(df_sample.columns[1:])

print("n_train:", len(df_train), "n_test:", len(df_test))
print("n_targets:", len(output_labels))
assert len(output_labels) == 30, f"Expected 30 targets, got {len(output_labels)}"

for c in input_columns:
    if c not in df_train.columns:
        df_train[c] = ""
    if c not in df_test.columns:
        df_test[c] = ""
    df_train[c] = df_train[c].fillna("").astype(str)
    df_test[c] = df_test[c].fillna("").astype(str)

y = df_train[output_labels].astype(np.float32).values

train_text = (
    df_train["question_title"]
    + " "
    + df_train["question_body"]
    + " "
    + df_train["answer"]
    + " [CAT] "
    + df_train["category"]
    + " [HOST] "
    + df_train["host"]
).values

test_text = (
    df_test["question_title"]
    + " "
    + df_test["question_body"]
    + " "
    + df_test["answer"]
    + " [CAT] "
    + df_test["category"]
    + " [HOST] "
    + df_test["host"]
).values

print("Example merged text length:", len(train_text[0]))



## === cell 3
VOCAB_SIZE = 50000
SEQ_LEN = 384  # was 256
EMB_DIM = 192  # was 128

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)

vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))
print("Vectorizer adapted. Vocab size:", len(vectorizer.get_vocabulary()))




## === cell 4
def create_model():
    inp = tf.keras.layers.Input(shape=(), dtype=tf.string)
    x = vectorizer(inp)
    x = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE, output_dim=EMB_DIM, mask_zero=True
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1500, activation="relu")(x)
    x = tf.keras.layers.Dense(1500, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mse",
)
model.summary()



## === cell 5
n = len(train_text)
idx = np.arange(n)
np.random.shuffle(idx)

val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

x_trn, y_trn = train_text[trn_idx], y[trn_idx]
x_val, y_val = train_text[val_idx], y[val_idx]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
    .shuffle(20000, seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=n_epoch,
    verbose=1,
)



## === cell 6
pred_test = model.predict(
    tf.data.Dataset.from_tensor_slices(test_text).batch(batch_size),
    verbose=1,
)
pred_test = np.clip(pred_test, 0.0, 1.0).astype(np.float32)

pred_train = model.predict(
    tf.data.Dataset.from_tensor_slices(train_text).batch(batch_size),
    verbose=0,
)
pred_train = np.clip(pred_train, 0.0, 1.0).astype(np.float32)

rng = np.random.RandomState(SEED)
pred_test_j = pred_test + rng.normal(0.0, 1e-7, size=pred_test.shape).astype(np.float32)
pred_train_j = pred_train + rng.normal(0.0, 1e-7, size=pred_train.shape).astype(
    np.float32
)

pred_cal = np.empty_like(pred_test_j, dtype=np.float32)

for j in range(pred_test_j.shape[1]):
    x = pred_train_j[:, j].astype(np.float32)
    t = y[:, j].astype(np.float32)

    order_x = np.argsort(x, kind="mergesort")
    x_sorted = x[order_x]
    t_sorted = t[order_x]

    pos = np.searchsorted(
        x_sorted, pred_test_j[:, j].astype(np.float32), side="right"
    ).astype(np.float32)
    u = (pos - 0.5) / float(len(x_sorted))
    u = np.clip(u, 0.0, 1.0)

    order_t = np.argsort(t_sorted, kind="mergesort")
    t_q = t_sorted[order_t]

    q_idx = np.minimum((u * (len(t_q) - 1)).astype(np.int64), len(t_q) - 1)
    pred_cal[:, j] = t_q[q_idx].astype(np.float32)

pred = np.clip(pred_cal, 0.0, 1.0).astype(np.float32)

sub = pd.DataFrame({"qa_id": df_test["qa_id"].values})

if pred.shape != (len(df_test), len(output_labels)):
    raise ValueError(
        f"Prediction shape {pred.shape} does not match test shape "
        f"({len(df_test)}, {len(output_labels)})."
    )

sub.loc[:, output_labels] = pred
sub = sub[df_sample.columns.tolist()]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv saved at:", os.path.abspath(out_path))
