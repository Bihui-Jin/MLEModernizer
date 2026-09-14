# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
import sys

print("Skipping pip install; Python:", sys.version)



## === cell 2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))



## === cell 3
DATA_ROOT = "/kaggle/input/lmsys-chatbot-arena"
train_path = f"{DATA_ROOT}/train.csv"
test_path = f"{DATA_ROOT}/test.csv"
sample_path = f"{DATA_ROOT}/sample_submission.csv"

read_csv_kwargs = {}
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(train_path, **read_csv_kwargs)
test = pd.read_csv(test_path, **read_csv_kwargs)
sample_submission = pd.read_csv(sample_path, **read_csv_kwargs)

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Sample Submission Shape:", sample_submission.shape)



## === cell 4
for col in ["prompt", "response_a", "response_b"]:
    train[col] = train[col].fillna("")
    test[col] = test[col].fillna("")

sep = " [SEP] "
train["combined_text"] = (
    train["prompt"].astype(str)
    + sep
    + train["response_a"].astype(str)
    + sep
    + train["response_b"].astype(str)
)
test["combined_text"] = (
    test["prompt"].astype(str)
    + sep
    + test["response_a"].astype(str)
    + sep
    + test["response_b"].astype(str)
)

target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
y = train[target_cols].values.astype("float32")

print(
    "y shape:",
    y.shape,
    "row sums (min/mean/max):",
    y.sum(1).min(),
    y.sum(1).mean(),
    y.sum(1).max(),
)



## === cell 5
from sklearn.model_selection import StratifiedKFold

MAX_LEN = 512
VOCAB_SIZE = 50000
OOV_TOKEN = "<OOV>"

vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize=None,
    split="whitespace",
    vocabulary=None,
)

text_ds = tf.data.Dataset.from_tensor_slices(train["combined_text"].values).batch(1024)
vectorizer.adapt(text_ds)

X_train = (
    vectorizer(tf.constant(train["combined_text"].values))
    .numpy()
    .astype(np.int32, copy=False)
)
X_test = (
    vectorizer(tf.constant(test["combined_text"].values))
    .numpy()
    .astype(np.int32, copy=False)
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

y_class = np.argmax(y, axis=1)


def build_gru_model(vocab_size: int, max_len: int) -> keras.Model:
    inp = keras.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model




## === cell 6
n_folds = 5
fold_predictions = []

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

BATCH_SIZE = 64
EPOCHS = 2  # unchanged

vocab_effective = int(vectorizer.vocabulary_size())
print("Effective vocab size:", vocab_effective)

AUTOTUNE = tf.data.AUTOTUNE

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_class)):
    print(f"\n=== Fold {fold+1}/{n_folds} ===")
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    train_ds = train_ds.shuffle(
        buffer_size=len(X_tr), seed=SEED + fold, reshuffle_each_iteration=True
    )
    train_ds = (
        train_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va))
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

    model = build_gru_model(vocab_effective, MAX_LEN)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    )
    preds = model.predict(test_ds, verbose=0)
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2 or preds.shape[1] != 3 or preds.shape[0] != X_test.shape[0]:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    fold_predictions.append(preds)
    print("Fold preds shape:", preds.shape)



## === cell 7
print("\n" + "=" * 80)
print("AVERAGING PREDICTIONS")
print("=" * 80)

if len(fold_predictions) == 0:
    avg_predictions = np.full((len(test), 3), 1 / 3, dtype=np.float32)
else:
    avg_predictions = np.mean(np.stack(fold_predictions, axis=0), axis=0)

eps = 1e-7
avg_predictions = np.clip(avg_predictions, eps, 1.0)
avg_predictions = avg_predictions / avg_predictions.sum(axis=1, keepdims=True)

print("Average predictions shape:", avg_predictions.shape)



## === cell 8
print("\n" + "=" * 80)
print("CREATING SUBMISSION")
print("=" * 80)

submission = sample_submission.copy()
submission = submission.merge(test[["id"]], on="id", how="right", validate="one_to_one")

submission["winner_model_a"] = avg_predictions[:, 0]
submission["winner_model_b"] = avg_predictions[:, 1]
submission["winner_tie"] = avg_predictions[:, 2]

output_path = "/kaggle/working/submission.csv"
submission[["id", "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(
    output_path, index=False
)

print("Saved:", output_path)
print("Shape:", submission.shape)



## === cell 9
print("\n" + "=" * 80)
print("SUBMISSION PREVIEW")
print("=" * 80)
print(submission.head(20))



## === cell 10
print("\n" + "=" * 80)
print("PREDICTION STATISTICS")
print("=" * 80)

print(
    f"Model A - Mean: {avg_predictions[:, 0].mean():.6f}, Std: {avg_predictions[:, 0].std():.6f}, Min: {avg_predictions[:, 0].min():.6f}, Max: {avg_predictions[:, 0].max():.6f}"
)
print(
    f"Model B - Mean: {avg_predictions[:, 1].mean():.6f}, Std: {avg_predictions[:, 1].std():.6f}, Min: {avg_predictions[:, 1].min():.6f}, Max: {avg_predictions[:, 1].max():.6f}"
)
print(
    f"Tie     - Mean: {avg_predictions[:, 2].mean():.6f}, Std: {avg_predictions[:, 2].std():.6f}, Min: {avg_predictions[:, 2].min():.6f}, Max: {avg_predictions[:, 2].max():.6f}"
)



## === cell 11
print("\n" + "=" * 80)
print("VERIFYING PROBABILITIES SUM TO 1")
print("=" * 80)

prob_sums = avg_predictions.sum(axis=1)
print("Min sum:", float(prob_sums.min()))
print("Max sum:", float(prob_sums.max()))
print("Mean sum:", float(prob_sums.mean()))



## === cell 12
print("\n" + "=" * 80)
print("INFERENCE COMPLETED")
print("=" * 80)
print("Submission file saved:", output_path)
print(
    "Submission columns:",
    submission[
        ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    ].columns.tolist(),
)
