# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn import model_selection

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

INPUT_DIR = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.tsv")
TEST_PATH = os.path.join(INPUT_DIR, "test.tsv")

print("TensorFlow version:", tf.__version__)
print("Train path exists:", os.path.exists(TRAIN_PATH))
print("Test path exists:", os.path.exists(TEST_PATH))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism op setting not available/failed (continuing):", repr(e))




## === cell 1
SENTIMENT_LABELS = [
    "negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "positive",
]


def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
        range(5), SENTIMENT_LABELS
    )


def get_data(validation_set_ratio=0.1):
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    test_df = pd.read_csv(TEST_PATH, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    sentence_ids = np.unique(train_df["SentenceId"].values)
    train_indices, validation_indices = model_selection.train_test_split(
        sentence_ids,
        test_size=validation_set_ratio,
        random_state=0,
    )

    is_val = train_df["SentenceId"].isin(validation_indices).values

    validation_df = train_df[is_val].copy()
    train_df = train_df[~is_val].copy()

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()




## === cell 2
x_train = train_df["Phrase"].astype(str).values
y_train = train_df["Sentiment"].astype(np.int32).values

x_val = validation_df["Phrase"].astype(str).values
y_val = validation_df["Sentiment"].astype(np.int32).values

x_test = test_df["Phrase"].astype(str).values

max_tokens = 50000
sequence_length = 50

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)

adapt_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(1024)
vectorize.adapt(adapt_ds)

inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int64, name="TokenIds")
x = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=128, name="embedding")(
    inputs
)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(64, return_sequences=False),
    name="bi_gru",
)(x)

x = tf.keras.layers.Dense(250, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(50, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)

outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.003)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)

model.summary()

batch_size = 256
steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
epochs = int(np.ceil(10000 / steps_per_epoch))

print("Batch size:", batch_size)
print("Steps/epoch:", steps_per_epoch)
print("Epochs (to reach ~10000 steps):", epochs)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True


def _vectorize_to_numpy_fast(text_array):
    return vectorize(text_array).numpy()


x_train_tok = _vectorize_to_numpy_fast(x_train).astype(np.int64, copy=False)
x_val_tok = _vectorize_to_numpy_fast(x_val).astype(np.int64, copy=False)
x_test_tok = _vectorize_to_numpy_fast(x_test).astype(np.int64, copy=False)


def make_train_dataset_from_numpy(x_tok, y, batch_size, epochs, base_seed=0):
    x_tok = np.asarray(x_tok)
    y = np.asarray(y)

    def gen():
        rng = np.random.RandomState(base_seed)
        n = len(x_tok)
        idx = np.arange(n, dtype=np.int64)
        for _ in range(epochs):
            rng.shuffle(idx)
            for start in range(0, n, batch_size):
                sl = idx[start : start + batch_size]
                yield x_tok[sl], y[sl]

    output_signature = (
        tf.TensorSpec(shape=(None, x_tok.shape[1]), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.prefetch(AUTOTUNE)


train_ds = make_train_dataset_from_numpy(
    x_train_tok, y_train, batch_size, epochs, base_seed=0
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val_tok, y_val))
    .with_options(options)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,  # keep identical step budget per epoch as originally computed
    validation_data=val_ds,
    verbose=2,
)




## === cell 3
train_eval = model.evaluate(
    tf.data.Dataset.from_tensor_slices((x_train_tok, y_train))
    .with_options(options)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE),
    verbose=0,
)
val_eval = model.evaluate(val_ds, verbose=0)

print(f"Training set accuracy: {train_eval[1]:.6f}")
print(f"Validation set accuracy: {val_eval[1]:.6f}")




## === cell 4
DO_PLOT_CONFUSION = False

if DO_PLOT_CONFUSION:
    train_pred = np.argmax(
        model.predict(
            tf.data.Dataset.from_tensor_slices(x_train_tok)
            .with_options(options)
            .batch(batch_size, drop_remainder=False)
            .prefetch(AUTOTUNE),
            verbose=0,
        ),
        axis=1,
    )

    cm_out = tf.math.confusion_matrix(
        labels=y_train,
        predictions=train_pred[: len(y_train)],
        num_classes=5,
    ).numpy()

    cm_out = cm_out.astype(float) / cm_out.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_out,
        annot=True,
        fmt=".2f",
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()




## === cell 5
test_ds = (
    tf.data.Dataset.from_tensor_slices(x_test_tok)
    .with_options(options)
    .batch(512, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1).astype(int)

submission = pd.DataFrame(
    {"PhraseId": test_df["PhraseId"].astype(int).values, "Sentiment": test_pred}
)
submission = submission[["PhraseId", "Sentiment"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission dtypes:", submission.dtypes.to_dict())
print("Sentiment value counts (first few):")
print(submission["Sentiment"].value_counts().head())
