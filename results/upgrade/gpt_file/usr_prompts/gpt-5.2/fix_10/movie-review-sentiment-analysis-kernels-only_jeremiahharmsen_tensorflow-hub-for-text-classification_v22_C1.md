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

# 5. Target score

0.64736

# 6. Current score

0.51837

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51546) has done: 'I fix the environment/runtime issues by replacing the deprecated/absent `tf.compat.v1.estimator` and the TF-Hub embedding column (which requires internet access and triggers TF/TFHub/protobuf issues) with an equivalent TensorFlow/Keras text model that trains locally. I keep the core approach the same (phrase text → embedding → small DNN → 5-class softmax, trained with dropout and Adagrad) and preserve the same train/validation split by `SentenceId`. I also ensure the code reads from the correct Kaggle paths and always writes a valid `submission.csv` with exactly `PhraseId,Sentiment`. Finally, I make the pipeline deterministic and robust so it runs end-to-end within the time limit.'
- What this solution (achieved 0.51888) has done: 'Main bottlenecks are (1) forcing pure-Python protobuf (very slow graph/model execution), (2) expensive `.isin(...)` splits on the full phrase-level dataframe, (3) `TextVectorization.adapt()` and training/prediction running through large NumPy arrays without `tf.data` pipelining, and (4) extra full-train predictions/plots not needed for submission. The changes below keep the exact same model, loss, optimizer, and training step budget (~10000 steps), but speed up input pipelines (cached/prefetched `tf.data`), make the train/val split O(n) via a boolean mask, keep protobuf on the default fast implementation, and avoid unnecessary heavy computations (confusion-matrix predict/plot) that don’t affect the submission. Seeds/determinism are preserved.'
- What this solution (achieved 0.19845) has done: 'The timeout is dominated by TensorFlow spending extra time on deterministic ops plus slow Python-protobuf parsing, and by an inefficient input pipeline that caches *after* shuffle (forcing large in-memory caching) and re-tokenizes strings every epoch in Python. I keep the exact same model, loss, optimizer, and total training steps, but make the pipeline equivalent and faster by (1) switching to the C++ protobuf implementation (safe and standard), (2) caching the *vectorized integer sequences* once so epochs don’t repeatedly run `TextVectorization`, and (3) avoiding caching a shuffled dataset while still preserving shuffling semantics. I also keep determinism/seeds intact and avoid any training-step/epoch changes.'
- What this solution (achieved 0.51837) has done: 'The timeout is dominated by (1) forcing the slow pure-Python protobuf implementation and (2) spending ~10,000 optimizer steps inside `model.fit`, where per-step overhead is inflated by `TextVectorization` running inside the `tf.data` pipeline and by caching/shuffling choices. I keep the exact model, loss, optimizer, step budget, and train/validation split semantics, but move vectorization out of the per-epoch pipeline (precompute integer token sequences once) and feed the model pre-vectorized tensors for training/validation/test. I also remove the protobuf override (keeping determinism) and adjust `tf.data` to avoid unnecessary `.cache()` of already-materialized arrays while still using prefetching. These changes are computation-equivalent (same vocabulary, same tokenization settings, same number of training steps) but cut substantial runtime overhead.'

# 9. Code solution

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
)

model.summary()

batch_size = 256
steps_per_epoch = int(np.ceil(len(x_train) / batch_size))
epochs = int(np.ceil(10000 / steps_per_epoch))

print("Batch size:", batch_size)
print("Steps/epoch:", steps_per_epoch)
print("Epochs (to reach ~10000 steps):", epochs)

AUTOTUNE = tf.data.AUTOTUNE

x_train_tok = vectorize(tf.constant(x_train)).numpy()
x_val_tok = vectorize(tf.constant(x_val)).numpy()

train_ds = tf.data.Dataset.from_tensor_slices((x_train_tok, y_train))
shuffle_buf = min(len(x_train_tok), 100_000)
train_ds = (
    train_ds.shuffle(shuffle_buf, seed=0, reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=False
    )
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val_tok, y_val))
val_ds = (
    val_ds.batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    verbose=2,
)




## === cell 3
train_eval = model.evaluate(train_ds, verbose=0)
val_eval = model.evaluate(val_ds, verbose=0)

print(f"Training set accuracy: {train_eval[1]:.6f}")
print(f"Validation set accuracy: {val_eval[1]:.6f}")




## === cell 4
DO_PLOT_CONFUSION = False

if DO_PLOT_CONFUSION:
    train_pred = np.argmax(
        model.predict(train_ds.map(lambda x, y: x), verbose=0),
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
x_test_tok = vectorize(tf.constant(x_test)).numpy()

test_ds = tf.data.Dataset.from_tensor_slices(x_test_tok)
test_ds = test_ds.batch(512).prefetch(tf.data.AUTOTUNE)

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
