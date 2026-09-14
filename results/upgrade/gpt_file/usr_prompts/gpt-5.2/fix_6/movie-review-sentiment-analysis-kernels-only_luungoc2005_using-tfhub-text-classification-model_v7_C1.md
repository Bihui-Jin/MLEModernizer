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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.58839

# 6. Current score

0.51606

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51634) has done: 'We need to fix two root issues preventing end-to-end execution: (1) an import-time protobuf incompatibility causing the TF Hub `MessageFactory.GetPrototype` error, and (2) a Keras 3 compatibility issue where `hub.KerasLayer` is not recognized as a valid Keras layer in `Sequential`. The minimal robust fix is to avoid TF Hub entirely (no internet assumed anyway) and replace it with an in-graph `TextVectorization -> Embedding -> GlobalAveragePooling` block while keeping the same Dense stack, loss, optimizer, and training loop semantics. I also correct the input paths to the provided `/kaggle/input` layout and ensure the submission file is written as a valid `.csv` with `PhraseId,Sentiment` matching the test order.'
- What this solution (achieved 0.51634) has done: 'I fix the TensorFlow import crash caused by the protobuf API change by pinning a compatible pure-Python protobuf implementation *before* importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in this environment). I keep your current non-TFHub model (TextVectorization → Embedding → pooling → Dense stack) intact, since it already avoids the TFHub/KerasLayer issue and preserves the solution’s core logic. I also make the input directory detection robust (preferring `/kaggle/input` but falling back to the provided `/kaggle/data` layout) and ensure the submission is written as a valid `.csv` with exactly `PhraseId,Sentiment` in the test row order.'
- What this solution (achieved 0.51634) has done: 'I fix the TensorFlow import crash caused by the protobuf 6 API change (`MessageFactory.GetPrototype` missing) by forcing TensorFlow to use the pure-Python protobuf runtime and by pinning an older protobuf at runtime (if needed) before importing TensorFlow. This is directly required to make the notebook run end-to-end and produce a submission CSV. I keep your current model core (TextVectorization → Embedding → GlobalAveragePooling → Dense stack, Adagrad, SparseCategoricalCrossentropy from_logits) unchanged, only adding the minimal environment/compat shims. I also make the data directory resolution keep working across `/kaggle/input` and `/kaggle/data`, and ensure the submission file is written with the required `PhraseId,Sentiment` columns.'
- What this solution (achieved 0.51606) has done: 'To move accuracy up toward 0.58839 with minimal disruption, I keep your exact model class (TextVectorization → Embedding → GlobalAveragePooling → Dense stack with Adagrad + SparseCategoricalCrossentropy logits) but fix two small issues that typically cap accuracy: (1) the Embedding `input_dim` should match the vectorizer’s realized vocabulary size (not the max_tokens constant), and (2) provide a tiny validation split plus `ReduceLROnPlateau` so Adagrad doesn’t get stuck at a suboptimal learning rate (this doesn’t change the training approach, just stabilizes optimization). I also add `ngrams=2` to the vectorizer, which is still the same feature-extraction family (TextVectorization) but usually yields a meaningful bump on this dataset with very small code changes. The submission writing and file paths remain unchanged, and the script still runs end-to-end within the time limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing

print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", e)
print("Listing /kaggle/input (if exists):")
try:
    print(os.listdir("/kaggle/input"))
except Exception as e:
    print("Could not list /kaggle/input:", e)
print("Listing /kaggle/data (if exists):")
try:
    print(os.listdir("/kaggle/data"))
except Exception as e:
    print("Could not list /kaggle/data:", e)



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        if int(pb_ver.split(".", 1)[0]) >= 6:
            print(
                "Detected protobuf",
                pb_ver,
                "- attempting to install protobuf<5 for TF compatibility...",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib
            import google.protobuf as gp

            importlib.reload(gp)
    except Exception as e:
        print("Protobuf compat install step skipped/failed (continuing):", repr(e))


_ensure_protobuf_compat()

seed = 197
import random
import tensorflow as tf

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TensorFlow:", tf.__version__)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)



## === cell 2
import pandas as pd
import os


def find_data_dir():
    candidates = [
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only",
        "/kaggle/input",
        "/kaggle/data/movie-review-sentiment-analysis-kernels-only",
        "/kaggle/data",
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "train.tsv")) and os.path.exists(
            os.path.join(d, "test.tsv")
        ):
            return d
    raise FileNotFoundError(
        "Could not find train.tsv/test.tsv under expected Kaggle input/data directories."
    )


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)

train_path = os.path.join(DATA_DIR, "train.tsv")
test_path = os.path.join(DATA_DIR, "test.tsv")
sample_path = os.path.join(DATA_DIR, "sampleSubmission.csv")

train = pd.read_csv(train_path, sep="\t")
test = pd.read_csv(test_path, sep="\t")

print(train.shape, test.shape)
print(train.columns.tolist())



## === cell 3
train.head()



## === cell 4
train_df = train.sample(frac=1, random_state=seed).reset_index(drop=True)

x_train = train_df["Phrase"].astype(str).values
y_train = train_df["Sentiment"].astype(np.int64).values

x_test = test["Phrase"].astype(str).values

num_classes = 5



## === cell 5
max_tokens = 50000
sequence_length = 40
embedding_dim = 50  # matches "dim50" spirit of the original hub module

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
    ngrams=2,
    name="text_vectorization",
)
vectorizer.adapt(x_train)

vocab_size = len(vectorizer.get_vocabulary())
print("Vectorizer vocab_size:", vocab_size)

inputs = tf.keras.Input(shape=(), dtype=tf.string, name="Phrase")
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)

x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)  # logits

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_embed_mlp")

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.003)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()



## === cell 6
batch_size = 256
steps = 10000
epochs = int(np.ceil((steps * batch_size) / len(x_train)))
epochs = max(1, epochs)

print("Train rows:", len(x_train), "Batch size:", batch_size, "Approx epochs:", epochs)



## === cell 7
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1,
    )
]

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.05,
    callbacks=callbacks,
    verbose=2,
)



## === cell 8
train_loss, train_acc = model.evaluate(
    x_train, y_train, batch_size=batch_size, verbose=0
)
print(f"Training set accuracy: {train_acc:.6f}")



## === cell 9
sub = pd.read_csv(sample_path)
sub.head()



## === cell 10
logits = model.predict(x_test, batch_size=batch_size, verbose=0)
pred = np.argmax(logits, axis=1).astype(np.int64)

if len(pred) != len(test):
    raise ValueError(f"Prediction length {len(pred)} != test rows {len(test)}")

submission = pd.DataFrame(
    {"PhraseId": test["PhraseId"].astype(np.int64).values, "Sentiment": pred}
)

submission = submission[["PhraseId", "Sentiment"]]
submission["PhraseId"] = submission["PhraseId"].astype(np.int64)
submission["Sentiment"] = submission["Sentiment"].astype(np.int64)

out_path = "sub_tfhub.csv"  # keep filename stable, but ensure it's a valid .csv
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", submission.shape)
print(submission.head())

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["PhraseId", "Sentiment"]
assert submission["Sentiment"].between(0, 4).all()
