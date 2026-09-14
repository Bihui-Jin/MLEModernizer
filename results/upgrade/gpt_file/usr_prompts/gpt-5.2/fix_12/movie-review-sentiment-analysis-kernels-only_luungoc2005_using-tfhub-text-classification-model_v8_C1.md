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

0.58945

# 6. Current score

0.5167

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow/TF-Hub incompatibility that’s causing both the protobuf `MessageFactory` error and the “KerasLayer is not a keras.Layer” error by using a local, built-in text vectorization + embedding pipeline (no Hub download, no protobuf issues). This keeps the overall approach the same (text embedding → Dense(500) → Dense(100) → softmax) while making it runnable in the Kaggle environment you described. I also ensure the model variable is always defined before training/prediction and that the submission is written as a proper `.csv` with the exact required columns and row count. The rest of the training loop and submission creation logic stays intact.'
- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow import/runtime failure caused by the protobuf API mismatch (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF (and clearing any already-imported protobuf modules). This keeps your modeling/training code intact and should let the notebook run end-to-end again in the provided environment. To nudge accuracy toward the target with minimal semantic change, I also ensure the embedding layer uses `input_dim=max_tokens` (not inferred) and that the submission is aligned to `test.tsv` PhraseIds (not relying on sampleSubmission ordering). The script then reliably write a valid `.csv` submission with the correct header and row count.'
- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` mismatch by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports and by proactively removing any already-imported `google.protobuf` modules so TensorFlow can’t bind against the incompatible C++ runtime. This is a correctness/runtime fix and should not change your modeling logic (TextVectorization → Embedding → GAP → Dense(500) → Dense(100) → softmax). I also make the embedding `input_dim` match the actual vocabulary size learned by `TextVectorization` (clipped by `max_tokens`) to avoid wasted/unused indices and slightly improve stability/accuracy without changing the architecture. Everything else (training loop, optimizer, submission formatting) remains the same and still write a valid `submission.csv`.'
- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` mismatch by forcing TensorFlow to use the pure-Python protobuf runtime and preventing the incompatible C++ implementation from loading (this is a runtime-only fix and should be score-neutral). Then I keep your exact modeling/training pipeline (TextVectorization → Embedding → GAP → Dense(500) → Dense(100) → softmax with Adagrad) but correct one logic issue that can hurt accuracy: ensure the Embedding `input_dim` is at least the maximum token index produced by the vectorizer (including OOV/mask indices), avoiding out-of-range/unused-index behavior. Finally, I keep submission formatting identical but make sure everything runs end-to-end and writes `submission.csv` with correct columns and row count.'
- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime early and also upgrading the environment variable to protobuf implementation version 3 (protobuf 6 expects this), while clearing any already-imported protobuf modules before importing TensorFlow. This is a runtime-only fix and does not change your model/training logic. Then I keep your exact TextVectorization→Embedding→GAP→Dense(500)→Dense(100) pipeline, but correct a subtle Embedding `input_dim` off-by-few issue by ensuring it’s at least the maximum token index produced by the vectorizer (which can improve stability/accuracy slightly without changing architecture). Finally, I keep the submission creation the same but ensure it always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.51666) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility (the `MessageFactory.GetPrototype` AttributeError) by forcing the pure-Python protobuf runtime early and downgrading the loaded protobuf modules before importing TensorFlow. Then I keep your exact model/training pipeline (TextVectorization → Embedding → GAP → Dense(500) → Dense(100) → softmax with Adagrad, same steps/batch sizing) but correct a subtle vocabulary/embedding indexing issue by setting the Embedding `input_dim` to `vectorize.vocabulary_size()` (which includes reserved tokens) to avoid any out-of-range/unused-index behavior and typically improves accuracy slightly. Finally, I keep the same submission creation but ensure it always writes a valid `submission.csv` with the required columns aligned to `test.tsv` PhraseIds.'
- What this solution (achieved 0.5167) has done: 'I fix the runtime crash in the protobuf patching logic by avoiding calls to the missing `MessageFactory.GetPrototype` entirely and instead (safely) aliasing `GetPrototype` to `GetMessageClass` when appropriate, without instantiating `MessageFactory`. This keeps the TensorFlow/TextVectorization model pipeline unchanged while making the notebook run end-to-end again. To nudge accuracy upward toward your target with minimal semantic impact, I also add a small validation split and train on the remaining data with the same optimizer/loss/architecture, which typically improves generalization versus training/evaluating on the full training set. Finally, the script still write a valid `submission.csv` with the required columns aligned to `test.tsv`.'
- What this solution (achieved 0.5167) has done: 'I fix the runtime crash on TensorFlow import by applying a safe protobuf compatibility shim (aliasing `MessageFactory.GetPrototype` to `GetMessageClass` when needed) before importing TensorFlow, without changing your model/training logic. Then I fix the `TextVectorization` construction error by removing the unsupported `num_oov_indices` argument (Keras 3/TensorFlow 2.18 doesn’t accept it) while keeping the same vectorization→embedding→GAP→Dense(500)→Dense(100)→softmax pipeline. Finally, I ensure `model` is always defined so later cells don’t fail, and the script writes a valid `submission.csv` with the exact required columns aligned to `test.tsv` PhraseIds.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory"):
        _MF = _message_factory.MessageFactory
        if not hasattr(_MF, "GetPrototype") and hasattr(_MF, "GetMessageClass"):
            setattr(_MF, "GetPrototype", _MF.GetMessageClass)
except Exception as _e:
    print("Warning: protobuf shim setup failed (continuing):", repr(_e))

CANDIDATE_INPUT_DIRS = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/movie-review-sentiment-analysis-kernels-only",
]
INPUT_DIR = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.exists(d):
        if os.path.exists(os.path.join(d, "train.tsv")) or os.path.exists(
            os.path.join(d, "test.tsv")
        ):
            INPUT_DIR = d
            break
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                if "train.tsv" in files and "test.tsv" in files:
                    INPUT_DIR = root
                    break
    if INPUT_DIR is not None:
        break

if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
if os.path.exists(INPUT_DIR):
    print("Top-level listing:", os.listdir(INPUT_DIR)[:50])



## === cell 1
seed = 197
import random

random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

try:
    import google.protobuf  # noqa: F401
except Exception as e:
    print("Warning: protobuf import failed (continuing):", repr(e))

import tensorflow as tf

tf.random.set_seed(seed)

print("TF:", tf.__version__)
print("PROTOBUF_IMPL:", os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"))
print(
    "PROTOBUF_IMPL_VER:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_path = os.path.join(INPUT_DIR, "train.tsv")
test_path = os.path.join(INPUT_DIR, "test.tsv")

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print(train_df.shape, test_df.shape)
train_df.head()



## === cell 3
x_text = train_df["Phrase"].astype(str).values
y = train_df["Sentiment"].astype("int32").values

x_test_text = test_df["Phrase"].astype(str).values
num_classes = 5



## === cell 4
max_tokens = 50000
seq_len = 40
embed_dim = 50  # keep same dimensionality as nnlm-en-dim50 to stay close in spirit

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=seq_len,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    name="text_vectorization",
)
vectorize.adapt(x_text)

vocab_size = int(vectorize.vocabulary_size())
embed_input_dim = max(vocab_size, 2)

text_input = tf.keras.Input(shape=(), dtype=tf.string, name="Phrase")
x = vectorize(text_input)

x = tf.keras.layers.Embedding(
    input_dim=embed_input_dim, output_dim=embed_dim, name="embedding"
)(x)

x = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)

x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=text_input, outputs=out)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.003)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 5
steps = 10000
batch_size = 256

steps_per_epoch = int(np.ceil(len(x_text) / batch_size))
epochs = int(np.ceil(steps / steps_per_epoch))

print(
    "N:",
    len(x_text),
    "batch_size:",
    batch_size,
    "steps_per_epoch:",
    steps_per_epoch,
    "epochs:",
    epochs,
)



## === cell 6
history = model.fit(
    x_text,
    y,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_split=0.1,
    verbose=2,
)



## === cell 7
train_loss, train_acc = model.evaluate(x_text, y, batch_size=batch_size, verbose=0)
print(f"Training set accuracy: {train_acc:.6f}")



## === cell 8
sample_path = os.path.join(INPUT_DIR, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    found = None
    for root, _, files in os.walk(INPUT_DIR):
        if "sampleSubmission.csv" in files:
            found = os.path.join(root, "sampleSubmission.csv")
            break
    if found is None:
        raise FileNotFoundError(
            "sampleSubmission.csv not found under INPUT_DIR: " + INPUT_DIR
        )
    sample_path = found

sub_template = pd.read_csv(sample_path)
sub_template.head()



## === cell 9
probs = model.predict(x_test_text, batch_size=batch_size, verbose=0)
pred = np.argmax(probs, axis=1).astype(int)

sub = pd.DataFrame({"PhraseId": test_df["PhraseId"].values, "Sentiment": pred})

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())



## === cell 10
assert list(sub.columns) == ["PhraseId", "Sentiment"]
assert sub["Sentiment"].between(0, 4).all()
assert sub.shape[0] == test_df.shape[0]
sub.head()
