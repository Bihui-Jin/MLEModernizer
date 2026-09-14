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

0.62585

# 6. Current score

0.51314

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51314) has done: 'I fix the environment import/runtime failures by removing the TF-Hub Estimator path (which breaks due to incompatible TensorFlow/TF-Hub/protobuf in this Kaggle image) and replace it with a minimal, equivalent text-classification pipeline that still trains a neural network on the Phrase text and predicts 5 sentiment classes. I keep the same train/validation split by SentenceId and produce a proper `submission.csv` with columns `PhraseId,Sentiment`. To stay within Kaggle’s offline environment, I use Keras + `TextVectorization` (no internet downloads) and a small embedding + dense network trained on the provided TSVs. This run end-to-end and yield a non-trivial accuracy score (expected to move toward the target vs. the all-2 baseline).'
- What this solution (achieved 0.51314) has done: 'The first failure happens at TensorFlow import due to an incompatible protobuf runtime; the safest minimal fix is to force TensorFlow to use the pure-Python protobuf implementation before importing TF. The second failure is a real training bug: `Embedding(input_dim=MAX_TOKENS)` can be too small because `TextVectorization` may emit token IDs up to the actual vocabulary size it learned (which can exceed your chosen cap depending on TF version/behavior), so we set `input_dim` from the vectorizer’s learned vocabulary size. These two changes keep the same overall model/training approach and should both unblock execution and improve accuracy toward the target by allowing the intended vocabulary size to be used without runtime errors. The submission writing logic is kept the same but we also ensure the output is integer class labels 0–4.'
- What this solution (achieved 0.51314) has done: 'We fix the two execution blockers: the TensorFlow/protobuf `MessageFactory.GetPrototype` crash at import time, and the Adagrad/Embedding index-out-of-range error during training. The protobuf crash is best handled by pinning protobuf to the pure-Python implementation *and* forcing the older API path (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`) before importing TensorFlow. The embedding crash happens because `TextVectorization` can emit token IDs up to `max_tokens` (including special tokens), so we set `Embedding.input_dim` to `MAX_TOKENS` (or larger) rather than the observed vocabulary length. These changes keep the same model/training pipeline but make it run end-to-end and should improve accuracy versus the broken run, producing a valid `submission.csv`.'
- What this solution (achieved 0.51314) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation and, if that still fails in this Kaggle image, falling back to a compatible protobuf version via a local pip install before importing TensorFlow. Then I fix the Embedding/Adagrad out-of-range error by setting `Embedding.input_dim` to the actual `TextVectorization` vocabulary size (with a small safety margin) so all token IDs are valid. These changes keep the same overall Keras TextVectorization→Embedding→pooling→dense softmax architecture and training loop, but unblock training and should improve accuracy toward the target by letting the intended vocabulary be learned/used correctly. Finally, I ensure the submission is written as `submission.csv` with the required `PhraseId,Sentiment` columns.'
- What this solution (achieved 0.51314) has done: 'I fix the TensorFlow import crash by forcing a protobuf version that is known to work with TF in Kaggle (installed *before* importing TensorFlow), and keep the pure-Python protobuf setting for safety. Then I fix the Embedding/Adagrad “indices not in [0, …)” runtime error by ensuring the Embedding `input_dim` is at least `MAX_TOKENS + 2`, because `TextVectorization(max_tokens=...)` can still emit ids up to that cap regardless of the observed vocabulary length. These are execution-blocking bugs; the resulting model/training loop and architecture stay the same, but the run complete and should improve accuracy from the broken state toward your target. Finally, I keep the same submission format and ensure `submission.csv` is written with correct columns and integer labels 0–4.'
- What this solution (achieved 0.51314) has done: 'I fix the runtime crash during training by making the Embedding `input_dim` consistent with the actual integer IDs produced by `TextVectorization` (the current mismatch is what triggers the Adagrad gather “indices not in [0, …)” error). I keep the same TextVectorization→Embedding→pooling→dense softmax model and the same train/validation split, only adjusting the embedding size to `max_tokens` (with a small safety margin) and adding a one-time debug check for the maximum token id to ensure it cannot exceed `input_dim - 1`. This change is correctness/stability focused and should also improve accuracy versus the broken/partially-trained run by allowing training to complete as intended. Finally, the script still write a valid `submission.csv` with `PhraseId,Sentiment` integer labels 0–4.'
- What this solution (achieved 0.51314) has done: 'I fix the training crash by making the Embedding `input_dim` match the actual ID range produced by `TextVectorization` (it can emit IDs larger than `MAX_TOKENS` once special/OOV buckets are included), and I add a quick full-train check to guarantee no token index exceeds the embedding size. This is an execution-blocking bug (not a modeling change) and should also improve accuracy versus a partially/broken training run because training can complete as intended. I also remove the hard-coded `MAX_TOKENS+2` assumption and instead size the embedding from the vectorizer configuration plus a small safety margin. The rest of the pipeline (split by SentenceId, vectorizer→embedding→pooling→dense softmax, Adagrad, epochs, and submission format) stays the same.'
- What this solution (achieved 0.51314) has done: 'We fix the training crash caused by `TextVectorization` emitting token IDs larger than the Embedding/optimizer slot variable size (the error shows indices up to ~14652 while the optimizer thinks the variable is size ~10241). The minimal, score-improving fix is to make `TextVectorization` deterministically cap IDs by setting `pad_to_max_tokens=True` and then set `Embedding.input_dim` exactly to the configured `max_tokens` (plus the expected special tokens are already included in that output space). This preserves the same model architecture/training loop while ensuring indices never go out of range, so training completes and the resulting accuracy should move up toward your target. Finally, we keep the same submission writing code and ensure the file is produced as `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = None

need_pb_pin = True
if pb_ver is not None:
    try:
        major = int(pb_ver.split(".", 1)[0])
        need_pb_pin = major >= 4
    except Exception:
        need_pb_pin = True

if need_pb_pin:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"]
    )

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn import model_selection

print("TensorFlow:", tf.__version__)

np.random.seed(0)
tf.random.set_seed(0)



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
    train_path = "/kaggle/input/train.tsv"
    test_path = "/kaggle/input/test.tsv"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        train_path = (
            "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv"
        )
        test_path = (
            "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv"
        )

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)].copy()
    train_df = train_df[train_df["SentenceId"].isin(train_indices)].copy()

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 256
MAX_TOKENS = 40000
SEQ_LEN = 40  # phrases are short; keep modest to run fast within limits


def df_to_ds(df, training=True, with_labels=True):
    if with_labels:
        ds = tf.data.Dataset.from_tensor_slices(
            (df["Phrase"].astype(str).values, df["Sentiment"].astype(np.int64).values)
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices(df["Phrase"].astype(str).values)
    if training:
        ds = ds.shuffle(min(len(df), 20000), seed=0, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = df_to_ds(train_df, training=True, with_labels=True)
val_ds = df_to_ds(validation_df, training=False, with_labels=True)
test_ds = df_to_ds(test_df, training=False, with_labels=False)

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    pad_to_max_tokens=True,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)

vectorize.adapt(train_df["Phrase"].astype(str).values)

cfg = vectorize.get_config()
max_tokens_cfg = int(cfg.get("max_tokens", MAX_TOKENS))
print(
    "Vectorizer max_tokens:",
    max_tokens_cfg,
    "| pad_to_max_tokens:",
    cfg.get("pad_to_max_tokens"),
)

embedding_input_dim = max_tokens_cfg

max_id_full = 0
for batch_text, _ in train_ds:
    ids = vectorize(batch_text)
    batch_max = int(tf.reduce_max(ids).numpy())
    if batch_max > max_id_full:
        max_id_full = batch_max
print(
    "Observed max token id on full train_ds:",
    max_id_full,
    "| embedding_input_dim:",
    embedding_input_dim,
)
if max_id_full >= embedding_input_dim:
    embedding_input_dim = max_id_full + 2
    print("Adjusted embedding_input_dim to:", embedding_input_dim)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(1,), dtype=tf.string),
        vectorize,
        tf.keras.layers.Embedding(
            input_dim=int(embedding_input_dim), output_dim=128, mask_zero=True
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

EPOCHS = 6
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2749909581.py in <cell line: 0>()
     87 
     88 EPOCHS = 6
---> 89 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
     90 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node adagrad/cond/cond/GatherV2_1 defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/2749909581.py", line 89, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 80, in train_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py", line 383, in apply_gradients

  File "/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py", line 448, in apply

  File "/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py", line 511, in _backend_apply_gradients

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py", line 120, in _backend_update_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py", line 134, in _distributed_tf_update_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py", line 131, in apply_grad_to_update_var

  File "/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adagrad.py", line 95, in update_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/numpy.py", line 6055, in divide

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/sparse.py", line 770, in sparse_wrapper

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/sparse.py", line 758, in func_for_union_indices

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/sparse.py", line 139, in indexed_slices_union_indices_and_values

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/sparse.py", line 142, in <lambda>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/sparse.py", line 136, in values_for_union

indices[450] = 11418 is not in [0, 10241)
	 [[{{node adagrad/cond/cond/GatherV2_1}}]] [Op:__inference_multi_step_on_iterator_26546]

## === cell 3
train_eval = model.evaluate(train_ds, verbose=0)
val_eval = model.evaluate(val_ds, verbose=0)

print("Training set accuracy:", float(train_eval[1]))
print("Validation set accuracy:", float(val_eval[1]))

preds_train = np.argmax(model.predict(train_ds, verbose=0), axis=1)
true_train = train_df["Sentiment"].values.astype(np.int32)

cm_out = tf.math.confusion_matrix(
    labels=true_train,
    predictions=preds_train.astype(np.int32),
    num_classes=5,
).numpy()

cm_out = cm_out.astype(float) / np.maximum(cm_out.sum(axis=1, keepdims=True), 1.0)

sns.heatmap(
    cm_out, annot=True, xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



## === cell 4
probs_test = model.predict(test_ds, verbose=0)
preds_test = np.argmax(probs_test, axis=1).astype(np.int64)

preds_test = np.clip(preds_test, 0, 4).astype(int)

submission = pd.DataFrame(
    {"PhraseId": test_df["PhraseId"].values.astype(int), "Sentiment": preds_test}
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Sentiment value counts:\n", submission["Sentiment"].value_counts().sort_index())
