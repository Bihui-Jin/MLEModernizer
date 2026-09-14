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

No external packages required in the script and installed.

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

0.6436523260725276

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.51314) has done: 'Most of the 600s overrun is coming from LSTM training on long sequences (maxlen=300) with a relatively small batch size, plus some avoidable CPU overhead in the input pipeline. I keep the exact same model and training semantics, but speed things up by (1) using `tf.data` with vectorized shuffling instead of materializing a full shuffled copy in NumPy, (2) enabling `tf.data` determinism and performance options, and (3) increasing the training batch size (this doesn’t change the objective/epochs/model; it reduces steps/epoch and input overhead while keeping accuracy essentially unchanged). I also avoid eager `.numpy()` conversions by keeping tokenized sequences as `tf.Tensor` where possible and only converting when needed by scikit split. Finally, I keep the GloVe early-exit logic but make the vocab->index map construction faster/leaner.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np  # linear algebra
import pandas as pd  # data processing
import time

for p in ("../input", "/kaggle/input"):
    if os.path.exists(p):
        print("Listing", p, "->", os.listdir(p)[:50])
        break



## === cell 1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


base = _first_existing(
    [
        "../input/movie-review-sentiment-analysis-kernels-only",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only",
    ]
)
train_path = _first_existing(
    [
        os.path.join(base, "train.tsv"),
        "../input/movie-review-sentiment-analysis-kernels-only/train.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv",
    ]
)
test_path = _first_existing(
    [
        os.path.join(base, "test.tsv"),
        "../input/movie-review-sentiment-analysis-kernels-only/test.tsv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv",
    ]
)
sub_path = _first_existing(
    [
        os.path.join(base, "sampleSubmission.csv"),
        "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",
        "/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",
    ]
)

train = pd.read_csv(train_path, sep="\t")
test = pd.read_csv(test_path, sep="\t")
sub = pd.read_csv(sub_path, sep=",")

print("Loaded:", train.shape, test.shape, sub.shape)



## === cell 3
x = train["Phrase"].astype(str)
y = train["Sentiment"].astype(int)



## === cell 4
embedding_dim = 300
maxlen = 300
glove_path = "../input/gloveembeddings/glove.6B.300d.txt"

MAX_VOCAB = 60000

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=maxlen,
)

t0 = time.time()
vectorize.adapt(tf.data.Dataset.from_tensor_slices(x.values).batch(8192))
print("Vectorizer adapt seconds:", round(time.time() - t0, 3))

vocab = vectorize.get_vocabulary()  # index 0 reserved for padding, 1 for OOV
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

y = to_categorical(y, num_classes=5)



## === cell 5
t0 = time.time()
print(
    "Train vectorization deferred to tf.data pipeline (seconds):",
    round(time.time() - t0, 3),
)



## === cell 6
t0 = time.time()

effective_vocab = min(vocab_size, MAX_VOCAB)

embedding_matrix = np.zeros((effective_vocab, embedding_dim), dtype=np.float32)

inv_idx = {w: i for i, w in enumerate(vocab[:effective_vocab]) if i >= 2 and w}
needed_total = len(inv_idx)

if os.path.exists(glove_path):
    found = 0
    remaining = needed_total
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue
            w, sep, rest = line.partition(" ")
            if not sep:
                continue
            idx = inv_idx.get(w)
            if idx is None:
                continue
            vec = np.fromstring(rest, sep=" ", dtype=np.float32)
            if vec.shape[0] != embedding_dim:
                continue
            embedding_matrix[idx] = vec
            found += 1
            remaining -= 1
            if remaining == 0:
                break
    print(f"Loaded {found} in-vocab GloVe vectors (filtered, early-exit).")
else:
    print(f"GloVe file not found at: {glove_path}")
    print(
        "Proceeding with randomly initialized embedding matrix (trainable=False as in original)."
    )
    rng = np.random.RandomState(42)
    embedding_matrix[1:] = rng.normal(
        loc=0.0, scale=0.05, size=(effective_vocab - 1, embedding_dim)
    ).astype(np.float32)

print("Embedding prep seconds:", round(time.time() - t0, 3))



## === cell 7
y_labels = np.argmax(y, axis=1)
idx = np.arange(len(x), dtype=np.int32)
idx_train, idx_test, y_train, y_test = train_test_split(
    idx, y, test_size=0.3, random_state=42, stratify=y_labels
)
x_train_text = x.values[idx_train]
x_test_text = x.values[idx_test]



## === cell 8
model = Sequential()



## === cell 9
model.add(
    Embedding(
        effective_vocab,
        embedding_dim,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 10
model.add(LSTM(75))



## === cell 11
model.add(Dense(128, activation="relu"))



## === cell 12
model.add(Dense(5, activation="softmax"))



## === cell 13
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 14
batch_size = 128
shuffle_seed = 42

options = tf.data.Options()
try:
    options.deterministic = True
except Exception:
    pass

SHUFFLE_BUFFER = 20000

train_ds = tf.data.Dataset.from_tensor_slices((x_train_text, y_train)).with_options(
    options
)
train_ds = train_ds.map(
    lambda phrase, label: (vectorize(phrase), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = (
    train_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=shuffle_seed, reshuffle_each_iteration=False
    )
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_test_text, y_test)).with_options(options)
val_ds = val_ds.map(
    lambda phrase, label: (vectorize(phrase), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
val_ds = (
    val_ds.batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=4,
    validation_data=val_ds,
    verbose=2,
)



## === cell 15
_ = test.head(0)



## === cell 16
pass



## === cell 17
_ = test.head(0)



## === cell 18
testing_phrase = test["Phrase"].astype(str)



## === cell 19
t0 = time.time()
test_text_ds = tf.data.Dataset.from_tensor_slices(testing_phrase.values)
test_ds = (
    test_text_ds.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(4096)
    .prefetch(tf.data.AUTOTUNE)
)
print("Test vectorization pipeline build seconds:", round(time.time() - t0, 3))



## === cell 20
proba = model.predict(test_ds, verbose=0)
predict = np.argmax(proba, axis=1).astype(int)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3712891059.py in <cell line: 0>()
----> 1 proba = model.predict(test_ds, verbose=0)
      2 predict = np.argmax(proba, axis=1).astype(int)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Cannot add tensor to the batch: number of elements does not match. Shapes are: [tensor]: [0], [batch]: [300] [Op:IteratorGetNext] name: 

## === cell 21
predict[0]



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1909701773.py in <cell line: 0>()
----> 1 predict[0]
      2 

NameError: name 'predict' is not defined

## === cell 22
test["Sentiment"] = predict



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3776737316.py in <cell line: 0>()
----> 1 test["Sentiment"] = predict
      2 

NameError: name 'predict' is not defined

## === cell 23
_ = test.head(0)



## === cell 24
submission = test[["PhraseId", "Sentiment"]].copy()
submission = submission.sort_values("PhraseId").reset_index(drop=True)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/804762704.py in <cell line: 0>()
----> 1 submission = test[["PhraseId", "Sentiment"]].copy()
      2 submission = submission.sort_values("PhraseId").reset_index(drop=True)
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Sentiment'] not in index"

## === cell 25
_ = submission.head(0)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1966700352.py in <cell line: 0>()
----> 1 _ = submission.head(0)
      2 

NameError: name 'submission' is not defined

## === cell 26
submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", submission.shape)
print(submission.dtypes)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/258338958.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission to submission.csv with shape:", submission.shape)
      3 print(submission.dtypes)

NameError: name 'submission' is not defined
