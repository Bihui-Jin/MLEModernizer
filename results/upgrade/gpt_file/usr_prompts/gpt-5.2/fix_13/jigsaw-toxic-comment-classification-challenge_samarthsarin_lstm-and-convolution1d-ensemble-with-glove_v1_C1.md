# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import re
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

INPUT_BASE = "../input/jigsaw-toxic-comment-classification-challenge"
if not os.path.exists(INPUT_BASE):
    INPUT_BASE = "../input"

print("Listing ../input:")
print(os.listdir("../input"))



## === cell 1
import tensorflow as tf

tf.random.set_seed(SEED)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))



## === cell 2
train_path = os.path.join(INPUT_BASE, "train.csv")
test_path = os.path.join(INPUT_BASE, "test.csv")
sub_path = os.path.join(INPUT_BASE, "sample_submission.csv")

usecols_train = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
usecols_test = ["id", "comment_text"]

dtype_train = {
    "toxic": "float32",
    "severe_toxic": "float32",
    "obscene": "float32",
    "threat": "float32",
    "insult": "float32",
    "identity_hate": "float32",
}

df = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=usecols_test)
sample_submission = pd.read_csv(sub_path)

df["comment_text"] = df["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

df.head()



## === cell 3
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype("float32")



## === cell 4
_basic_stopwords = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
}

_token_re = re.compile(r"[a-z']+")


def cleaning(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    words = _token_re.findall(text)
    words = [w for w in words if w not in _basic_stopwords and len(w) > 1]
    return " ".join(words)




## === cell 5
_stopwords = _basic_stopwords
_token_re = re.compile(r"[a-z']+")


def _clean_one(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    toks = _token_re.findall(s)
    toks = [t for t in toks if (len(t) > 1 and t not in _stopwords)]
    return " ".join(toks)


def _clean_list_fast(texts):
    return [_clean_one(t) for t in texts]


cache_dir = "../working" if os.path.exists("../working") else "."
train_clean_cache = os.path.join(cache_dir, "train_clean_comment.npy")
test_clean_cache = os.path.join(cache_dir, "test_clean_comment.npy")

if os.path.exists(train_clean_cache) and os.path.exists(test_clean_cache):
    try:
        df["clean_comment"] = np.load(train_clean_cache, allow_pickle=True).astype(str)
        test["clean_comment"] = np.load(test_clean_cache, allow_pickle=True).astype(str)
    except Exception:
        df["clean_comment"] = _clean_list_fast(df["comment_text"].values)
        test["clean_comment"] = _clean_list_fast(test["comment_text"].values)
else:
    df["clean_comment"] = _clean_list_fast(df["comment_text"].values)
    test["clean_comment"] = _clean_list_fast(test["comment_text"].values)
    try:
        np.save(
            train_clean_cache,
            df["clean_comment"].values.astype(object),
            allow_pickle=True,
        )
        np.save(
            test_clean_cache,
            test["clean_comment"].values.astype(object),
            allow_pickle=True,
        )
    except Exception:
        pass

df[["comment_text", "clean_comment"]].head()



## === cell 6
MAX_FEATURES = 200000  # keep as provided
MAXLEN = 300

text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_mode="int",
    output_sequence_length=MAXLEN,
    standardize=None,  # already cleaned
    split="whitespace",
    pad_to_max_tokens=False,
)

x_text_train = df["clean_comment"].astype(str).to_numpy()

adapt_bs = 16384
adapt_ds = tf.data.Dataset.from_tensor_slices(x_text_train).batch(adapt_bs)
text_vectorizer.adapt(adapt_ds)

vocab_size = min(MAX_FEATURES, len(text_vectorizer.get_vocabulary()))
print("Vocab size (capped):", vocab_size, "Labels shape:", y.shape)



## === cell 7
model = Sequential()
model.add(
    Embedding(input_dim=vocab_size, output_dim=300, input_length=MAXLEN, trainable=True)
)
model.add(LSTM(50))
model.add(Dense(64, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 8
n = len(df)
val_size = int(0.2 * n)
train_size = n - val_size

batch_size = 32

options = tf.data.Options()
options.experimental_deterministic = True

x_train_text = df["clean_comment"].iloc[:train_size].astype(str).to_numpy()
y_train = y[:train_size]
x_val_text = df["clean_comment"].iloc[train_size:].astype(str).to_numpy()
y_val = y[train_size:]

x_text_test = test["clean_comment"].astype(str).to_numpy()

vec_cache_train = os.path.join(
    cache_dir, f"vec_train_len{MAXLEN}_max{MAX_FEATURES}.npy"
)
vec_cache_val = os.path.join(cache_dir, f"vec_val_len{MAXLEN}_max{MAX_FEATURES}.npy")
vec_cache_test = os.path.join(cache_dir, f"vec_test_len{MAXLEN}_max{MAX_FEATURES}.npy")


def _vectorize_to_numpy(texts, cache_path, bs):
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, mmap_mode="r")
            return arr
        except Exception:
            pass
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(bs).with_options(options)
    ds = ds.map(text_vectorizer, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    out = np.empty((len(texts), MAXLEN), dtype=np.int32)
    idx = 0
    for batch in ds:
        b = batch.numpy().astype(np.int32, copy=False)
        out[idx : idx + b.shape[0]] = b
        idx += b.shape[0]
    try:
        np.save(cache_path, out)
    except Exception:
        pass
    return out


x_train_seq = _vectorize_to_numpy(x_train_text, vec_cache_train, bs=8192)
x_val_seq = _vectorize_to_numpy(x_val_text, vec_cache_val, bs=8192)
x_test_seq = _vectorize_to_numpy(x_text_test, vec_cache_test, bs=16384)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_seq, y_train)).with_options(
    options
)
train_ds = train_ds.shuffle(
    buffer_size=min(train_size, 50000),
    seed=SEED,
    reshuffle_each_iteration=True,
)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val_seq, y_val)).with_options(options)
val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=2,
)

test_ds = tf.data.Dataset.from_tensor_slices(x_test_seq).with_options(options)
test_ds = test_ds.batch(4096).prefetch(tf.data.AUTOTUNE)

predict = model.predict(test_ds, verbose=1)

print("Pred shape:", predict.shape)
print("First row:", predict[0])

sample_submission = sample_submission[["id"] + list_classes].copy()
sample_submission[list_classes] = predict.astype(np.float32)
sample_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
