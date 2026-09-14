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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
_stopwords_fro = frozenset(_stopwords)

_non_allowed_re = re.compile(r"[^a-z']+")
_space_re = re.compile(r"\s+")
_onechar_token_re = re.compile(r"(?:^| )([a-z'])(?: |$)")


def _fast_clean_text(t: str) -> str:
    t = t.lower()
    t = _non_allowed_re.sub(" ", t)
    t = _space_re.sub(" ", t).strip()
    if not t:
        return ""
    toks = t.split(" ")
    toks = [w for w in toks if (len(w) > 1 and w not in _stopwords_fro)]
    return " ".join(toks)


def _clean_series(s: pd.Series) -> np.ndarray:
    arr = s.astype(str).to_numpy()
    return np.fromiter(
        (_fast_clean_text(x) for x in arr), dtype=object, count=arr.shape[0]
    )


all_text = pd.concat(
    [df["comment_text"].astype(str), test["comment_text"].astype(str)],
    axis=0,
    ignore_index=True,
)
all_clean = _clean_series(all_text)

df["clean_comment"] = all_clean[: len(df)]
test["clean_comment"] = all_clean[len(df) :]

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

adapt_bs = 4096
adapt_ds = tf.data.Dataset.from_tensor_slices(x_text_train).batch(adapt_bs)
text_vectorizer.adapt(adapt_ds)

pad_seq = text_vectorizer(x_text_train).numpy().astype(np.int32, copy=False)

vocab_size = min(MAX_FEATURES, len(text_vectorizer.get_vocabulary()))

print(
    "Vocab size (capped):",
    vocab_size,
    "Train padded shape:",
    pad_seq.shape,
    "Labels shape:",
    y.shape,
)



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
n = pad_seq.shape[0]
val_size = int(0.2 * n)
train_size = n - val_size

x_train = pad_seq[:train_size]
y_train = y[:train_size]
x_val = pad_seq[train_size:]
y_val = y[train_size:]

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds.shuffle(
        buffer_size=min(train_size, 50000),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(batch_size)
    .cache()
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=2,
)



## === cell 9
x_text_test = test["clean_comment"].astype(str).to_numpy()
test_pad_seq = text_vectorizer(x_text_test).numpy().astype(np.int32, copy=False)

print("Test padded shape:", test_pad_seq.shape)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_pad_seq)
    .batch(2048)
    .prefetch(tf.data.AUTOTUNE)
)
predict = model.predict(test_ds, verbose=1)

print("Pred shape:", predict.shape)
print("First row:", predict[0])

sample_submission = sample_submission[["id"] + list_classes].copy()
sample_submission[list_classes] = predict.astype(np.float32)

sample_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
