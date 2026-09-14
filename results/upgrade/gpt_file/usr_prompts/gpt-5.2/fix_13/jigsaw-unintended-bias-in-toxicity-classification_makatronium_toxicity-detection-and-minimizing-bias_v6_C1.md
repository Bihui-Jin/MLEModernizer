# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LSTM,
    Bidirectional,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## === cell 1
BASE_INPUT = "../input/jigsaw-unintended-bias-in-toxicity-classification"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing train.csv at {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing test.csv at {TEST_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"




## === cell 2
identity_columns_all = [
    "asian",
    "atheist",
    "bisexual",
    "black",
    "buddhist",
    "christian",
    "female",
    "heterosexual",
    "hindu",
    "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability",
    "jewish",
    "latino",
    "male",
    "muslim",
    "other_disability",
    "other_gender",
    "other_race_or_ethnicity",
    "other_religion",
    "other_sexual_orientation",
    "physical_disability",
    "psychiatric_or_mental_illness",
    "transgender",
    "white",
]
usecols = ["id", "target", "comment_text"] + identity_columns_all

dtype_map = {"id": np.int64, "target": np.float32, "comment_text": str}
for c in identity_columns_all:
    dtype_map[c] = np.float32

train_full = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols,
    dtype=dtype_map,
    engine="c",
    low_memory=False,
)
n_rows = len(train_full)




## === cell 3
identity_columns = identity_columns_all  # all exist in the official train.csv

train_idn = (
    train_full[identity_columns].fillna(0).to_numpy(dtype=np.float32, copy=False)
)
train_tgt = train_full["target"].to_numpy(dtype=np.float32, copy=False)

weights = np.ones((n_rows,), dtype=np.float32) / 4.0
weights += (train_idn >= 0.5).sum(axis=1).astype(bool).astype(np.int32) / 4.0
weights += (
    (
        (train_tgt >= 0.5).astype(np.int32)
        + (train_idn < 0.5).sum(axis=1).astype(bool).astype(np.int32)
    )
    > 1
).astype(np.int32) / 4.0
weights += (
    (
        (train_tgt < 0.5).astype(np.int32)
        + (train_idn >= 0.5).sum(axis=1).astype(bool).astype(np.int32)
    )
    > 1
).astype(np.int32) / 4.0

loss_weight = 1.0 / weights.mean()
train_label_all = np.vstack([(train_tgt >= 0.5).astype(np.int32), weights]).T

del train_idn, train_tgt, weights
gc.collect()




## === cell 4
all_idx = np.arange(n_rows, dtype=np.int64)
tr_idx, va_idx = train_test_split(all_idx, test_size=0.20, random_state=SEED)

y_train = train_label_all[tr_idx]
y_valid = train_label_all[va_idx]

print("Total rows:", n_rows, "Train:", len(tr_idx), "Valid:", len(va_idx))




## === cell 5
from nltk.tokenize import TweetTokenizer

tweet_tok = TweetTokenizer()

_punct = "/-'?!.,#$%'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + "∞θ÷α•à−β∅³π‘₹´°£€\\×™√²—–&"
_trans_tbl = str.maketrans({c: " " for c in _punct})


def preprocess(x: str) -> str:
    x = x.translate(_trans_tbl)
    toks = tweet_tok.tokenize(x)
    for i, t in enumerate(toks):
        if t and t[0] == "'":
            toks[i] = t[1:]
    return " ".join(toks)




## === cell 6
def _preprocess_texts(texts):
    texts = np.asarray(texts, dtype=object)
    out = np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        out[i] = preprocess(t)
    return out


def _preprocess_texts_parallel(texts, chunk_size=150_000, max_workers=None):
    import os
    from concurrent.futures import ProcessPoolExecutor

    texts = np.asarray(texts, dtype=object)
    n = len(texts)
    if n == 0:
        return texts
    if n <= chunk_size:
        return _preprocess_texts(texts)

    if max_workers is None:
        max_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

    chunks = [texts[start : start + chunk_size] for start in range(0, n, chunk_size)]
    outs = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for arr in ex.map(_preprocess_texts, chunks, chunksize=1):
            outs.append(arr)
    return np.concatenate(outs, axis=0)




## === cell 7
MAX_NUM_WORDS = 10000
TARGET_COLUMN = "target"
TEXT_COLUMN = "comment_text"

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

MAX_SEQUENCE_LENGTH = 250


def pad_text(texts, tokenizer_obj):
    seqs = tokenizer_obj.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, dtype="int32")




## === cell 8
train_full[TEXT_COLUMN] = train_full[TEXT_COLUMN].fillna("").astype(str)

X_tr_raw = train_full.loc[tr_idx, TEXT_COLUMN].to_numpy(dtype=object, copy=False)
X_va_raw = train_full.loc[va_idx, TEXT_COLUMN].to_numpy(dtype=object, copy=False)

gc.collect()




## === cell 9
X_tr_text = _preprocess_texts_parallel(X_tr_raw, chunk_size=150_000, max_workers=None)
X_va_text = _preprocess_texts_parallel(X_va_raw, chunk_size=150_000, max_workers=None)

tokenizer.fit_on_texts(X_tr_text)

del X_tr_raw, X_va_raw
gc.collect()




## === cell 10
X_tr_pad = pad_text(X_tr_text, tokenizer)
X_va_pad = pad_text(X_va_text, tokenizer)

BATCH_SIZE = 128

train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (X_tr_pad, y_train.astype(np.float32, copy=False))
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices(
        (X_va_pad, y_valid.astype(np.float32, copy=False))
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 11
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 10

candidate_paths = [
    "./glove.6B.100d.txt",
    os.path.join("../input", "glove6b100dtxt", "glove.6B.100d.txt"),
    os.path.join("../input", "glove-6b-100d", "glove.6B.100d.txt"),
    os.path.join("../input", "glove6b100d", "glove.6B.100d.txt"),
]

EMBEDDINGS_PATH = None
for p in candidate_paths:
    if os.path.exists(p):
        EMBEDDINGS_PATH = p
        break

print("GloVe path:", EMBEDDINGS_PATH)




## === cell 12
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = np.random.normal(
    0.0, 0.05, size=(num_words, EMBEDDINGS_DIMENSION)
).astype(np.float32)

if EMBEDDINGS_PATH is not None:
    print("loading embeddings (filtered to needed vocab)...")

    w2i_needed = {w: i for w, i in word_index.items() if i < num_words}

    num_words_in_embedding = 0
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ", EMBEDDINGS_DIMENSION)
            if len(parts) != EMBEDDINGS_DIMENSION + 1:
                continue
            w = parts[0]
            i = w2i_needed.get(w)
            if i is None:
                continue
            embedding_matrix[i] = np.fromstring(parts[1], sep=" ", dtype=np.float32)
            num_words_in_embedding += 1

    print(
        "Found embeddings for", num_words_in_embedding, "words out of", len(w2i_needed)
    )
    gc.collect()
else:
    print("GloVe not found; using random embeddings (submission will still be valid).")




## === cell 13
model = Sequential()
model.add(
    Embedding(
        num_words,
        EMBEDDINGS_DIMENSION,
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

print("Compiling model...")
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"],
)
print("Compiled model!")




## === cell 14
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_lstm_toxic.weights.h5",
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)




## === cell 15
print("Training model...")
start = time.time()

history = model.fit(
    train_ds,
    epochs=2,  # keep as in provided code
    validation_data=valid_ds,
    verbose=1,
    callbacks=[model_checkpoint_callback],
)

end = time.time()
print("Training duration: {} minutes".format((end - start) / 60.0))




## === cell 16
_ = None




## === cell 17
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", TEXT_COLUMN],
    dtype={"id": np.int64, TEXT_COLUMN: str},
    engine="c",
    low_memory=False,
)
test[TEXT_COLUMN] = test[TEXT_COLUMN].fillna("").astype(str)

test_text_raw = test[TEXT_COLUMN].to_numpy(dtype=object, copy=False)
test_text = _preprocess_texts_parallel(
    test_text_raw, chunk_size=150_000, max_workers=None
)

submission = pd.read_csv(SAMPLE_SUB_PATH)  # columns: id, prediction

test_padded = pad_text(test_text, tokenizer)
pred = model.predict(test_padded, batch_size=1024, verbose=1)[:, 1]

submission["prediction"] = pred.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 18
del (
    y_train,
    y_valid,
    test_padded,
    test,
    embedding_matrix,
    X_tr_text,
    X_va_text,
    X_tr_pad,
    X_va_pad,
    train_ds,
    valid_ds,
    train_label_all,
    tr_idx,
    va_idx,
    train_full,
)
gc.collect()
