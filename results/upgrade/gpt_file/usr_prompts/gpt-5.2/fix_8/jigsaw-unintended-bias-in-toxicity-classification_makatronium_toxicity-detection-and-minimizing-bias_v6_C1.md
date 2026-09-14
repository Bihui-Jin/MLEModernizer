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

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D
from keras.optimizers import Adam
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

dtype_map = {"id": np.int64, "target": np.float32}
for c in identity_columns_all:
    dtype_map[c] = np.float32

train = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols,
    dtype=dtype_map,
)




## === cell 3
train["comment_text"] = train["comment_text"].fillna("").astype(str)




## === cell 4
_ = None




## === cell 5
print(train.shape)




## === cell 6
_ = None




## === cell 7
_ = None




## === cell 8
_ = None




## === cell 9
from nltk.tokenize import TweetTokenizer

tweet_tok = TweetTokenizer()

_punct = "/-'?!.,#$%'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + "∞θ÷α•à−β∅³π‘₹´°£€\\×™√²—–&"
_trans_tbl = str.maketrans({c: " " for c in _punct})


def cleaning_text(x: str) -> str:
    return x.translate(_trans_tbl)


def handle_contractions(x: str):
    return tweet_tok.tokenize(x)


def fix_quote(tokens):
    tokens = [t[1:] if t.startswith("'") else t for t in tokens]
    return " ".join(tokens)


def preprocess(x: str) -> str:
    x = cleaning_text(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x




## === cell 10
import multiprocessing as mp

_MP_CTX = mp.get_context("fork") if hasattr(mp, "get_context") else mp


def _preprocess_worker(text: str) -> str:
    return preprocess(text)


def _unique_preserve_order(texts):
    idx = {}
    uniq = []
    inv = np.empty(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        j = idx.get(t)
        if j is None:
            j = len(uniq)
            idx[t] = j
            uniq.append(t)
        inv[i] = j
    return uniq, inv


def _preprocess_texts_fast(texts):
    texts = np.asarray(texts, dtype=object)

    uniq, inv = _unique_preserve_order(texts)

    n = len(uniq)
    if n == 0:
        return texts

    cpu = os.cpu_count() or 2
    workers = max(1, min(6, cpu - 1))
    if n < 20000 or workers == 1:
        out_uniq = np.empty(n, dtype=object)
        for i in range(n):
            out_uniq[i] = preprocess(uniq[i])
    else:
        chunksize = max(200, min(5000, n // (workers * 8)))
        with _MP_CTX.Pool(processes=workers) as pool:
            out_uniq = list(pool.imap(_preprocess_worker, uniq, chunksize=chunksize))
        out_uniq = np.asarray(out_uniq, dtype=object)

    return out_uniq[inv]




## === cell 11
MAX_NUM_WORDS = 10000
TARGET_COLUMN = "target"
TEXT_COLUMN = "comment_text"

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

MAX_SEQUENCE_LENGTH = 250


def pad_text(texts, tokenizer_obj):
    seqs = tokenizer_obj.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, dtype="int32")




## === cell 12
identity_columns = [c for c in identity_columns_all if c in train.columns]

idn = train[identity_columns].fillna(0).to_numpy(dtype=np.float32, copy=False)
tgt = train[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False)

weights = np.ones((len(train),), dtype=np.float32) / 4.0
weights += (idn >= 0.5).sum(axis=1).astype(bool).astype(np.int32) / 4.0
weights += (
    (
        (tgt >= 0.5).astype(np.int32)
        + (idn < 0.5).sum(axis=1).astype(bool).astype(np.int32)
    )
    > 1
).astype(np.int32) / 4.0
weights += (
    (
        (tgt < 0.5).astype(np.int32)
        + (idn >= 0.5).sum(axis=1).astype(bool).astype(np.int32)
    )
    > 1
).astype(np.int32) / 4.0

loss_weight = 1.0 / weights.mean()
train_label = np.vstack([(tgt >= 0.5).astype(np.int32), weights]).T




## === cell 13
texts_all_raw = train[TEXT_COLUMN].to_numpy(dtype=object, copy=False)

texts_all = _preprocess_texts_fast(texts_all_raw)

X_tr_text, X_va_text, y_train, y_valid = train_test_split(
    texts_all, train_label, test_size=0.20, random_state=SEED
)

tokenizer.fit_on_texts(texts_all)

X_train = pad_text(X_tr_text, tokenizer)
X_valid = pad_text(X_va_text, tokenizer)

del idn, tgt, texts_all_raw, texts_all, X_tr_text, X_va_text
gc.collect()




## === cell 14
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 10
BATCH_SIZE = 128

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




## === cell 15
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = np.random.normal(
    0.0, 0.05, size=(num_words, EMBEDDINGS_DIMENSION)
).astype(np.float32)

if EMBEDDINGS_PATH is not None:
    print("loading embeddings (filtered to needed vocab)...")

    needed = {w for w, i in word_index.items() if i < num_words}

    num_words_in_embedding = 0
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split()
            if len(values) != EMBEDDINGS_DIMENSION + 1:
                continue
            word = values[0]
            if word not in needed:
                continue
            coefs = np.asarray(values[1:], dtype=np.float32)
            i = word_index.get(word)
            if i is not None and i < num_words:
                embedding_matrix[i] = coefs
                num_words_in_embedding += 1

    print("Found embeddings for", num_words_in_embedding, "words out of", len(needed))
    gc.collect()
else:
    print("GloVe not found; using random embeddings (submission will still be valid).")




## === cell 16
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




## === cell 17
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_lstm_toxic.weights.h5",
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)




## === cell 18
print("Training model...")
start = time.time()
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=2,  # keep as in provided code
    validation_data=(X_valid, y_valid),
    verbose=1,
    callbacks=[model_checkpoint_callback],
)
end = time.time()
print("Training duration: {} minutes".format((end - start) / 60.0))




## === cell 19
_ = None




## === cell 20
test = pd.read_csv(TEST_PATH, usecols=["id", TEXT_COLUMN], dtype={"id": np.int64})
test[TEXT_COLUMN] = test[TEXT_COLUMN].fillna("").astype(str)

test_text = test[TEXT_COLUMN].to_numpy(dtype=object, copy=False)
test_text = _preprocess_texts_fast(test_text)

submission = pd.read_csv(SAMPLE_SUB_PATH)  # columns: id, prediction

test_padded = pad_text(test_text, tokenizer)
pred = model.predict(test_padded, batch_size=1024, verbose=1)[:, 1]

submission["prediction"] = pred.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 21
del (
    X_train,
    X_valid,
    y_train,
    y_valid,
    test_padded,
    test,
    train,
    embedding_matrix,
)
gc.collect()
