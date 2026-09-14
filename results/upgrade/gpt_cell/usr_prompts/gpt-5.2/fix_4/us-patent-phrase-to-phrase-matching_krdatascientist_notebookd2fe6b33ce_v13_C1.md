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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

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
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.444

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_sub = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)
df_train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
df_test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")



## === cell 2
df_train.head(3)



## === cell 3
import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
    except Exception:
        major = None

    if major is None or major >= 6:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<6"]
        )
        os.execl(sys.executable, sys.executable, *sys.argv)


_ensure_protobuf_compatible()

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten



## === cell 4
input_text = (
    df_train.anchor + " " + df_train.context + " " + df_train.target
).str.lower()



## === cell 5
token = Tokenizer()



## === cell 6
token.fit_on_texts(input_text)



## === cell 7
vocab_size = len(token.word_index) + 1



## === cell 8
embedding_doc = token.texts_to_sequences(input_text)



## === cell 9
max([len(i) for i in embedding_doc])



## === cell 10
max_length = 20
padded_docs = pad_sequences(embedding_doc, maxlen=max_length, padding="post")



## === cell 11
y_train = pd.get_dummies(df_train.score).values



## === cell 12
e = tf.keras.layers.Embedding(vocab_size, 50, input_length=max_length)



## === cell 13
import spacy


def _load_spacy_model():
    preferred = ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")
    for name in preferred:
        try:
            nlp = spacy.load(name)
            return nlp
        except Exception:
            continue
    return spacy.blank("en")


nlp = _load_spacy_model()
print("Loaded spaCy pipeline:", nlp.meta.get("name", "blank/en"))
print("spaCy pipe names:", nlp.pipe_names)
print(
    "Has vectors:",
    getattr(nlp.vocab, "vectors_length", 0) > 0,
    "vectors_length:",
    getattr(nlp.vocab, "vectors_length", 0),
)




## === cell 14
def _streamed_similarity(
    anchor_series, target_series, batch_size=256, verbose_every=10000
):
    anchors = anchor_series.fillna("").astype(str).tolist()
    targets = target_series.fillna("").astype(str).tolist()

    disable = [
        p
        for p in ("parser", "ner", "lemmatizer", "attribute_ruler", "tagger", "textcat")
        if p in nlp.pipe_names
    ]

    sims = []
    pairs = zip(anchors, targets)
    for i, (a, t) in enumerate(pairs):
        pass  # just to keep structure clear; real batching below

    pairs = zip(anchors, targets)
    for i, (a_doc, t_doc) in enumerate(
        zip(
            nlp.pipe((a for a, _ in pairs), batch_size=batch_size, disable=disable),
            nlp.pipe(
                (t for _, t in zip(anchors, targets)),
                batch_size=batch_size,
                disable=disable,
            ),
        )
    ):
        break

    sims = []
    n = len(anchors)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        a_batch = anchors[start:end]
        t_batch = targets[start:end]
        a_docs = nlp.pipe(a_batch, batch_size=batch_size, disable=disable)
        t_docs = nlp.pipe(t_batch, batch_size=batch_size, disable=disable)
        for j, (d1, d2) in enumerate(zip(a_docs, t_docs), start=start):
            s = float(d1.similarity(d2))
            if verbose_every and (j % verbose_every == 0):
                print(d1.text[:60], "|", d2.text[:60], "=>", s)
            sims.append(s)
    return np.asarray(sims, dtype=np.float32)




## === cell 15
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(df_train)), test_size=0.15, random_state=42
)

sim_train = _streamed_similarity(
    df_train.loc[train_idx, "anchor"],
    df_train.loc[train_idx, "target"],
    batch_size=256,
    verbose_every=20000,
)
sim_val = _streamed_similarity(
    df_train.loc[val_idx, "anchor"],
    df_train.loc[val_idx, "target"],
    batch_size=256,
    verbose_every=20000,
)

y_tr = df_train.loc[train_idx, "score"].to_numpy(dtype=np.float32)
y_va = df_train.loc[val_idx, "score"].to_numpy(dtype=np.float32)

A = np.vstack([sim_train, np.ones_like(sim_train)]).T
a, b = np.linalg.lstsq(A, y_tr, rcond=None)[0]

pred_val = np.clip(a * sim_val + b, 0.0, 1.0)
val_corr = np.corrcoef(pred_val, y_va)[0, 1]
print(f"Calibrator: a={a:.6f}, b={b:.6f}, val Pearson={val_corr:.6f}")



## === cell 16
sim_test = _streamed_similarity(
    df_test["anchor"], df_test["target"], batch_size=256, verbose_every=20000
)
test_pred = np.clip(a * sim_test + b, 0.0, 1.0)

submission = df_test[["id"]].copy()
submission["score"] = test_pred.astype(np.float32)

assert len(submission) == len(
    df_sub
), f"Submission rows {len(submission)} != sample rows {len(df_sub)}"
assert list(submission.columns) == ["id", "score"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("score stats:", submission["score"].describe())
