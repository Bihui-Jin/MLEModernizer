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

0.27571

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27571) has done: 'Your notebook currently fails to yield a Kaggle score mainly because it tries to `pip install protobuf<6` and restart the kernel, but your environment already has TensorFlow 2.18 and protobuf 6.33 installed; forcing protobuf<6 likely break TensorFlow and prevent completion/submission creation. I remove that protobuf downgrading/restart logic so the pipeline runs end-to-end and reliably writes `submission.csv`. To move toward the target score (0.444) with minimal core-logic change, I keep your spaCy-similarity + linear calibrator approach but make it deterministic and safer by (1) using `context` in the similarity text (since your token model uses it and it helps correlation), and (2) guarding against NaN Pearson on edge cases while keeping predictions in [0,1]. These are small, legitimate changes that typically improve Pearson correlation without changing the overall approach.'
- What this solution (achieved 0.27571) has done: 'The crash happens during `import tensorflow as tf` because TensorFlow 2.18.0 is incompatible with the installed `protobuf==6.33.0`, triggering a known protobuf `MessageFactory.GetPrototype` AttributeError at import time. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids the failing C++/upb path. This change is localized to cell 3 only and keeps the model/training logic unchanged. All existing names imported in cell 3 remain available for cell 4 and onward.'
- What this solution (achieved 0.27571) has done: 'The crash happens while importing TensorFlow because `protobuf==6.33.0` is incompatible with TensorFlow 2.18 in this environment, and forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` doesn’t resolve it. The earliest unblocker is to pin protobuf to a TensorFlow-compatible version (<5) before importing TensorFlow. The minimal fix is to add a small runtime pip install in the failing cell *before* the TensorFlow import, then proceed with the existing imports and seed setting unchanged. This keeps the model/training logic intact and only addresses the environment incompatibility causing the import-time `MessageFactory.GetPrototype` error.'

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
import os

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

np.random.seed(42)
tf.random.set_seed(42)


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
            nlp_local = spacy.load(name)
            return nlp_local
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
    anchor_series,
    target_series,
    context_series=None,
    batch_size=256,
    verbose_every=10000,
):
    anchors = anchor_series.fillna("").astype(str).tolist()
    targets = target_series.fillna("").astype(str).tolist()
    if context_series is None:
        contexts = [""] * len(anchors)
    else:
        contexts = context_series.fillna("").astype(str).tolist()

    disable = [
        p
        for p in ("parser", "ner", "lemmatizer", "attribute_ruler", "tagger", "textcat")
        if p in nlp.pipe_names
    ]

    n = len(anchors)
    sims = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        a_batch = [
            f"{a} {c}".strip() for a, c in zip(anchors[start:end], contexts[start:end])
        ]
        t_batch = [
            f"{t} {c}".strip() for t, c in zip(targets[start:end], contexts[start:end])
        ]
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
    context_series=df_train.loc[train_idx, "context"],
    batch_size=256,
    verbose_every=20000,
)
sim_val = _streamed_similarity(
    df_train.loc[val_idx, "anchor"],
    df_train.loc[val_idx, "target"],
    context_series=df_train.loc[val_idx, "context"],
    batch_size=256,
    verbose_every=20000,
)

y_tr = df_train.loc[train_idx, "score"].to_numpy(dtype=np.float32)
y_va = df_train.loc[val_idx, "score"].to_numpy(dtype=np.float32)

A = np.vstack([sim_train, np.ones_like(sim_train)]).T
a, b = np.linalg.lstsq(A, y_tr, rcond=None)[0]

pred_val = np.clip(a * sim_val + b, 0.0, 1.0)

if np.std(pred_val) == 0.0 or np.std(y_va) == 0.0:
    val_corr = 0.0
else:
    val_corr = float(np.corrcoef(pred_val, y_va)[0, 1])

print(f"Calibrator: a={a:.6f}, b={b:.6f}, val Pearson={val_corr:.6f}")



## === cell 16
sim_test = _streamed_similarity(
    df_test["anchor"],
    df_test["target"],
    context_series=df_test["context"],
    batch_size=256,
    verbose_every=20000,
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
