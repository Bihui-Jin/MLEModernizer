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
For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

## Metric
Micro F1. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

## Submission Format
For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```
`
## Data
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.

# 2. Python version

3.8

# 3. Installed packages

beautifulsoup4==4.13.4
geopandas==0.14.4
nltk==3.9.2
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
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        input/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        working/
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
```

-> data/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> data/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> working/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

# 5. Target score

0.09758

# 6. Current score

0.57105

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50184) has done: 'Your code currently produces a submission but it likely be invalid/very low-scoring because it (1) appends “_long/_short” to test example_ids that already don’t include that suffix (creating IDs that don’t match the sample submission), and (2) uses randomness without a fixed seed, making results unstable. I keep your same heuristic core logic (BeautifulSoup paragraph extraction + word overlap + fallback) but fix the ID construction to exactly match `sample_submission.csv` and make predictions deterministic with fixed seeds. I also align the output ordering to the sample submission so every required row is present exactly once, which is essential to get a valid Kaggle score.'
- What this solution (achieved 0.54262) has done: 'Your current score (0.50184) is far above the target (0.09758), so to move *toward* the target we should deliberately reduce performance with the smallest, safest change that preserves the same pipeline and produces a valid submission. The most controlled way is to keep your long-answer heuristic exactly as-is (so the submission stays well-formed), but make short answers blank for all rows; on NQ this typically drops micro-F1 substantially because you lose most short-answer true positives while still keeping long-answer behavior intact. This avoids randomness/instability and doesn’t change model/training (there is none), only the prediction post-processing for the short field. The submission formatting (IDs/order) remains aligned to `sample_submission.csv`.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.54262) is far above the target (0.09758), so the smallest reliable way to move toward the target (without changing the overall heuristic pipeline) is to intentionally reduce recall by outputting blank predictions for both long and short answers. This keeps the same data reading, parsing, determinism, and submission formatting aligned to `sample_submission.csv`, but removes the heuristic “matches” that are driving F1 high. This should pull the micro-F1 down substantially and predictably (often near zero), which is much closer to 0.09758 than 0.54262. The submission remains valid, complete, and deterministic.'
- What this solution (achieved 0.56581) has done: 'Your current score (0.57117) is far above the target (0.09758), so we should deliberately reduce performance with the smallest, safest change that preserves the same pipeline and produces a valid submission. The most controlled way (without changing parsing/heuristic core logic) is to keep generating long/short candidates as you do, but output blank predictions for a fixed fraction of rows to lower recall/precision and bring micro-F1 down toward the target rather than near-zero. To keep results stable, the blanking is done deterministically from the base example_id via a hash, so repeated runs produce identical submissions. The submission remains aligned to `sample_submission.csv` ordering and schema.'
- What this solution (achieved 0.57006) has done: 'Your current score (0.56581) is far above the target (0.09758), so we should deliberately reduce performance in a controlled, deterministic way while keeping the same parsing + heuristic prediction core logic and a valid submission. The smallest reliable knob you already have is `BLANK_RATE`, which blanks both long/short predictions for a fraction of IDs; increasing it reduce recall (and usually F1) predictably. I only adjust `BLANK_RATE` upward (keeping the same stable hash gating) and add a quick printed summary so you can see how many rows are blanked before submitting. Everything else (data reading, BeautifulSoup extraction, overlap heuristic, deterministic mapping to sample_submission order) remains unchanged.'
- What this solution (achieved 0.57105) has done: 'Your current score (0.57006) is far above the target (0.09758), so the correct direction is to *decrease* performance in a controlled, deterministic way while keeping the exact same parsing + heuristic core logic and producing a valid submission. The smallest safe knob is the existing `BLANK_RATE`, which deterministically blanks both long and short predictions for a fraction of base example_ids; increasing it reduces recall (and usually micro-F1) predictably without changing the heuristic itself. I only raise `BLANK_RATE` and keep the stable hash gating and submission alignment identical, so the pipeline stays valid and repeatable. This should move the score closer to the target band with minimal code change.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys

for m in list(sys.modules):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf

print(tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import json


def read_lines_m(path, max_limit=4000):
    rlm = []
    ml = max_limit
    with open(path, "r") as f:
        for l in f:
            rlm.append(json.loads(l))
            ml -= 1
            if ml <= 0:
                break
    return pd.DataFrame(rlm)


p = "../input/tensorflow2-question-answering/"
train = read_lines_m(p + "simplified-nq-train.jsonl")
train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)
test = read_lines_m(p + "simplified-nq-test.jsonl").reset_index(drop=True)
sub = pd.read_csv(p + "sample_submission.csv")
train.shape, test.shape, sub.shape



## === cell 2
i = 99
print("URL:", train.document_url[i])
print(train.question_text[i])
print(train.long_answer_candidates[i][0])
print(
    " ".join(
        train.document_text[i].split()[
            train.annotations[i][0]["long_answer"]["start_token"] : train.annotations[
                i
            ][0]["long_answer"]["end_token"]
        ]
    )
)
if len(train.annotations[i][0]["short_answers"]) > 0:
    print(
        " ".join(
            train.document_text[i].split()[
                train.annotations[i][0]["short_answers"][0][
                    "start_token"
                ] : train.annotations[i][0]["short_answers"][0]["end_token"]
            ]
        )
    )



## === cell 3
la = [
    t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
    for t in train.annotations
]
sa = [
    t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
    for t in train.annotations
    if len(t[0]["short_answers"]) > 0
]
np.median(la), np.median(sa)



## === cell 4
from bs4 import BeautifulSoup as b
from nltk.corpus import stopwords
import random, nltk

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    _sw = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    _sw = stopwords.words("english")
_sw_set = set(_sw)


def qa_word_match(q, a):
    q = q.lower().split()
    q = [q1 for q1 in q if q1 not in _sw_set]
    tm = 0
    a2 = a[0] if len(a) else ""
    for a1 in a:
        m = np.sum([1 for w in a1.lower().split() if w in q])
        if m > tm:
            tm = int(m)
            a2 = str(a1)
    return a2


result = []
for i in range(len(test.example_id)):
    s = b(test.document_text[i], "html.parser")
    p_tags = s.find_all("p", text=True)
    p_txt = [
        pt.get_text() for pt in p_tags if pt.get_text() and len(pt.get_text()) > 50
    ]

    if len(p_txt) > 0:
        a = qa_word_match(test.question_text[i], p_txt)
        pos = test.document_text[i].find(a)
        if pos >= 0:
            r = len(test.document_text[i][:pos].split()) - 1
            r = max(r, 0)
        else:
            r = 0
        long_answer = "".join([str(r), ":", str(r + len(p_txt[0].split()) + 2)])
    else:
        try:
            r = random.randrange(390, len(test.document_text[i].split()))
        except Exception:
            r = 7
        long_answer = "".join([str(r), ":", str(r + 114)])

    short_answer = ""

    result.append([str(test.example_id[i]), long_answer, short_answer])

pred_base = pd.DataFrame(result, columns=["base_example_id", "pred_long", "pred_short"])
pred_base.head()



## === cell 5
import hashlib

sub_out = sub.copy()
sub_out["base_example_id"] = (
    sub_out["example_id"].astype(str).str.replace(r"_(long|short)$", "", regex=True)
)

long_map = dict(zip(pred_base["base_example_id"], pred_base["pred_long"]))
short_map = dict(zip(pred_base["base_example_id"], pred_base["pred_short"]))

BLANK_RATE = 0.995  # was 0.95


def _stable_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return (v % 10_000_000) / 10_000_000.0


def _pick_pred(row):
    base_id = str(row["base_example_id"])
    ex_id = str(row["example_id"])

    if _stable_u01(base_id) < BLANK_RATE:
        return ""

    if ex_id.endswith("_long"):
        return (
            str(long_map.get(base_id, ""))
            if long_map.get(base_id, "") is not None
            else ""
        )
    else:
        return (
            str(short_map.get(base_id, ""))
            if short_map.get(base_id, "") is not None
            else ""
        )


sub_out["PredictionString"] = sub_out.apply(_pick_pred, axis=1)
sub_out = sub_out[["example_id", "PredictionString"]]

sub_out.to_csv("submission.csv", index=False)

print(sub_out.head(10))
print("Wrote submission.csv with shape:", sub_out.shape)
print("Missing predictions (NaN):", sub_out["PredictionString"].isna().sum())
print("Blank predictions:", (sub_out["PredictionString"].astype(str) == "").sum())

_tmp = sub_out.copy()
_tmp["base_example_id"] = (
    _tmp["example_id"].astype(str).str.replace(r"_(long|short)$", "", regex=True)
)
blank_by_base = _tmp.groupby("base_example_id")["PredictionString"].apply(
    lambda s: (s.astype(str) == "").all()
)
print("Base IDs total:", blank_by_base.shape[0])
print("Base IDs fully blanked:", int(blank_by_base.sum()))
print("Base IDs blanked fraction:", float(blank_by_base.mean()))
print("Configured BLANK_RATE:", BLANK_RATE)
