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

0.0998

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys, subprocess, os


def _ensure_protobuf_compatible():
    """
    Minimal robustness fix: avoid hard dependency on 'packaging'.
    If protobuf major version is >=5, pin to 4.25.3 (commonly compatible with TF Hub setups).
    This change is only to ensure the code runs to completion and produces submission.csv.
    """
    try:
        import google.protobuf

        try:
            from packaging import version  # may not exist in this environment

            pb_ver = version.parse(google.protobuf.__version__)
            major = pb_ver.major
        except Exception:
            v = str(getattr(google.protobuf, "__version__", "0.0.0"))
            major = int(v.split(".")[0]) if v and v[0].isdigit() else 0

        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print("Warning: protobuf compatibility check failed:", repr(e))


_ensure_protobuf_compatible()

import tensorflow as tf

print("TensorFlow:", tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import json

BASE = "../input/"

TRAIN_PATH = os.path.join(BASE, "simplified-nq-train.jsonl")
TEST_PATH = os.path.join(BASE, "simplified-nq-test.jsonl")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")


def read_jsonl(path, max_limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if max_limit is not None and (i + 1) >= max_limit:
                break
    return pd.DataFrame(rows)


train = read_jsonl(TRAIN_PATH, max_limit=4000)
train["D"] = [
    (
        ann[0]["long_answer"]["start_token"]
        if isinstance(ann, list) and len(ann) > 0
        else -1
    )
    for ann in train.annotations
]
train = train[train["D"] > -1].reset_index(drop=True)

test = read_jsonl(TEST_PATH, max_limit=None).reset_index(drop=True)
sub = pd.read_csv(SAMPLE_SUB)

print("Shapes:", train.shape, test.shape, sub.shape)



## === cell 2
i = 99 if len(train) > 100 else 0
print("URL:", train.document_url.iloc[i])
print(train.question_text.iloc[i])
print(train.long_answer_candidates.iloc[i][0])
doc_tokens = train.document_text.iloc[i].split()
st = train.annotations.iloc[i][0]["long_answer"]["start_token"]
en = train.annotations.iloc[i][0]["long_answer"]["end_token"]
print("LONG:", " ".join(doc_tokens[st:en])[:300], "...")
if len(train.annotations.iloc[i][0].get("short_answers", [])) > 0:
    sst = train.annotations.iloc[i][0]["short_answers"][0]["start_token"]
    sen = train.annotations.iloc[i][0]["short_answers"][0]["end_token"]
    print("SHORT:", " ".join(doc_tokens[sst:sen]))



## === cell 3
la = [
    t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
    for t in train.annotations
]
sa = [
    t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
    for t in train.annotations
    if len(t[0].get("short_answers", [])) > 0
]
print(
    "Median long len:",
    np.median(la),
    "Median short len:",
    (np.median(sa) if len(sa) else None),
)



## === cell 4
from bs4 import BeautifulSoup as b
import nltk
from nltk.corpus import stopwords

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP = set(stopwords.words("english"))

np.random.seed(0)


def best_para_and_score(q, a_list):
    q_words = [w for w in q.lower().split() if w not in STOP]
    q_set = set(q_words)
    best = a_list[0] if len(a_list) else ""
    best_m = -1
    best_len = 0
    for a in a_list:
        a_words = a.lower().split()
        m = sum(1 for w in a_words if w in q_set)
        if m > best_m:
            best_m = m
            best = str(a)
            best_len = len(a_words)
    denom = max(1, len(q_set))
    score = best_m / denom
    return best, score, best_m, best_len


YESNO_PREFIX = {
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "must",
    "am",
}


def is_strong_yesno_question(q_text: str) -> bool:
    q = q_text.strip().lower()
    if not q:
        return False
    first = q.split()[0]
    if first not in YESNO_PREFIX:
        return False
    return ("?" in q_text) or (len(q.split()) <= 12)


def infer_yesno_from_text(text: str):
    t = " " + " ".join(text.lower().split()) + " "
    has_yes = (
        (" yes " in t)
        or (" yes," in t)
        or (" yes." in t)
        or (" yes;" in t)
        or (" yes:" in t)
    )
    has_no = (
        (" no " in t)
        or (" no," in t)
        or (" no." in t)
        or (" no;" in t)
        or (" no:" in t)
    )
    if has_yes and not has_no:
        return "YES"
    if has_no and not has_yes:
        return "NO"
    return ""


def best_candidate_span_from_paragraph(best_para_text: str, doc_tokens, candidates):
    """
    Returns (start_token, end_token) for the best candidate whose token text overlaps the paragraph.
    If no good match, returns (None, None).
    """
    if not candidates:
        return None, None

    para_tokens = best_para_text.split()
    if len(para_tokens) < 5:
        return None, None

    para_set = set(t.lower() for t in para_tokens if t and t.lower() not in STOP)
    if not para_set:
        return None, None

    best = (None, None)
    best_score = 0.0

    for c in candidates:
        if not c.get("top_level", False):
            continue
        st = int(c.get("start_token", -1))
        en = int(c.get("end_token", -1))
        if st < 0 or en <= st or en > len(doc_tokens):
            continue

        cand_tokens = doc_tokens[st:en]
        cand_set = set(t.lower() for t in cand_tokens if t and t.lower() not in STOP)
        if not cand_set:
            continue

        inter = len(para_set & cand_set)
        score = inter / max(1, len(para_set))
        if score > best_score:
            best_score = score
            best = (st, en)

    if best[0] is None or best_score < 0.15:
        return None, None
    return best


test_idx = {str(eid): i for i, eid in enumerate(test.example_id.tolist())}

pred_map = {}

for exid, row_i in test_idx.items():
    doc_text = test.document_text.iloc[row_i]
    q_text = test.question_text.iloc[row_i]

    s = b(doc_text, "html.parser")
    paras = [
        p.get_text(" ", strip=True)
        for p in s.find_all("p")
        if len(p.get_text(strip=True)) > 50
    ]

    doc_tokens = doc_text.split()
    doc_len = len(doc_tokens)

    long_answer = ""
    short_answer = ""

    if len(paras) > 0 and doc_len > 0:
        best_para, norm_score, raw_match, para_len = best_para_and_score(q_text, paras)

        if (raw_match >= 2) and (norm_score >= 0.25):
            candidates = test.long_answer_candidates.iloc[row_i]
            span = best_candidate_span_from_paragraph(best_para, doc_tokens, candidates)
            if span[0] is not None:
                long_answer = f"{span[0]}:{span[1]}"
            else:
                long_answer = ""

            if is_strong_yesno_question(q_text):
                short_answer = infer_yesno_from_text(best_para)
            else:
                short_answer = ""
        else:
            long_answer = ""
            short_answer = ""
    else:
        long_answer = ""
        short_answer = ""

    pred_map[exid + "_long"] = long_answer
    pred_map[exid + "_short"] = short_answer


sub_out = sub[["example_id"]].copy()

sub_out["PredictionString"] = sub_out["example_id"].map(pred_map).fillna("").astype(str)

assert len(sub_out) == len(
    sub
), f"Submission length {len(sub_out)} != sample {len(sub)}"
assert list(sub_out.columns) == ["example_id", "PredictionString"]

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head(6))
