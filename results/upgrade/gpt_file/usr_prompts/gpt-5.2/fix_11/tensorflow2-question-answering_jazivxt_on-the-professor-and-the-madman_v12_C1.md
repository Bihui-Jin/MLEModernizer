# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import sys, subprocess, os

import tensorflow as tf

print("TensorFlow:", tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import json

BASE = "../input/"

TRAIN_PATH = os.path.join(BASE, "simplified-nq-train.jsonl")

TEST_PATH_PRIMARY = os.path.join(BASE, "simplified-nq-test.jsonl")
TEST_PATH_FALLBACK = os.path.join(BASE, "simplified-nq-kaggle-test.jsonl")
TEST_PATH = (
    TEST_PATH_PRIMARY if os.path.exists(TEST_PATH_PRIMARY) else TEST_PATH_FALLBACK
)

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

sub = pd.read_csv(SAMPLE_SUB)

print("Shapes:", train.shape, sub.shape)
print("Resolved test path:", TEST_PATH)
print("Test path exists:", os.path.exists(TEST_PATH))



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

    if best[0] is None or best_score < 0.07:
        return None, None
    return best


def best_candidate_span_from_question(q_text: str, doc_tokens, candidates):
    """
    Fallback selector when paragraph->candidate overlap fails.
    Still selects directly from provided long_answer_candidates (top_level only), using question-word overlap.
    """
    if not candidates:
        return None, None

    q_words = [w for w in q_text.lower().split() if w and w not in STOP]
    q_set = set(q_words)
    if not q_set:
        return None, None

    best = (None, None)
    best_m = 0
    best_len = 10**18

    for c in candidates:
        if not c.get("top_level", False):
            continue
        st = int(c.get("start_token", -1))
        en = int(c.get("end_token", -1))
        if st < 0 or en <= st or en > len(doc_tokens):
            continue

        cand_tokens = doc_tokens[st:en]
        cand_words = [t.lower() for t in cand_tokens if t and t.lower() not in STOP]
        if not cand_words:
            continue

        m = sum(1 for w in cand_words if w in q_set)
        if (m > best_m) or (m == best_m and m > 0 and (en - st) < best_len):
            best_m = m
            best_len = en - st
            best = (st, en)

    if best[0] is None or best_m < 1:
        return None, None
    return best


def best_candidate_span_direct(q_text: str, doc_tokens, candidates):
    """
    Last-resort long answer selection directly from candidates, using a normalized overlap score.
    """
    if not candidates:
        return None, None

    q_words = [w for w in q_text.lower().split() if w and w not in STOP]
    q_set = set(q_words)
    if not q_set:
        return None, None

    best = (None, None)
    best_score = 0.0
    best_len = 10**18

    for c in candidates:
        if not c.get("top_level", False):
            continue
        st = int(c.get("start_token", -1))
        en = int(c.get("end_token", -1))
        if st < 0 or en <= st or en > len(doc_tokens):
            continue

        cand_tokens = doc_tokens[st:en]
        cand_words = [t.lower() for t in cand_tokens if t and t.lower() not in STOP]
        if not cand_words:
            continue

        m = sum(1 for w in cand_words if w in q_set)
        score = m / max(1, len(q_set))
        clen = en - st
        if (score > best_score) or (
            score == best_score and score > 0 and clen < best_len
        ):
            best_score = score
            best_len = clen
            best = (st, en)

    if best[0] is None or best_score < 0.10:
        return None, None
    return best


def _extract_paragraphs(doc_text: str):
    if "<p" in doc_text.lower():
        s = b(doc_text, "html.parser")
        paras = [
            p.get_text(" ", strip=True)
            for p in s.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]
        if paras:
            return paras

    chunks = [c.strip() for c in doc_text.replace("\r", "\n").split("\n") if c.strip()]
    paras = [c for c in chunks if len(c) > 50]
    return paras


def iter_test_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


needed_exids = set()
for sid in sub["example_id"].astype(str).tolist():
    if sid.endswith("_long"):
        needed_exids.add(sid[: -len("_long")])
    elif sid.endswith("_short"):
        needed_exids.add(sid[: -len("_short")])

needed_exids = set(map(str, needed_exids))
print("Needed unique test example_ids:", len(needed_exids))

pred_long = {}
pred_short = {}

seen = 0
for row in iter_test_rows(TEST_PATH):
    exid = str(row.get("example_id", ""))
    if exid not in needed_exids:
        continue

    seen += 1
    doc_text = row.get("document_text", "")
    q_text = row.get("question_text", "")
    candidates = row.get("long_answer_candidates", [])

    paras = _extract_paragraphs(doc_text)
    doc_tokens = doc_text.split()
    doc_len = len(doc_tokens)

    long_answer = ""
    short_answer = ""

    if doc_len > 0:
        best_para = ""
        norm_score = 0.0
        raw_match = 0

        if len(paras) > 0:
            best_para, norm_score, raw_match, para_len = best_para_and_score(
                q_text, paras
            )

        if (raw_match >= 1 and norm_score >= 0.12) or (
            raw_match >= 2 and norm_score >= 0.08
        ):
            span = (None, None)
            if best_para:
                span = best_candidate_span_from_paragraph(
                    best_para, doc_tokens, candidates
                )
            if span[0] is None:
                span = best_candidate_span_from_question(q_text, doc_tokens, candidates)
            if span[0] is None:
                span = best_candidate_span_direct(q_text, doc_tokens, candidates)

            if span[0] is not None:
                long_answer = f"{span[0]}:{span[1]}"

            if best_para and is_strong_yesno_question(q_text):
                short_answer = infer_yesno_from_text(best_para)
        else:
            span = best_candidate_span_direct(q_text, doc_tokens, candidates)
            if span[0] is not None:
                long_answer = f"{span[0]}:{span[1]}"
            short_answer = ""
    else:
        long_answer = ""
        short_answer = ""

    pred_long[exid] = long_answer
    pred_short[exid] = short_answer

    if seen >= len(needed_exids):
        break

print("Collected predictions for example_ids:", seen)

for exid in needed_exids:
    pred_long.setdefault(exid, "")
    pred_short.setdefault(exid, "")

pred_map = {}
for sub_id in sub["example_id"].astype(str).tolist():
    if sub_id.endswith("_long"):
        exid = sub_id[: -len("_long")]
        pred_map[sub_id] = pred_long.get(exid, "")
    elif sub_id.endswith("_short"):
        exid = sub_id[: -len("_short")]
        pred_map[sub_id] = pred_short.get(exid, "")
    else:
        pred_map[sub_id] = ""

sub_out = sub[["example_id"]].copy()
sub_out["PredictionString"] = sub_out["example_id"].map(pred_map).fillna("").astype(str)

assert len(sub_out) == len(
    sub
), f"Submission length {len(sub_out)} != sample {len(sub)}"
assert list(sub_out.columns) == ["example_id", "PredictionString"]

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head(6))
print("Non-empty long preds:", sum(1 for v in pred_long.values() if v))
print("Non-empty short preds:", sum(1 for v in pred_short.values() if v))
