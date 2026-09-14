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
    try:
        import google.protobuf
        from packaging import version

        pb_ver = version.parse(google.protobuf.__version__)
        if pb_ver.major >= 5:
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
    with open(path, "r") as f:
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
import random
import nltk
from nltk.corpus import stopwords

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP = set(stopwords.words("english"))


def qa_word_match(q, a_list):
    q_words = [w for w in q.lower().split() if w not in STOP]
    best = a_list[0] if len(a_list) else ""
    best_m = -1
    for a in a_list:
        m = sum(1 for w in a.lower().split() if w in q_words)
        if m > best_m:
            best_m = m
            best = str(a)
    return best


random.seed(0)
np.random.seed(0)

result = []

for i in range(len(test.example_id)):
    doc_text = test.document_text.iloc[i]
    q_text = test.question_text.iloc[i]

    s = b(doc_text, "html.parser")
    paras = [p.get_text() for p in s.find_all("p", text=True) if len(p.get_text()) > 50]

    doc_tokens = doc_text.split()
    doc_len = len(doc_tokens)

    if len(paras) > 0:
        best_para = qa_word_match(q_text, paras)
        char_pos = doc_text.find(best_para)
        if char_pos >= 0:
            start_tok = max(0, len(doc_text[:char_pos].split()) - 1)
        else:
            start_tok = 0
        para_len = len(best_para.split())
        end_tok = min(doc_len, start_tok + para_len + 2)
        long_answer = f"{start_tok}:{end_tok}"
    else:
        if doc_len > 500:
            start_tok = random.randrange(390, doc_len)
        else:
            start_tok = min(7, max(0, doc_len - 1))
        end_tok = min(doc_len, start_tok + 114)
        long_answer = f"{start_tok}:{end_tok}"

    aux = set(
        [
            "am",
            "are",
            "can",
            "could",
            "did",
            "do",
            "does",
            "has",
            "have",
            "is",
            "may",
            "should",
            "was",
            "were",
            "will",
        ]
    )
    if any(w in aux for w in q_text.lower().split()):
        short_answer = random.choice(["YES", "NO"])
    else:
        if doc_len > 0:
            r = random.randrange(start_tok, min(doc_len, start_tok + 114))
            short_answer = f"{r}:{min(doc_len, r + 2)}"
        else:
            short_answer = ""

    exid = str(test.example_id.iloc[i])
    result.append([exid + "_long", long_answer])
    result.append([exid + "_short", short_answer])

pred_df = pd.DataFrame(result, columns=["example_id", "PredictionString"])

sub_out = sub[["example_id"]].merge(pred_df, on="example_id", how="left")
sub_out["PredictionString"] = sub_out["PredictionString"].fillna("")

assert len(sub_out) == len(
    sub
), f"Submission length {len(sub_out)} != sample {len(sub)}"
assert set(sub_out.columns) == {"example_id", "PredictionString"}

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head(6))
