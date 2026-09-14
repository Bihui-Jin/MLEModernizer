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

0.00572

# 6. Current score

0.00319

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00319) has done: 'I fix the TensorFlow import crash by pinning protobuf to a compatible version at runtime before importing TF (this is a known TF/protobuf mismatch issue). Then I fix the submission length/IDs bug by reading and using the provided `sample_submission.csv` as the authoritative list of required `example_id` rows, ensuring we output exactly the same number of rows in the same format. Finally, I keep your core “random span + yes/no heuristic” logic but make it deterministic and robust to short documents so it always produces valid `start:end` token ranges and never throws. This run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys, subprocess, os


def _ensure_protobuf_compat():
    try:
        import google.protobuf

        ver = getattr(google.protobuf, "__version__", "")
        major = int(ver.split(".")[0]) if ver and ver[0].isdigit() else 999
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility step failed:", repr(e))


_ensure_protobuf_compat()

try:
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
except Exception as e:
    tf = None
    print("TensorFlow import failed (continuing without TF):", repr(e))




## === cell 1
import numpy as np
import pandas as pd
import json
import random


def read_lines_m(path, max_limit=None):
    rows = []
    with open(path, "r") as f:
        for i, l in enumerate(f):
            rows.append(json.loads(l))
            if max_limit is not None and (i + 1) >= max_limit:
                break
    return pd.DataFrame(rows)


BASE = "/kaggle/input/tensorflow2-question-answering/"
TRAIN_PATH = BASE + "simplified-nq-train.jsonl"
TEST_PATH = BASE + "simplified-nq-test.jsonl"
SAMPLE_SUB_PATH = BASE + "sample_submission.csv"

train = read_lines_m(TRAIN_PATH, max_limit=4000)
train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)

test = read_lines_m(TEST_PATH, max_limit=None).reset_index(drop=True)
sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train/test/sub shapes:", train.shape, test.shape, sub.shape)
print("sub columns:", sub.columns.tolist())




## === cell 2
i = min(99, len(train) - 1)
print("URL:", train.document_url[i])
print(train.question_text[i])
print(train.long_answer_candidates[i][0])

doc_tokens = train.document_text[i].split()
ls = train.annotations[i][0]["long_answer"]["start_token"]
le = train.annotations[i][0]["long_answer"]["end_token"]
print(" ".join(doc_tokens[ls:le]))

if len(train.annotations[i][0]["short_answers"]) > 0:
    ss = train.annotations[i][0]["short_answers"][0]["start_token"]
    se = train.annotations[i][0]["short_answers"][0]["end_token"]
    print(" ".join(doc_tokens[ss:se]))




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
print(
    "median long/short:",
    (float(np.median(la)) if la else None),
    (float(np.median(sa)) if sa else None),
)




## === cell 4

random.seed(12345)

YN_TRIGGERS = set(
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


def _safe_randrange(a, b):
    if b is None:
        return int(a)
    if b <= a:
        return int(a)
    return random.randrange(a, b)


def predict_for_row(example_id_full, test_row):
    """
    example_id_full: like '{example_id}_long' or '{example_id}_short'
    test_row: row from test dataframe
    """
    tokens = str(test_row["document_text"]).split()
    n = len(tokens)

    if n < 5:
        if example_id_full.endswith("_long"):
            return ""
        q_words = set(str(test_row["question_text"]).lower().split())
        return (
            random.choice(["YES", "NO"])
            if len(YN_TRIGGERS.intersection(q_words)) > 0
            else ""
        )

    long_len = 114
    max_start = max(1, n - long_len - 1)
    start_long = _safe_randrange(0, max_start)
    end_long = min(n, start_long + long_len)

    if example_id_full.endswith("_long"):
        return f"{start_long}:{end_long}"

    q_words = set(str(test_row["question_text"]).lower().split())
    if len(YN_TRIGGERS.intersection(q_words)) > 0:
        return random.choice(["YES", "NO"])
    else:
        ss = _safe_randrange(start_long, max(start_long + 1, end_long - 2))
        se = min(n, ss + 2)
        if se <= ss:
            return ""
        return f"{ss}:{se}"


test_id_to_idx = {
    str(eid): idx for idx, eid in enumerate(test["example_id"].astype(str).values)
}

preds = []
missing = 0

for eid_full in sub["example_id"].astype(str).values:
    if eid_full.endswith("_long"):
        base_id = eid_full[:-5]
    elif eid_full.endswith("_short"):
        base_id = eid_full[:-6]
    else:
        base_id = eid_full

    idx = test_id_to_idx.get(base_id, None)
    if idx is None:
        missing += 1
        preds.append("")
        continue

    preds.append(predict_for_row(eid_full, test.iloc[idx]))

out = sub.copy()
out["PredictionString"] = preds

print("Rows written:", len(out), "Missing base_ids:", missing)
out.to_csv("submission.csv", index=False)

assert out.shape[0] == sub.shape[0]
assert set(out.columns) == set(["example_id", "PredictionString"])
print(out.head())
