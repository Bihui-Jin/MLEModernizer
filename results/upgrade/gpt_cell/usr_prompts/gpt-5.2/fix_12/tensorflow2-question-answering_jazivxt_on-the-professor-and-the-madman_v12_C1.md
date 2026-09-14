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

0.57117

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50185) has done: 'I first fix the submission formatting bug that currently prevents any valid score: the test `example_id` already contains `_long/_short`, but the code appends them again, producing invalid IDs. Then I make prediction generation deterministic (fixed random seeds) so you can reliably iterate toward the target score without run-to-run swings. Finally, I ensure required NLTK stopwords are available (downloading quietly if missing) to avoid runtime errors and guarantee the script always writes a proper `submission.csv` matching the sample submission IDs.'
- What this solution (achieved 0.55672) has done: 'Your current score (0.50185) is far above the target (0.0998), so the smallest way to move toward the target is to legitimately reduce predictive “hits” without breaking submission validity. I keep the same heuristic core logic (paragraph selection + yes/no detection + random fallback) but add a low “answering rate” gate that outputs blank predictions for most examples; this reduce the number of accidental exact token matches and bring F1 down toward the target. I also cache stopwords once (no semantic change) to keep runtime stable within the 600s limit. The submission IDs/format and file writing remain unchanged and valid.'
- What this solution (achieved 0.56837) has done: 'Your current score (0.55672) is far above the target (0.0998), so we should *legitimately reduce* the number of correct exact-match spans by answering fewer questions while keeping the same heuristic core logic and submission validity. I (1) lower the `ANSWER_PROB` gate so more rows are blank (which pushes F1 down toward the target), and (2) fix a small bug where the long answer span length was computed using `p_tags[0]` instead of the actually selected paragraph `a`, which could accidentally increase matches (we want fewer). I also keep the existing ID-handling logic and determinism so the score is stable run-to-run and the CSV remains valid. These are minimal changes that should move the score closer to ~0.10 without changing the overall approach.'
- What this solution (achieved 0.57048) has done: 'Your current score (0.56837) is far above the target (0.0998), so to move *toward* the target with minimal, legitimate change, we should reduce the number of potentially-correct exact-match spans by outputting blanks more often. I only lower the existing `ANSWER_PROB` gating probability (keeping the same heuristic logic, randomness, and submission formatting) so fewer questions are answered at all. This should decrease micro-F1 toward ~0.10 while preserving end-to-end execution and a valid `submission.csv`. No model/architecture/training logic is changed.'
- What this solution (achieved 0.5711) has done: 'Your current score (0.57048) is far above the target (0.0998), so we should legitimately move it downward with the smallest possible change while keeping the same heuristic approach and valid submission format. The most direct lever already in your code is the `ANSWER_PROB` gate that controls how often you attempt any (potentially-correct) span; lowering it further produce more blank predictions and reduce micro-F1. I only adjust `ANSWER_PROB` downward (no architecture/logic changes, same randomness/seeds, same ID handling), which should reduce accidental exact matches and move the score toward the target band. The script still run end-to-end and write a valid `submission.csv` matching the sample submission IDs.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.5711) is far above the target (0.0998), so we should *legitimately reduce* correct exact-match spans with the smallest possible change. The most direct lever in your existing core logic is the `ANSWER_PROB` gate; lowering it output blank predictions for more examples, decreasing micro-F1 toward the target while keeping submission validity unchanged. I only adjust `ANSWER_PROB` downward and keep the same heuristic extraction, randomness/seeds, and ID handling. This should move the score down (closer to 0.10) without changing the approach or risking format errors.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is far above the target (0.0998), so we should legitimately move the score downward with the smallest possible change. The only change I make is to the existing `ANSWER_PROB` gate so the script outputs blank predictions for far more examples, reducing the chance of exact-match spans and lowering micro-F1 toward the target band. I keep the same heuristic extraction logic, randomness/seeds, ID handling, and submission formatting so the pipeline remains stable and valid. The script still run end-to-end and write `submission.csv` with the required columns and IDs.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _pb_ver is not None and (_major(_pb_ver) is not None) and _major(_pb_ver) >= 6:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    for m in list(sys.modules):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

print(tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import json


def read_lines_m(path, max_limit=4000):
    rlm = []
    ml = max_limit
    for l in open(path, "r"):
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

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

random.seed(0)
np.random.seed(0)

ANSWER_PROB = 0.000001

_STOP = set(stopwords.words("english"))


def qa_word_match(q, a):
    q = q.lower().split()
    q = [q1 for q1 in q if q1 not in _STOP]
    tm = 0
    a2 = a[0]
    for a1 in a:
        m = np.sum([1 for w in a1.lower().split() if w in q])
        if m > tm:
            tm = int(m)
            a2 = str(a1)
    return a2


result = []
for i in range(len(test.example_id)):
    base_id = test.example_id[i]

    if random.random() > ANSWER_PROB:
        long_answer = ""
        short_answer = ""
    else:
        s = b(test.document_text[i], "html.parser")
        p_tags = [
            pt.get_text()
            for pt in s.find_all("p", text=True)
            if len(pt.get_text()) > 50
        ]
        if len(p_tags) > 0:
            a = qa_word_match(test.question_text[i], p_tags)
            r = test.document_text[i].find(a)
            r = len(test.document_text[i][:r].split()) - 1

            long_answer = "".join([str(r), ":", str(r + len(a.split()) + 2)])
        else:
            try:
                r = random.randrange(390, len(test.document_text[i].split()))
            except Exception:
                r = 7
            long_answer = "".join([str(r), ":", str(r + 114)])

        if (
            len(
                [
                    q
                    for q in [
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
                    if q in test.question_text[i].lower().split()
                ]
            )
            > 0
        ):
            short_answer = random.choice(["YES", "NO"])
        else:
            r = random.randrange(r, r + 114)
            short_answer = "".join([str(r), ":", str(r + 2)])

    result.append(
        [base_id + "_long", long_answer]
        if not (base_id.endswith("_long") or base_id.endswith("_short"))
        else [base_id.replace("_short", "_long"), long_answer]
    )
    result.append(
        [base_id + "_short", short_answer]
        if not (base_id.endswith("_long") or base_id.endswith("_short"))
        else [base_id.replace("_long", "_short"), short_answer]
    )

pred_df = pd.DataFrame(result, columns=["example_id", "PredictionString"])

out = sub[["example_id"]].merge(pred_df, on="example_id", how="left")
out["PredictionString"] = out["PredictionString"].fillna("")

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print("Missing predictions filled with blanks:", (out["PredictionString"] == "").sum())
print(out.head(6))
