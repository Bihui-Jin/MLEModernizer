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

0.21367

# 6. Current score

0.08429

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08429) has done: 'The timeout is dominated by reading the entire 1.7GB test JSONL into a DataFrame and then running BeautifulSoup over every full document, both of which add large constant overhead and memory pressure. I keep the exact heuristic logic (first long paragraph if present; otherwise random span; YES/NO based on auxiliary verbs; otherwise random short span), but make it streaming: read test line-by-line, compute predictions immediately, and never materialize the full test DataFrame. I also replace the per-example BeautifulSoup parse with an equivalent fast regex extraction of `<p>...</p>` text (still selecting the first paragraph with length > 50), and cache `doc_text.split()` once per example to avoid repeated tokenization. Submission assembly stays identical in semantics and uses the same paths.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import json
from bs4 import BeautifulSoup as b
import random
import re

print("Imports OK")




## === cell 1
def read_lines_m(path, max_limit=None):
    """Read newline-delimited JSON into a DataFrame."""
    rows = []
    with open(path, "r") as f:
        if max_limit is None:
            for line in f:
                rows.append(json.loads(line))
        else:
            ml = int(max_limit)
            for line in f:
                rows.append(json.loads(line))
                ml -= 1
                if ml <= 0:
                    break
    return pd.DataFrame(rows)


p = "../input/tensorflow2-question-answering/"

train = read_lines_m(p + "simplified-nq-train.jsonl", max_limit=4000)
train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)

test_path = p + "simplified-nq-test.jsonl"

sub = pd.read_csv(p + "sample_submission.csv")

print("train/sub shapes:", train.shape, sub.shape)
print("test will be streamed from:", test_path)



## === cell 2
i = min(99, len(train) - 1)
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
print("median long answer length:", np.median(la))
print("median short answer length:", np.median(sa) if len(sa) else None)



## === cell 4
random.seed(0)  # deterministic output (stability); does not change core logic

aux_verbs = {
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
}

_p_re = re.compile(r"<p[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
_tag_re = re.compile(r"<[^>]+>")


def _extract_first_long_paragraph(doc_text, min_chars=50):
    """Return first paragraph text (tags stripped) with length > min_chars, else None."""
    for m in _p_re.finditer(doc_text):
        inner = m.group(1)
        txt = _tag_re.sub(" ", inner)
        txt = (
            txt.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )
        txt = " ".join(txt.split())
        if len(txt) > min_chars:
            return txt
    return None


pred_map = {}

with open(test_path, "r") as f:
    for line in f:
        row = json.loads(line)
        ex_id = str(row["example_id"])
        doc_text = row["document_text"]
        q_text = row["question_text"]

        doc_tokens = doc_text.split()

        para0 = _extract_first_long_paragraph(doc_text, min_chars=50)

        if para0 is not None:
            pos = doc_text.find(para0)
            r = len(doc_text[:pos].split()) - 1
            long_answer = "".join([str(r), ":", str(r + len(para0.split()) + 2)])
        else:
            try:
                r = random.randrange(390, len(doc_tokens))
            except Exception:
                r = 7
            long_answer = "".join([str(r), ":", str(r + 114)])

        if len([q for q in aux_verbs if q in q_text.lower().split()]) > 0:
            short_answer = random.choice(["YES", "NO"])
        else:
            try:
                rr = random.randrange(r, r + 114)
            except ValueError:
                rr = max(0, r)
            short_answer = "".join([str(rr), ":", str(rr + 2)])

        pred_map[ex_id] = (long_answer, short_answer)

out_pred = []
for ex_id in sub["example_id"].astype(str).tolist():
    if ex_id.endswith("_long"):
        base = ex_id[:-5]
        out_pred.append(pred_map.get(base, ("", ""))[0])
    elif ex_id.endswith("_short"):
        base = ex_id[:-6]
        out_pred.append(pred_map.get(base, ("", ""))[1])
    else:
        out_pred.append("")

submission = pd.DataFrame(
    {"example_id": sub["example_id"], "PredictionString": out_pred}
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Non-null predictions:", submission["PredictionString"].notna().sum())
