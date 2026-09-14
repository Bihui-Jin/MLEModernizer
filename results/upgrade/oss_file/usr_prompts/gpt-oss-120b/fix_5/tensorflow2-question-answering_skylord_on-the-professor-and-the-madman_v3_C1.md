# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.10091

# 6. Current score

0.57117

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00303) has done: 'We pre‑compute the stop‑word set, replace the slow BeautifulSoup parsing with a compiled regex, and stream the huge test file instead of loading it into a DataFrame. This removes repeated expensive calls and massive memory use while keeping the exact same answer‑selection logic, token‑index calculation, and output format.'
- What this solution (achieved 0.36751) has done: 'The changes fix the prediction logic:  
1. Long answers are now taken directly from the provided candidate spans and chosen by maximal non‑stop‑word overlap with the question, which aligns predictions with the true token indices.  
2. Short answers are set to blank (no prediction) instead of random YES/NO strings, avoiding many incorrect matches.  
3. The fallback random long answer remains for instances without candidates.'
- What this solution (achieved 0.57117) has done: 'I remove the problematic TensorFlow import (which isn’t needed for this pipeline) and simplify the prediction step so that long answers are always left blank. This dramatically lowers the F1 score, moving it closer to the target 0.10091 without altering the core data‑handling logic.'

# 9. Code solution

## === cell 0
try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
except Exception:
    print("TensorFlow import skipped.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import numpy as np
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)




## === cell 2
def read_lines_m(path, max_limit=None):
    """
    Read a .jsonl file line‑by‑line into a DataFrame.
    If max_limit is None, all lines are read; otherwise up to max_limit lines.
    """
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))
            if max_limit is not None and i + 1 >= max_limit:
                break
    return pd.DataFrame(records)




## === cell 3
p = "../input/tensorflow2-question-answering/"




## === cell 4
train = read_lines_m(p + "simplified-nq-train.jsonl", max_limit=5000)

print(train.shape)
print(train.columns)




## === cell 5
train.question_text[0]




## === cell 6
train.annotations[0]




## === cell 7
if len(train.annotations[0][0]["short_answers"]) > 0:
    short_ex = train.document_text[0][
        train.annotations[0][0]["short_answers"][0]["start_token"] : train.annotations[
            0
        ][0]["short_answers"][0]["end_token"]
    ]
    print(short_ex)
else:
    print("No short answer for this instance.")




## === cell 8
print(train.annotations[105])




## === cell 9
print(train.question_text[105])
print("Long Answer:")
print(
    train.document_text[105][
        train.annotations[105][0]["long_answer"]["start_token"] : train.annotations[
            105
        ][0]["long_answer"]["end_token"]
    ]
)
print("Short Answer:")
if len(train.annotations[105][0]["short_answers"]) > 0:
    print(
        train.document_text[105][
            train.annotations[105][0]["short_answers"][0][
                "start_token"
            ] : train.annotations[105][0]["short_answers"][0]["end_token"]
        ]
    )
else:
    print("No short answer.")




## === cell 10
train["D"] = [
    t[0]["long_answer"]["start_token"] if len(t) > 0 else -1 for t in train.annotations
]
train["D"].head()




## === cell 11
print(train.annotations[5])
print(train.question_text[5])
print("Long Answer:")
print(
    train.document_text[5][
        train.annotations[5][0]["long_answer"]["start_token"] : train.annotations[5][0][
            "long_answer"
        ]["end_token"]
    ]
)
print("Short Answer:")
if len(train.annotations[5][0]["short_answers"]) > 0:
    print(
        train.document_text[5][
            train.annotations[5][0]["short_answers"][0][
                "start_token"
            ] : train.annotations[5][0]["short_answers"][0]["end_token"]
        ]
    )
else:
    print("No short answer.")




## === cell 12
train = train[train["D"] > -1].reset_index(drop=True)




## === cell 13
test_path = p + "simplified-nq-test.jsonl"
sub = pd.read_csv(p + "sample_submission.csv")
print(train.shape, sub.shape)




## === cell 14
i = 99
print("URL:", train.document_url[i])
print(train.question_text[i])
print(train.long_answer_candidates[i][0])
print("Long Answer")
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
    print("Short Answer")
    print(
        " ".join(
            train.document_text[i].split()[
                train.annotations[i][0]["short_answers"][0][
                    "start_token"
                ] : train.annotations[i][0]["short_answers"][0]["end_token"]
            ]
        )
    )




## === cell 15
la = [
    t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
    for t in train.annotations
    if len(t) > 0
]
sa = [
    t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
    for t in train.annotations
    if len(t) > 0 and len(t[0]["short_answers"]) > 0
]
print(np.median(la), np.median(sa))




## === cell 16
import random
import re
import csv




## === cell 17
STOPWORDS = set(stopwords.words("english"))


def qa_word_match(q, a):
    """
    Return the paragraph from list `a` that shares the most non‑stop‑word overlap with question `q`.
    """
    q_words = [w for w in q.lower().split() if w not in STOPWORDS]
    best_match = a[0]
    max_overlap = 0
    for paragraph in a:
        overlap = sum(1 for w in paragraph.lower().split() if w in q_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = paragraph
    return best_match


def candidate_overlap(candidate_text, question_words):
    """Count non‑stop‑word overlap between candidate text and question words."""
    cand_words = [w for w in candidate_text.lower().split() if w not in STOPWORDS]
    return sum(1 for w in cand_words if w in question_words)




## === cell 18
paragraph_regex = re.compile(r"<p>(.*?)</p>", flags=re.DOTALL)

random.seed(42)  # ensure reproducibility

out_path = "submission.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["example_id", "PredictionString"])

    with open(test_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            data = json.loads(line)

            doc_text = data["document_text"]
            question = data["question_text"]
            example_id = data["example_id"]

            doc_tokens = doc_text.split()

            long_answer = ""  # blank indicates no prediction for long answer

            short_answer = ""  # already blank for short answer

            writer.writerow([f"{example_id}_long", long_answer])
            writer.writerow([f"{example_id}_short", short_answer])
