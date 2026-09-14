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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    pass

import os
import json
import random
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as b



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_dir = os.path.abspath(os.path.join(os.getcwd(), "input"))

test_path = os.path.join(base_dir, "simplified-nq-test.jsonl")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

if not os.path.isfile(test_path):
    raise FileNotFoundError(f"Test file not found at expected location: {test_path}")

if os.path.isfile(sample_sub_path):
    _sample_sub = pd.read_csv(sample_sub_path, nrows=5)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2181687131.py in <cell line: 0>()
      8 # Verify that test file exists; raise a clear error if not.
      9 if not os.path.isfile(test_path):
---> 10     raise FileNotFoundError(f"Test file not found at expected location: {test_path}")
     11 
     12 # Load a sample submission to obtain the column order (optional, just for sanity).

FileNotFoundError: Test file not found at expected location: /kaggle/working/input/simplified-nq-test.jsonl

## === cell 2
result_rows = []

with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        example_id = item["example_id"]
        doc_text = item["document_text"]
        question = item["question_text"]

        soup = b(doc_text, "html.parser")
        paragraphs = [
            p.get_text()
            for p in soup.find_all("p", text=True)
            if len(p.get_text()) > 50
        ]

        doc_tokens = doc_text.split()
        doc_len = len(doc_tokens)

        if paragraphs:
            para = paragraphs[0]
            start_char = doc_text.find(para)
            start_token = len(doc_text[:start_char].split())
            end_token = min(start_token + len(para.split()) + 2, doc_len)
            long_answer = f"{start_token}:{end_token}"
        else:
            if doc_len > 390:
                start_token = random.randrange(390, doc_len)
            else:
                start_token = random.randrange(0, doc_len)
            end_token = min(start_token + 114, doc_len)
            long_answer = f"{start_token}:{end_token}"

        if any(
            aux in question.lower().split()
            for aux in [
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
        ):
            short_answer = random.choice(["YES", "NO"])
        else:
            if doc_len > start_token + 2:
                short_start = random.randrange(
                    start_token, min(start_token + 114, doc_len - 2)
                )
            else:
                short_start = max(0, doc_len - 2)
            short_answer = f"{short_start}:{short_start + 2}"

        result_rows.append([f"{example_id}_long", long_answer])
        result_rows.append([f"{example_id}_short", short_answer])

submission_df = pd.DataFrame(result_rows, columns=["example_id", "PredictionString"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, total rows: {len(result_rows)}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2202462967.py in <cell line: 0>()
      2 result_rows = []
      3 
----> 4 with open(test_path, "r", encoding="utf-8") as f:
      5     for line in f:
      6         item = json.loads(line)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/input/simplified-nq-test.jsonl'
