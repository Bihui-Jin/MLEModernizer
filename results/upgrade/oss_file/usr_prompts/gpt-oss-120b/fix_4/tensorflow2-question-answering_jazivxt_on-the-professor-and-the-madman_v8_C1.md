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

0.00776

# 6. Current score

0.00389

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00306) has done: 'We replace the full loading of the huge test file with a line‑by‑line stream, avoiding the massive DataFrame allocation and costly BeautifulSoup parsing.  By extracting the first paragraph with simple string operations (much faster than BS) and writing each prediction directly to the CSV, memory use stays tiny and the whole pipeline finishes well under the 600 s limit while preserving the exact prediction logic.'
- What this solution (achieved 0.00389) has done: 'I keep the overall pipeline unchanged but improve the heuristics used for generating predictions.  
1. For the long answer I now extract **all** `<p>...</p>` sections and pick the paragraph with the most tokens, which is more likely to contain the true answer than always using the first paragraph.  
2. For yes‑no questions the short answer is no longer random; the presence of the word “not” in the question tips the guess toward “NO”, otherwise “YES”.  
These small, deterministic tweaks should raise the micro‑F1 from 0.00306 toward the target 0.00776 while preserving the original logic and format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import json
import random
import csv




## === cell 1
def read_lines_m(path, max_limit=None):
    """
    Read a newline‑delimited JSON file.
    If max_limit is None, read the whole file; otherwise stop after max_limit lines.
    """
    records = []
    count = 0
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
            count += 1
            if max_limit is not None and count >= max_limit:
                break
    return pd.DataFrame(records)




## === cell 2
p = "../input/tensorflow2-question-answering/"

train = read_lines_m(p + "simplified-nq-train.jsonl", max_limit=4000)

train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)

sub = pd.read_csv(p + "sample_submission.csv")
print("Shapes -> train:", train.shape, "sample_sub:", sub.shape)




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
print("Median long answer length:", np.median(la))
print("Median short answer length:", np.median(sa) if sa else None)




## === cell 4
test_path = p + "simplified-nq-test.jsonl"
output_path = "submission.csv"

yes_no_words = {
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


def select_long_answer(doc_text):
    """
    Extract all <p>...</p> paragraphs and return the token span of the longest one.
    If no paragraph tags are found, fall back to a random span around token 390.
    """
    paras = []
    pos = 0
    while True:
        start_tag = doc_text.find("<p>", pos)
        if start_tag == -1:
            break
        end_tag = doc_text.find("</p>", start_tag)
        if end_tag == -1:
            break
        para_text = doc_text[start_tag + 3 : end_tag]
        paras.append((para_text, start_tag))
        pos = end_tag + 4

    if paras:
        para_text, start_tag = max(paras, key=lambda x: len(x[0].split()))
        prefix = doc_text[:start_tag]
        start_tok = len(prefix.split())
        long_span = f"{start_tok}:{start_tok + len(para_text.split())}"
        return long_span
    else:
        total_tokens = len(doc_text.split())
        start_tok = random.randint(390, max(391, total_tokens - 1))
        return f"{start_tok}:{start_tok + 114}"


def decide_short_answer(question, long_start_tok):
    """
    For yes/no questions, use a simple heuristic:
    - if the word 'not' appears -> NO
    - else -> YES
    For other questions, pick a short span of 2 tokens within the long answer region.
    """
    q_words = set(question.lower().split())
    if yes_no_words.intersection(q_words):
        return "NO" if "not" in q_words else "YES"
    else:
        short_start = random.randint(long_start_tok, long_start_tok + 112)
        return f"{short_start}:{short_start + 2}"


with open(test_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["example_id", "PredictionString"])

    for line in f_in:
        sample = json.loads(line)

        doc_text = sample["document_text"]
        q_text = sample["question_text"]
        example_id = sample["example_id"]

        long_answer = select_long_answer(doc_text)
        long_start = int(long_answer.split(":")[0])

        short_answer = decide_short_answer(q_text, long_start)

        writer.writerow([example_id + "_long", long_answer])
        writer.writerow([example_id + "_short", short_answer])

print("Submission written with", sum(1 for _ in open(output_path)) - 1, "rows.")
