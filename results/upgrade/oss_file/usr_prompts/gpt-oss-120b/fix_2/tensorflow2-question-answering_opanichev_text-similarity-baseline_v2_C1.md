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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.1543

# 6. Current score

0.35071

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.35071) has done: 'The fix removes the missing **Levenshtein** import, corrects the tqdm import, and eliminates the artificial `n_samples` limit so the script processes the entire test set. This creates a submission with exactly 2 rows per test example (long + short), matching the required length and producing a valid `submission.csv`. No core modeling logic is changed.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
r_buf = [
    "is",
    "are",
    "do",
    "does",
    "did",
    "was",
    "were",
    "will",
    "can",
    "the",
    "a",
    "what",
    "where",
    "when",
    "which",
]


def clean(x):
    r_buf_local = ["is", "are", "the", "a", "what", "where", "when", "which"]
    x = x.lower()
    for r in r_buf_local:
        x = x.replace(r, "")
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]


def predict(json_data):
    candidates = json_data["long_answer_candidates"]
    doc_tokenized = json_data["document_text"].split(" ")
    question = json_data["question_text"]

    scores = []
    q_tokens = question.split(" ")
    for c in candidates:
        s, e = c["start_token"], c["end_token"]
        score = sum(1 for w in doc_tokenized[s:e] if w in q_tokens)
        scores.append(score)

    ans = candidates[np.argmax(scores)]
    ans_long = f"{ans['start_token']}:{ans['end_token']}"

    if q_tokens[0] in bin_question_tokens:
        ans_short = "YES"
    else:
        ans_short = ""
    return ans_long, ans_short




## === cell 2
ids = []
anns = []
preds = []

n_samples = 5000  # keep a modest subset for quick local validation

with open(
    "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl", "r"
) as json_file:
    cnt = 0
    for line in tqdm(json_file, total=n_samples):
        json_data = json.loads(line)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        l_ann = f"{json_data['annotations'][0]['long_answer']['start_token']}:{json_data['annotations'][0]['long_answer']['end_token']}"
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if len(json_data["annotations"][0]["short_answers"]) > 0:
                s_ann = f"{json_data['annotations'][0]['short_answers'][0]['start_token']}:{json_data['annotations'][0]['short_answers'][0]['end_token']}"
            else:
                s_ann = ""
        else:
            s_ann = json_data["annotations"][0]["yes_no_answer"]

        anns.append(l_ann)
        anns.append(s_ann)

        l_pred, s_pred = predict(json_data)
        preds.append(l_pred)
        preds.append(s_pred)

        cnt += 1
        if cnt >= n_samples:
            break

train_ann = pd.DataFrame(
    {"example_id": ids, "CorrectString": anns, "PredictionString": preds}
)
f1 = f1_score(
    train_ann["CorrectString"].values,
    train_ann["PredictionString"].values,
    average="micro",
)
print(f"F1-score on validation subset: {f1:.4f}")



## === cell 3
ids = []
preds = []

test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"
with open(test_path, "r") as json_file:
    for line in tqdm(json_file):
        json_data = json.loads(line)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        l_pred, s_pred = predict(json_data)
        preds.append(l_pred)
        preds.append(s_pred)

subm = pd.DataFrame({"example_id": ids, "PredictionString": preds})
print(f"Submission shape: {subm.shape}")  # should be (2 * num_test_examples, 2)



## === cell 4
subm.to_csv("submission.csv", index=False)
print("submission.csv written, first rows:")
print(subm.head())
