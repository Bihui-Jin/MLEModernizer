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
scipy==1.15.3
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

0.17788

# 6. Current score

0.57115

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.34986) has done: 'I lower the confidence threshold so the model makes a long‑answer prediction more often, and I output an empty string for “no answer” instead of the placeholder “-1:-1”. These two small tweaks keep the core logic unchanged while ensuring the submission format matches the competition’s expectations, which should move the micro‑F1 score toward the target value.'
- What this solution (achieved 0.54472) has done: 'I raise the confidence SCORE_THRESHOLD and stop automatically predicting a “YES” short answer. These tiny tweaks make the model output blank answers more often and remove a systematic false‑positive short answer, which lower the micro‑F1 score from 0.3499 toward the target of ~0.178 while keeping the core logic unchanged.'
- What this solution (achieved 0.57115) has done: 'I lower the confidence SCORE_THRESHOLD from 1.0 to a moderate value (0.85). With the original threshold the model never makes a long‑answer prediction, resulting in an F1 of 0. A lower threshold allows a few reasonable predictions, moving the micro‑F1 toward the target 0.17788 while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.metrics import f1_score
from scipy import spatial
import concurrent.futures  # parallel processing


def levenshtein_distance(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()




## === cell 1
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

html_tags = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
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
    "of",
    "in",
    "and",
    "on",
    "what",
    "where",
    "when",
    "which",
] + html_tags


def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    x = re.sub(" +", " ", x)
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = list(ENGLISH_STOP_WORDS.union(["book"]))

SCORE_THRESHOLD = 0.85


def _tf_vector(text_tokens, vocab_set):
    """Return a (counter dict, L2 norm) for tokens that appear in vocab_set."""
    cnt = {}
    for w in text_tokens:
        if w in vocab_set:
            cnt[w] = cnt.get(w, 0) + 1
    norm = np.sqrt(sum(v * v for v in cnt.values()))
    return cnt, norm


def _cosine_sim(q_cnt, q_norm, t_cnt, t_norm):
    if q_norm == 0 or t_norm == 0:
        return 0.0
    dot = sum(q_cnt.get(w, 0) * t_cnt.get(w, 0) for w in q_cnt if w in t_cnt)
    return dot / (q_norm * t_norm)


def predict(json_data, annotated=False):
    candidates = json_data["long_answer_candidates"]
    candidates = [c for c in candidates if c["top_level"] is True]
    doc_tokenized = json_data["document_text"].split()
    vocab_set = set(doc_tokenized)

    question = json_data["question_text"]
    question_tokens = question.split()
    q_cnt, q_norm = _tf_vector(question_tokens, vocab_set)

    if annotated:
        ann = json_data["annotations"][0]

    scores = []
    for c in candidates:
        s, e = c["start_token"], c["end_token"]
        t = " ".join(doc_tokenized[s:e])
        _ = levenshtein_distance(clean(question), clean(t))

        t_cnt, t_norm = _tf_vector(t.split(), vocab_set)
        score = _cosine_sim(q_cnt, q_norm, t_cnt, t_norm)
        scores.append(score)

    if len(scores) == 0:
        ans_long = ""
        ans_short = ""
    else:
        ans_idx = int(np.argmax(scores))
        ans = candidates[ans_idx]
        if np.max(scores) < SCORE_THRESHOLD:
            ans_long = ""  # treat low‑score as “no answer”.
        else:
            ans_long = f"{ans['start_token']}:{ans['end_token']}"
        ans_short = ""

    if annotated:
        ann_long_text = " ".join(
            doc_tokenized[
                ann["long_answer"]["start_token"] : ann["long_answer"]["end_token"]
            ]
        )
        if ann["yes_no_answer"] == "NONE":
            if len(ann["short_answers"]) > 0:
                ann_short_text = " ".join(
                    doc_tokenized[
                        ann["short_answers"][0]["start_token"] : ann["short_answers"][
                            0
                        ]["end_token"]
                    ]
                )
            else:
                ann_short_text = ""
        else:
            ann_short_text = ann["yes_no_answer"]
    else:
        ann_long_text = ""
        ann_short_text = ""

    ans_long_text = (
        " ".join(
            doc_tokenized[int(ans_long.split(":")[0]) : int(ans_long.split(":")[1])]
        )
        if ans_long != ""
        else ""
    )
    ans_short_text = ans_short if ans_short else ""

    return (
        ans_long,
        ans_short,
        question,
        ann_long_text,
        ann_short_text,
        ans_long_text,
        ans_short_text,
    )




## === cell 2
ids = []
anns = []
preds = []
questions = []
ann_texts = []
ans_texts = []

n_samples = 2000

train_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl"

with open(train_path, "r") as json_file:
    cnt = 0
    for line in tqdm(json_file, total=n_samples * 2):
        json_data = json.loads(line)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        l_ans = f"{json_data['annotations'][0]['long_answer']['start_token']}:{json_data['annotations'][0]['long_answer']['end_token']}"
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if len(json_data["annotations"][0]["short_answers"]) > 0:
                s = json_data["annotations"][0]["short_answers"][0]
                s_ans = f"{s['start_token']}:{s['end_token']}"
            else:
                s_ans = ""
        else:
            s_ans = json_data["annotations"][0]["yes_no_answer"]

        anns.append(l_ans)
        anns.append(s_ans)

        (
            l_ans_pred,
            s_ans_pred,
            question,
            ann_long_text,
            ann_short_text,
            ans_long_text,
            ans_short_text,
        ) = predict(json_data, annotated=True)

        preds.append(l_ans_pred)
        preds.append(s_ans_pred)
        questions.append(question)
        questions.append(question)
        ann_texts.append(ann_long_text)
        ann_texts.append(ann_short_text)
        ans_texts.append(ans_long_text)
        ans_texts.append(ans_short_text)

        cnt += 1
        if cnt >= n_samples:
            break

train_ann = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions,
        "CorrectString": anns,
        "CorrectText": ann_texts,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
train_ann.to_csv("train_data.csv", index=False)




## === cell 3
f1 = f1_score(
    train_ann["CorrectString"].values,
    train_ann["PredictionString"].values,
    average="micro",
)
print(f"F1-score on validation sample: {f1:.4f}")




## === cell 4
ids = []
preds = []
questions = []
ans_texts = []

test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"


def _process_line(line: str):
    """Return the data needed for the submission for one test line."""
    json_data = json.loads(line)
    l_ans_pred, s_ans_pred, question, _, _, ans_long_text, ans_short_text = predict(
        json_data
    )
    return (
        f"{json_data['example_id']}_long",
        f"{json_data['example_id']}_short",
        l_ans_pred,
        s_ans_pred,
        question,
        ans_long_text,
        ans_short_text,
    )


max_workers = os.cpu_count() or 1

with open(test_path, "r") as json_file, concurrent.futures.ThreadPoolExecutor(
    max_workers=max_workers
) as executor:
    results = list(executor.map(_process_line, json_file))

for (
    id_long,
    id_short,
    pred_long,
    pred_short,
    q,
    ans_long_text,
    ans_short_text,
) in results:
    ids.append(id_long)
    ids.append(id_short)
    preds.append(pred_long)
    preds.append(pred_short)
    questions.append(q)
    questions.append(q)
    ans_texts.append(ans_long_text)
    ans_texts.append(ans_short_text)

subm = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
subm.to_csv("test_data.csv", index=False)
subm[["example_id", "PredictionString"]].to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
