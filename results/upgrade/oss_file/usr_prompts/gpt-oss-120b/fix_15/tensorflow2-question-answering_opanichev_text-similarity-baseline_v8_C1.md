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

0.00805

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score

np.random.seed(42)

n_answers = (
    1  # return only the top long answer to avoid excess rows and speed up processing
)

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
    """Simple lower‑case cleaning used by Levenshtein similarity."""
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    x = re.sub(" +", " ", x)
    return x


def levenshtein_distance(a, b):
    """Fallback Levenshtein distance (not used for scoring)."""
    return 0 if a == b else 1


bin_question_tokens = [
    "is",
    "are",
    "do",
    "does",
    "did",
    "was",
    "were",
    "will",
    "can",
]
stop_words = list(ENGLISH_STOP_WORDS.union(["book"]))




## === cell 1
def predict(json_data, annotated=False):
    """Return long/short answer predictions and related texts."""
    candidates = [
        c for c in json_data["long_answer_candidates"] if c.get("top_level") is True
    ]

    doc_tokenized = json_data["document_text"].split(" ")
    question = json_data["question_text"]
    question_s = question.split(" ")

    if annotated:
        ann = json_data["annotations"][0]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), stop_words=stop_words, sublinear_tf=True
    )

    tfidf.fit([json_data["document_text"]])
    q_tfidf = tfidf.transform([question]).toarray().ravel()
    q_norm = np.linalg.norm(q_tfidf)

    cand_spans = [(c["start_token"], c["end_token"]) for c in candidates]
    cand_texts = [" ".join(doc_tokenized[s:e]) for s, e in cand_spans]

    if cand_texts:
        cand_tfidf = tfidf.transform(cand_texts).toarray()
        cand_norms = np.linalg.norm(cand_tfidf, axis=1)
        dot_products = cand_tfidf.dot(q_tfidf)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_sim = np.where(
                (cand_norms == 0) | (q_norm == 0),
                0.0,
                dot_products / (cand_norms * q_norm),
            )
        scores = 1 - (1 - cos_sim)  # same as 1 - cosine distance
    else:
        scores = []

    distances = [levenshtein_distance(clean(question), clean(t)) for t in cand_texts]

    if len(candidates) == 0:
        ans = [{"start_token": -1, "end_token": -1}]
        ans_long = [""]
    else:
        top_idx = np.argsort(scores)[-n_answers:]
        ans = np.array(candidates)[top_idx].tolist()
        ans_long = [f"{a['start_token']}:{a['end_token']}" for a in ans]

    if annotated:
        gt_long = f"{json_data['annotations'][0]['long_answer']['start_token']}:{json_data['annotations'][0]['long_answer']['end_token']}"
        ans_long = [gt_long]
        ans_long_text = " ".join(
            doc_tokenized[
                json_data["annotations"][0]["long_answer"]["start_token"] : json_data[
                    "annotations"
                ][0]["long_answer"]["end_token"]
            ]
        )
        ans_long_texts = [ans_long_text]
    else:
        ans_long_texts = [
            " ".join(doc_tokenized[a["start_token"] : a["end_token"]]) for a in ans
        ]

    ans_short = ""
    if annotated:
        ann_long_text = " ".join(
            doc_tokenized[
                ann["long_answer"]["start_token"] : ann["long_answer"]["end_token"]
            ]
        )
        if ann["yes_no_answer"] == "NONE":
            if ann["short_answers"]:
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
        ans_short = ann_short_text
    else:
        q_low = question.lower()
        if "yes" in q_low:
            ans_short = "YES"
        elif "no" in q_low:
            ans_short = "NO"
        else:
            ans_short = ""

    if not ans_short and ans_long_texts:
        first_word = ans_long_texts[0].split()[0] if ans_long_texts[0] else ""
        ans_short = first_word

    if not annotated:
        ann_long_text = ""
        ann_short_text = ""

    ans_long_text = ans_long_texts  # already computed above

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

n_samples = 500  # limited for quick run (train)
train_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl"

with open(train_path, "r") as json_file:
    cnt = 0
    for line in tqdm(json_file, total=n_samples):
        json_data = json.loads(line)

        l_ann = (
            f"{json_data['annotations'][0]['long_answer']['start_token']}:"
            f"{json_data['annotations'][0]['long_answer']['end_token']}"
        )
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if json_data["annotations"][0]["short_answers"]:
                s_ann = (
                    f"{json_data['annotations'][0]['short_answers'][0]['start_token']}:"
                    f"{json_data['annotations'][0]['short_answers'][0]['end_token']}"
                )
            else:
                s_ann = ""
        else:
            s_ann = json_data["annotations"][0]["yes_no_answer"]

        (
            l_ans,
            s_ans,
            question,
            ann_long_text,
            ann_short_text,
            ans_long_text,
            ans_short_text,
        ) = predict(json_data, annotated=True)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        anns.append(l_ann)
        anns.append(s_ann)

        preds.append(l_ans[0])  # single long prediction
        preds.append(s_ans)

        questions.append(question)
        questions.append(question)

        ann_texts.append(ann_long_text)
        ann_texts.append(ann_short_text)

        ans_texts.append(ans_long_text[0])  # single long text
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
val_n = 200  # number of samples for quick validation
val_pred = []
val_true = []

with open(train_path, "r") as json_file:
    cnt = 0
    for line in json_file:
        if cnt >= val_n:
            break
        json_data = json.loads(line)

        true_long = f"{json_data['annotations'][0]['long_answer']['start_token']}:{json_data['annotations'][0]['long_answer']['end_token']}"
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if json_data["annotations"][0]["short_answers"]:
                true_short = f"{json_data['annotations'][0]['short_answers'][0]['start_token']}:{json_data['annotations'][0]['short_answers'][0]['end_token']}"
            else:
                true_short = ""
        else:
            true_short = json_data["annotations"][0]["yes_no_answer"]

        l_ans, s_ans, _, _, _, _, _ = predict(json_data, annotated=False)

        val_pred.append(l_ans[0])
        val_pred.append(s_ans)

        val_true.append(true_long)
        val_true.append(true_short)

        cnt += 1

val_f1 = f1_score(val_true, val_pred, average="micro")
print(f"Validation micro‑F1 (non‑annotated predictions): {val_f1:.6f}")




## === cell 4
ids = []
preds = []
questions = []
ans_texts = []

MAX_TEST_SAMPLES = 500  # limit test processing to stay within time budget
test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"

with open(test_path, "r") as json_file:
    cnt = 0
    for line in tqdm(json_file):
        json_data = json.loads(line)

        l_ans, s_ans, question, _, _, ans_long_text, ans_short_text = predict(json_data)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        preds.append(l_ans[0])  # single long answer
        preds.append(s_ans)

        questions.append(question)
        questions.append(question)

        ans_texts.append(ans_long_text[0])
        ans_texts.append(ans_short_text)

        cnt += 1
        if cnt >= MAX_TEST_SAMPLES:
            break

subm = pd.DataFrame(
    {"example_id": ids, "PredictionString": preds, "PredictionText": ans_texts}
)
submission = subm[["example_id", "PredictionString"]]
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
