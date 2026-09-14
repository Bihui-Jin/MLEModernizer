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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.36266) has done: 'The timeout is dominated by per-example TF‑IDF fitting and per-candidate dense cosine computations; we keep the same TF‑IDF + cosine selection logic but restructure it to fit once on the document and transform all candidates in one sparse batch, avoiding repeated vectorizer calls and dense conversions. We also eliminate the unused Levenshtein computation (it’s calculated but never used for decisions), and speed up text cleaning with a precompiled regex. Finally, we fix the submission-length error by generating predictions strictly from `sample_submission.csv` order (2000 example_ids) using a map built from the test JSONL, which preserves evaluation semantics while matching Kaggle’s required rows.'
- What this solution (achieved 0.54893) has done: 'Your pipeline likely didn’t yield a Kaggle score because the submission is invalid for this competition: you’re outputting `-1:-1` for “no long answer”, but Kaggle expects a blank string for no prediction (as shown in the required format). I make the smallest change to output blanks for null long answers while keeping your TF‑IDF + cosine candidate selection and thresholding identical. I also ensure short answers are blank whenever the long answer is blank (so you don’t accidentally predict YES/NO with no long span), and keep the submission row order exactly matching `sample_submission.csv`. These changes should turn “Not yielded” into a valid submission and move the score toward your target by restoring correct evaluation semantics.'
- What this solution (achieved 0.5711) has done: 'Your current score (0.54893) is far above the target (0.17788), so we should make the smallest change that legitimately lowers performance toward the target without changing the TF‑IDF + cosine core logic. The most direct lever already in your code is the long-answer cosine threshold: raising it makes fewer long answers predicted, which typically reduces micro-F1 on NQ (more blanks). I only adjust `LONG_ANS_MIN_COS` upward and keep everything else identical, including vectorization, candidate selection, and submission formatting/order. This should move the score down toward the target band while preserving valid Kaggle semantics (blank strings for no answer).'
- What this solution (achieved 0.57113) has done: 'Your current score (0.5711) is much higher than the target (0.17788), so to move toward the target we should legitimately *lower* performance with the smallest possible change. The simplest safe lever already in your code is the long-answer cosine threshold: increasing it output more blank long answers, which typically reduces micro-F1 on Natural Questions. I only raise `LONG_ANS_MIN_COS` further and keep the TF‑IDF + cosine candidate selection, short-answer logic, and submission formatting/order identical. This should move the leaderboard score downward toward the target band while preserving a valid `submission.csv`.'
- What this solution (achieved 0.57113) has done: 'Your current score (0.57113) is far above the target (0.17788), so to move closer we should *legitimately reduce* micro-F1 with the smallest possible change. The least invasive lever already present is `LONG_ANS_MIN_COS`: raising it output more blank long answers (and consequently blank short answers), typically lowering F1 while preserving identical TF‑IDF + cosine selection logic and submission semantics. I only increase this threshold and keep everything else (vectorization, candidate selection, formatting, and row order) unchanged. This should move the score downward toward the target tolerance band without risking an invalid submission.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57113) is far above the target (0.17788), so we should make the smallest legitimate change that reduces micro-F1 by predicting fewer non-empty answers. The least invasive lever already in your code is `LONG_ANS_MIN_COS`; increasing it blank more long answers, and your existing logic then also blank short answers, lowering the score while keeping the TF‑IDF + cosine selection core logic identical. I only adjust this single threshold and keep submission formatting/order unchanged to ensure a valid CSV. This should move the score downward toward the target band without altering architecture, training, or feature extraction semantics.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is much higher than the target (0.17788), so we should make the smallest legitimate change that *reduces* micro‑F1 by predicting fewer non-empty answers. The least invasive lever in your existing logic is `LONG_ANS_MIN_COS`: increasing it blank more long answers (and your code already blanks short answers when long is blank), lowering score while preserving the same TF‑IDF + cosine selection pipeline. I only raise this threshold (no other logic changes) and keep submission formatting/order identical to avoid invalid submissions. This should move the leaderboard score downward toward the target band.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is far above the target (0.17788), so the smallest legitimate way to move closer is to predict fewer non-empty answers. Without changing your TF‑IDF + cosine selection core logic or submission semantics, we can do this by raising the existing `LONG_ANS_MIN_COS` threshold so more long answers become blank (and your code already blanks short answers when long is blank). This is a single-parameter change with minimal risk and should reduce micro‑F1 toward the target band. Everything else (vectorization, candidate selection, formatting, and row order) is kept identical to preserve validity and runtime.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import re
import os

from sklearn.metrics import f1_score
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

TRAIN_PATH = "/kaggle/input/simplified-nq-train.jsonl"
TEST_PATH = "/kaggle/input/simplified-nq-test.jsonl"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

print("Found input files (top-level):")
if os.path.exists("/kaggle/input"):
    for name in sorted(os.listdir("/kaggle/input"))[:50]:
        print("/kaggle/input/" + name)
print("TRAIN exists:", os.path.exists(TRAIN_PATH))
print("TEST exists:", os.path.exists(TEST_PATH))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
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

_r_pat = re.compile(
    "|".join(re.escape(r) for r in sorted(set(r_buf), key=len, reverse=True))
)
_space_pat = re.compile(r" +")


def clean(x: str) -> str:
    x = x.lower()
    x = _r_pat.sub("", x)
    x = _space_pat.sub(" ", x)
    return x.strip()


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = list(text.ENGLISH_STOP_WORDS.union(["book"]))


def _cosine_sim_sparse_row(q_vec, t_mat):
    q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
    if q_norm == 0:
        return np.zeros(t_mat.shape[0], dtype=np.float32)

    t_norms = np.sqrt(t_mat.multiply(t_mat).sum(axis=1)).A1
    dots = (t_mat @ q_vec.T).toarray().ravel()

    denom = t_norms * float(q_norm)
    out = np.zeros_like(dots, dtype=np.float32)
    nz = denom > 0
    out[nz] = (dots[nz] / denom[nz]).astype(np.float32, copy=False)
    out[~np.isfinite(out)] = 0.0
    return out


LONG_ANS_MIN_COS = 0.35


def predict(json_data, annotated=False):
    candidates = [
        c for c in json_data["long_answer_candidates"] if c.get("top_level") is True
    ]
    doc_text = json_data["document_text"]
    doc_tokenized = doc_text.split(" ")
    question = json_data["question_text"]
    question_s = question.split(" ")

    if annotated:
        ann = json_data["annotations"][0]

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)

    if candidates:
        cand_texts = []
        starts = np.empty(len(candidates), dtype=np.int32)
        ends = np.empty(len(candidates), dtype=np.int32)
        for i, c in enumerate(candidates):
            s, e = c["start_token"], c["end_token"]
            starts[i] = s
            ends[i] = e
            cand_texts.append(" ".join(doc_tokenized[s:e]))

        tfidf.fit(cand_texts + [question])

        q_tfidf = tfidf.transform([question])  # sparse (1, n_features)
        t_tfidf = tfidf.transform(cand_texts)  # sparse (n_candidates, n_features)

        scores = _cosine_sim_sparse_row(q_tfidf, t_tfidf)
        best_i = int(np.argmax(scores)) if scores.size else -1
        best_score = float(scores[best_i]) if best_i >= 0 else 0.0

        ans = (
            candidates[best_i] if best_i >= 0 else {"start_token": -1, "end_token": -1}
        )
    else:
        tfidf.fit([question] if question.strip() else ["empty"])
        scores = np.array([], dtype=np.float32)
        best_score = 0.0
        ans = {"start_token": -1, "end_token": -1}

    if (len(scores) == 0) or (best_score < LONG_ANS_MIN_COS):
        ans_long = ""
    else:
        ans_long = f"{ans['start_token']}:{ans['end_token']}"

    if (
        len(question_s) > 0
        and question_s[0].lower() in bin_question_tokens
        and ans_long != ""
    ):
        ans_short = "YES"
    else:
        ans_short = ""

    if annotated:
        ann_long_text = " ".join(
            doc_tokenized[
                ann["long_answer"]["start_token"] : ann["long_answer"]["end_token"]
            ]
        )
        if ann["yes_no_answer"] == "NONE":
            if len(ann["short_answers"]) > 0:
                sa0 = ann["short_answers"][0]
                ann_short_text = " ".join(
                    doc_tokenized[sa0["start_token"] : sa0["end_token"]]
                )
            else:
                ann_short_text = ""
        else:
            ann_short_text = ann["yes_no_answer"]
    else:
        ann_long_text = ""
        ann_short_text = ""

    ans_long_text = (
        ""
        if ans["start_token"] < 0
        else " ".join(doc_tokenized[ans["start_token"] : ans["end_token"]])
    )
    if len(ans_short) > 0 or ans_short == "YES":
        ans_short_text = ans_short
    else:
        ans_short_text = ""

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

n_samples = 50  # keep small to avoid long runtime / memory issues on 15.7GB train

if os.path.exists(TRAIN_PATH):
    with open(TRAIN_PATH, "r") as json_file:
        cnt = 0
        for line in tqdm(json_file, total=n_samples):
            json_data = json.loads(line)

            ids.append(str(json_data["example_id"]) + "_long")
            ids.append(str(json_data["example_id"]) + "_short")

            l_ans = (
                str(json_data["annotations"][0]["long_answer"]["start_token"])
                + ":"
                + str(json_data["annotations"][0]["long_answer"]["end_token"])
            )
            if json_data["annotations"][0]["yes_no_answer"] == "NONE":
                if len(json_data["annotations"][0]["short_answers"]) > 0:
                    s_ans = (
                        str(
                            json_data["annotations"][0]["short_answers"][0][
                                "start_token"
                            ]
                        )
                        + ":"
                        + str(
                            json_data["annotations"][0]["short_answers"][0]["end_token"]
                        )
                    )
                else:
                    s_ans = ""
            else:
                s_ans = json_data["annotations"][0]["yes_no_answer"]

            anns.append(l_ans)
            anns.append(s_ans)

            (
                l_pred,
                s_pred,
                question,
                ann_long_text,
                ann_short_text,
                ans_long_text,
                ans_short_text,
            ) = predict(json_data, annotated=True)

            preds.append(l_pred)
            preds.append(s_pred)
            questions.append(question)
            questions.append(question)
            ann_texts.append(ann_long_text)
            ann_texts.append(ann_short_text)
            ans_texts.append(ans_long_text)
            ans_texts.append(ans_short_text)

            cnt += 1
            if cnt >= n_samples:
                break

    train_ann = pd.DataFrame()
    train_ann["example_id"] = ids
    train_ann["question"] = questions
    train_ann["CorrectString"] = anns
    train_ann["CorrectText"] = ann_texts
    train_ann["PredictionString"] = preds
    train_ann["PredictionText"] = ans_texts
    train_ann.to_csv("train_data.csv", index=False)

    try:
        f1 = f1_score(
            train_ann["CorrectString"].values,
            train_ann["PredictionString"].values,
            average="micro",
        )
        print(
            f"Smoke-test micro-F1 on {n_samples} train examples (not Kaggle metric): {f1:.4f}"
        )
    except Exception as e:
        print("Could not compute smoke-test f1:", repr(e))

    print(train_ann.head(6).to_string(index=False))
else:
    print("Train file not found; skipping train smoke test.")




## === cell 3
assert os.path.exists(TEST_PATH), f"Missing test file at {TEST_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"

pred_map = (
    {}
)  # base_example_id -> (long_pred, short_pred, question, long_text, short_text)

with open(TEST_PATH, "r") as json_file:
    for line in tqdm(json_file):
        json_data = json.loads(line)
        ex_id = str(json_data["example_id"])

        (
            l_ans,
            s_ans,
            question,
            _ann_long_text,
            _ann_short_text,
            ans_long_text,
            ans_short_text,
        ) = predict(json_data)

        pred_map[ex_id] = (l_ans, s_ans, question, ans_long_text, ans_short_text)

sample = pd.read_csv(SAMPLE_SUB_PATH)
if "example_id" not in sample.columns or "PredictionString" not in sample.columns:
    raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")

out_pred = np.empty(len(sample), dtype=object)

debug_rows = {
    "example_id": sample["example_id"].astype(str).tolist(),
    "question": [],
    "PredictionString": [],
    "PredictionText": [],
}

for i, full_id in enumerate(sample["example_id"].astype(str).values):
    base_id, kind = full_id.rsplit("_", 1)  # kind in {"long","short"}
    if base_id in pred_map:
        l_ans, s_ans, q, long_text, short_text = pred_map[base_id]
        if kind == "long":
            out_pred[i] = l_ans
            debug_rows["PredictionText"].append(long_text)
        else:
            out_pred[i] = s_ans if l_ans != "" else ""
            debug_rows["PredictionText"].append(short_text if l_ans != "" else "")
        debug_rows["question"].append(q)
        debug_rows["PredictionString"].append(out_pred[i])
    else:
        out_pred[i] = ""
        debug_rows["question"].append("")
        debug_rows["PredictionString"].append("")
        debug_rows["PredictionText"].append("")

subm = pd.DataFrame(
    {"example_id": sample["example_id"].astype(str), "PredictionString": out_pred}
)
subm["PredictionString"] = subm["PredictionString"].fillna("").astype(str)

if len(subm) != len(sample):
    raise ValueError(f"Row count mismatch: subm={len(subm)} sample={len(sample)}")
if not (subm["example_id"].values == sample["example_id"].astype(str).values).all():
    raise ValueError(
        "example_id order mismatch vs sample_submission.csv (will invalidate submission)."
    )

debug = pd.DataFrame(debug_rows)
debug.to_csv("test_data.csv", index=False)

subm.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", subm.shape)
print(subm.head(10).to_string(index=False))
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Using LONG_ANS_MIN_COS =", LONG_ANS_MIN_COS)
print(
    "Non-empty long predictions:",
    int(
        (subm[subm["example_id"].str.endswith("_long")]["PredictionString"] != "").sum()
    ),
)
print(
    "Non-empty short predictions:",
    int(
        (
            subm[subm["example_id"].str.endswith("_short")]["PredictionString"] != ""
        ).sum()
    ),
)
