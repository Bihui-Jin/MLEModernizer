# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

np.random.seed(0)

DATA_PATH = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"
SAMPLE_SUB_PATH = "/kaggle/input/tensorflow2-question-answering/sample_submission.csv"



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
    "of",
    "in",
    "and",
    "on",
    "what",
    "where",
    "when",
    "which",
]


def clean(x: str) -> str:
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = sorted(list(text.ENGLISH_STOP_WORDS.union(["book"])))




## === cell 2
def _iter_test_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line:
                yield json.loads(line)


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


n_test = _count_lines(DATA_PATH)

tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)

doc_iter_for_fit = (rec["document_text"] for rec in _iter_test_jsonl(DATA_PATH))
tfidf.fit(doc_iter_for_fit)



## === cell 3
ids = [None] * (2 * n_test)
preds = [None] * (2 * n_test)
questions = [None] * (2 * n_test)
ans_texts = [None] * (2 * n_test)


def predict_fast(json_data):
    candidates = [
        c for c in json_data["long_answer_candidates"] if c["top_level"] == True
    ]
    doc_text = json_data["document_text"]
    doc_tokens = doc_text.split(" ")
    question = json_data["question_text"]
    question_s = question.split(" ")

    q_vec = tfidf.transform([question])  # (1, n_features), CSR
    q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
    if q_norm == 0 or len(candidates) == 0:
        best_idx = 0
    else:
        D = tfidf.transform(doc_tokens)

        dots = np.empty(len(candidates), dtype=np.float64)
        c_norms = np.empty(len(candidates), dtype=np.float64)

        for i, c in enumerate(candidates):
            s, e = c["start_token"], c["end_token"]
            if e <= s:
                dots[i] = 0.0
                c_norms[i] = 0.0
                continue
            c_vec = D[s:e].sum(axis=0)  # (1, n_features), sparse matrix/matrix-like
            dots[i] = float(np.asarray(c_vec @ q_vec.T).ravel()[0])
            c_norms[i] = np.sqrt(c_vec.multiply(c_vec).sum())

        denom = c_norms * float(q_norm)
        sims = np.zeros_like(dots, dtype=np.float64)
        nonzero = denom != 0
        sims[nonzero] = dots[nonzero] / denom[nonzero]
        best_idx = int(np.argmax(sims))

    ans = candidates[best_idx] if candidates else {"start_token": 0, "end_token": 0}
    ans_long = f"{ans['start_token']}:{ans['end_token']}"

    if question_s and question_s[0] in bin_question_tokens:
        ans_short = "YES"
    else:
        ans_short = ""

    ans_long_text = " ".join(doc_tokens[ans["start_token"] : ans["end_token"]])
    ans_short_text = ans_short if (len(ans_short) > 0 or ans_short == "YES") else ""

    return ans_long, ans_short, question, ans_long_text, ans_short_text


write_i = 0
for json_data in tqdm(_iter_test_jsonl(DATA_PATH), total=n_test):
    exid = str(json_data["example_id"])
    ids[write_i] = exid + "_long"
    ids[write_i + 1] = exid + "_short"

    l_ans, s_ans, question, ans_long_text, ans_short_text = predict_fast(json_data)

    preds[write_i] = l_ans
    preds[write_i + 1] = s_ans

    questions[write_i] = question
    questions[write_i + 1] = question

    ans_texts[write_i] = ans_long_text
    ans_texts[write_i + 1] = ans_short_text

    write_i += 2

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

subm.head(10)


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2441825380.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     61[0m     [0mids[0m[0;34m[[0m[0mwrite_i[0m [0;34m+[0m [0;36m1[0m[0;34m][0m [0;34m=[0m [0mexid[0m [0;34m+[0m [0;34m"_short"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m [0;34m[0m[0m
[0;32m---> 63[0;31m     [0ml_ans[0m[0;34m,[0m [0ms_ans[0m[0;34m,[0m [0mquestion[0m[0;34m,[0m [0mans_long_text[0m[0;34m,[0m [0mans_short_text[0m [0;34m=[0m [0mpredict_fast[0m[0;34m([0m[0mjson_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     64[0m [0;34m[0m[0m
[1;32m     65[0m     [0mpreds[0m[0;34m[[0m[0mwrite_i[0m[0;34m][0m [0;34m=[0m [0ml_ans[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2441825380.py[0m in [0;36mpredict_fast[0;34m(json_data)[0m
[1;32m     33[0m             [0;31m# Fix: dot product result may be a numpy.matrix (no .toarray()); convert robustly.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m             [0mdots[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mfloat[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mc_vec[0m [0;34m@[0m [0mq_vec[0m[0;34m.[0m[0mT[0m[0;34m)[0m[0;34m.[0m[0mravel[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 35[0;31m             [0mc_norms[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mc_vec[0m[0;34m.[0m[0mmultiply[0m[0;34m([0m[0mc_vec[0m[0;34m)[0m[0;34m.[0m[0msum[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     36[0m [0;34m[0m[0m
[1;32m     37[0m         [0mdenom[0m [0;34m=[0m [0mc_norms[0m [0;34m*[0m [0mfloat[0m[0;34m([0m[0mq_norm[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'matrix' object has no attribute 'multiply'
