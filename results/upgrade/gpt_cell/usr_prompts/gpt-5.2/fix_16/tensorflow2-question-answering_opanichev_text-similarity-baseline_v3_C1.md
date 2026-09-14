# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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


tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)


def _doc_text_iter(path: str):
    for rec in _iter_test_jsonl(path):
        yield rec["document_text"]


tfidf.fit(_doc_text_iter(DATA_PATH))


def _csr_row_l2_norm(v):
    return float(np.sqrt(v.multiply(v).sum()))




## === cell 3
def _doc_token_term_cumsums_from_tokens(doc_tokens, term_ids, term_vals, vocab_get):
    """
    Equivalent to _doc_token_term_cumsums but avoids re-splitting doc_text and avoids
    rebuilding dicts that can be created from already available arrays.
    Returns:
      pos_by_term: dict[int, np.ndarray]
      cum_by_term: dict[int, np.ndarray]
    """
    if not doc_tokens:
        return {}, {}
    if term_ids.size == 0:
        return {}, {}

    w_by_term = dict(zip(term_ids.tolist(), term_vals.tolist()))
    term_set = set(term_ids.tolist())

    counts = {}
    pos_lists = {}
    for i, tok in enumerate(doc_tokens):
        tid = vocab_get(tok)
        if tid is None:
            continue
        counts[tid] = counts.get(tid, 0) + 1
        if tid in pos_lists:
            pos_lists[tid].append(i)
        else:
            pos_lists[tid] = [i]

    pos_by_term = {}
    cum_by_term = {}
    for tid, positions in pos_lists.items():
        if tid not in term_set:
            continue
        c = counts[tid]
        if c <= 0:
            continue
        total_w = w_by_term.get(tid, 0.0)
        if total_w == 0.0:
            continue
        per_occ = total_w / c
        pos_arr = np.asarray(positions, dtype=np.int32)

        cum = np.empty(pos_arr.shape[0] + 1, dtype=np.float64)
        cum[0] = 0.0
        np.cumsum(np.full(pos_arr.shape[0], per_occ, dtype=np.float64), out=cum[1:])

        pos_by_term[tid] = pos_arr
        cum_by_term[tid] = cum

    return pos_by_term, cum_by_term


def _span_dot_and_norm(starts, ends, q_terms, q_weights, pos_by_term, cum_by_term):
    """
    Compute dot(span_vec, q_vec) and ||span_vec|| for many spans.
    Span vector is sum of per-token TF-IDF vectors, implemented via per-term cumulative sums.
    """
    n = starts.shape[0]
    dots = np.zeros(n, dtype=np.float64)
    norms2 = np.zeros(n, dtype=np.float64)

    for tid, qw in zip(q_terms, q_weights):
        tid_i = int(tid)
        pos = pos_by_term.get(tid_i)
        if pos is None:
            continue
        cum = cum_by_term[tid_i]

        left = np.searchsorted(pos, starts, side="left")
        right = np.searchsorted(pos, ends, side="left")
        contrib = cum[right] - cum[left]
        dots += contrib * float(qw)
        norms2 += contrib * contrib

    norms = np.sqrt(norms2, dtype=np.float64)
    return dots, norms




## === cell 4
def predict_fast(json_data, q_terms, q_weights, q_norm, bin_q, vocab_get):
    candidates = [
        c for c in json_data["long_answer_candidates"] if c["top_level"] == True
    ]
    doc_text = json_data["document_text"]

    doc_tokens = doc_text.split(" ")
    n_rows = len(doc_tokens)

    if q_norm == 0.0 or len(candidates) == 0 or n_rows == 0:
        best_idx = 0
    else:
        m = len(candidates)
        starts = np.empty(m, dtype=np.int32)
        ends = np.empty(m, dtype=np.int32)
        for j, c in enumerate(candidates):
            starts[j] = c["start_token"]
            ends[j] = c["end_token"]

        starts = np.clip(starts, 0, n_rows)
        ends = np.clip(ends, 0, n_rows)
        bad = ends <= starts

        Xd = tfidf.transform([doc_text])  # CSR row; same as before
        if Xd.nnz == 0:
            best_idx = 0
        else:
            term_ids = Xd.indices
            term_vals = Xd.data
            pos_by_term, cum_by_term = _doc_token_term_cumsums_from_tokens(
                doc_tokens, term_ids, term_vals, vocab_get
            )

            if not pos_by_term:
                best_idx = 0
            else:
                dots, c_norms = _span_dot_and_norm(
                    starts, ends, q_terms, q_weights, pos_by_term, cum_by_term
                )
                denom = c_norms * q_norm
                sims = np.zeros_like(dots, dtype=np.float64)
                nonzero = denom != 0.0
                sims[nonzero] = dots[nonzero] / denom[nonzero]
                sims[bad] = 0.0
                best_idx = int(np.argmax(sims))

    ans = candidates[best_idx] if candidates else {"start_token": 0, "end_token": 0}
    ans_long = f"{ans['start_token']}:{ans['end_token']}"

    ans_short = "YES" if bin_q else ""

    ans_long_text = " ".join(doc_tokens[ans["start_token"] : ans["end_token"]])
    ans_short_text = ans_short if (len(ans_short) > 0 or ans_short == "YES") else ""

    return ans_long, ans_short, ans_long_text, ans_short_text


vocab_get = tfidf.vocabulary_.get

ids = []
preds = []
questions_out = []
ans_texts = []

for json_data in tqdm(_iter_test_jsonl(DATA_PATH), total=2000, desc="Predicting"):
    exid = str(json_data["example_id"])
    q = json_data["question_text"]
    q0 = q.split(" ")[0] if q else ""
    bin_q = q0 in bin_question_tokens

    Qrow = tfidf.transform([q])  # 1xV CSR row
    if Qrow.nnz == 0:
        q_terms = np.empty(0, dtype=np.int32)
        q_weights = np.empty(0, dtype=np.float64)
        q_norm = 0.0
    else:
        q_terms = Qrow.indices.astype(np.int32, copy=False)
        q_weights = Qrow.data.astype(np.float64, copy=False)
        q_norm = float(np.sqrt(np.dot(q_weights, q_weights))) if q_weights.size else 0.0

    l_ans, s_ans, ans_long_text, ans_short_text = predict_fast(
        json_data,
        q_terms,
        q_weights,
        q_norm,
        bool(bin_q),
        vocab_get,
    )

    ids.append(exid + "_long")
    preds.append(l_ans)
    questions_out.append(q)
    ans_texts.append(ans_long_text)

    ids.append(exid + "_short")
    preds.append(s_ans)
    questions_out.append(q)
    ans_texts.append(ans_short_text)

subm = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions_out,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
subm.to_csv("test_data.csv", index=False)
subm[["example_id", "PredictionString"]].to_csv("submission.csv", index=False)

subm.head(10)
