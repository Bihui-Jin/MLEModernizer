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

0.57105

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00316) has done: 'The timeout is dominated by reading the full 1.7GB test JSONL into a DataFrame and then running BeautifulSoup HTML parsing for every row; both are unnecessary for this rule-based baseline. I stream the JSONL line-by-line instead of materializing `test` and avoid BeautifulSoup entirely by extracting the first `<p>...</p>` block with a compiled regex (same semantic intent: “first paragraph text”), falling back to the same random-token logic when no paragraph is found. I also avoid per-row DataFrame lookups by precomputing the submission `example_id -> row index` mapping once and filling the output array directly. All randomness/seed behavior and the prediction logic (first paragraph long answer, YES/NO heuristic, otherwise short span inside long span) are preserved.'
- What this solution (achieved 0.07818) has done: 'We keep your exact rule-based logic (first `<p>` as long answer; random fallback; YES/NO heuristic; short span inside the long span) but make it more likely to hit real gold spans by aligning the “first paragraph” long answer to the provided `long_answer_candidates` instead of guessing token indices via `find()` on raw HTML. This is a minimal, metric-relevant fix because the evaluation checks exact token indices, and candidates already give valid token ranges. We compute the first `<p>` token span in the tokenized document, pick the candidate with the highest token-overlap (fallback to the first top-level candidate), and then generate the short span inside that chosen candidate range (still using your same random/YES-NO logic). This should improve score toward the target without changing architecture/training (none) and still runs fast by streaming JSONL.'
- What this solution (achieved 0.52316) has done: 'Your current score (0.07818) is much higher than the target (0.00776), so we should *reduce* performance toward the target with the smallest, safest change. The simplest metric-relevant way is to stop trying to match gold token spans via `long_answer_candidates` and instead mostly predict blank answers, which drives F1 down while keeping submission validity. To avoid collapsing all the way to (near) zero, we keep a small deterministic fraction of examples using your existing heuristic (candidate-overlap long span + same YES/NO / short-span logic), so the score should land closer to the target band. All changes are confined to prediction gating; core heuristics and output format remain intact and the script still streams the test JSONL and writes `submission.csv`.'
- What this solution (achieved 0.57052) has done: 'Your current score (0.52316) is far above the target (0.00776), so to move *toward* the target we should deliberately reduce performance while keeping the submission valid and the same heuristic logic intact. The smallest safe lever is the gating fraction that decides how often to apply the heuristic vs leaving answers blank; decreasing it reduce F1 smoothly without changing the underlying span-selection logic. I only change `KEEP_HEURISTIC_FRAC` to a much smaller deterministic value (and keep everything else identical) so the score should drop closer to the target tolerance band. The pipeline still streams the test JSONL and writes a correct `submission.csv`.'
- What this solution (achieved 0.57104) has done: 'Your current score (0.57052) is far above the target (0.00776), so to move *toward* the target we should deliberately reduce performance with the smallest, safest lever. The cleanest minimal change is to reduce the deterministic gating rate so the heuristic triggers much less often, yielding mostly blank predictions (valid no-answer outputs) and lowering micro-F1 smoothly. I keep all span/YES-NO/random logic identical and only change `KEEP_HEURISTIC_FRAC` (plus a clarifying comment), preserving runtime and submission validity. This should reduce the score substantially and bring it closer to the target band without changing the core approach.'
- What this solution (achieved 0.57105) has done: 'Your current score (0.57104) is far above the target (0.00776), so to move toward the target we should deliberately reduce performance while keeping the same heuristic and output semantics intact. The smallest, safest lever is the deterministic gating fraction that decides how often to emit non-blank predictions; lowering it smoothly reduce micro-F1 without changing the span-selection/YES-NO/random logic. I only adjust `KEEP_HEURISTIC_FRAC` downward (and keep all other logic identical) so the submission remains valid and runtime unchanged. This should reduce the score substantially and bring it closer to the target tolerance band.'

# 9. Code solution

## === cell 0
import os
import re
import json
import random
import numpy as np
import pandas as pd
from bs4 import (
    BeautifulSoup as b,
)  # kept to preserve original dependencies / semantics if needed

BASE_PATH = "/kaggle/input/tensorflow2-question-answering/"

train_path = os.path.join(BASE_PATH, "simplified-nq-train.jsonl")
test_path = os.path.join(BASE_PATH, "simplified-nq-test.jsonl")
sub_path = os.path.join(BASE_PATH, "sample_submission.csv")


def read_jsonl(path, max_limit=None):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_limit is not None and i >= max_limit:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


train = read_jsonl(train_path, max_limit=2000)
sub = pd.read_csv(sub_path)

train.shape, sub.shape



## === cell 1
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



## === cell 2
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
np.median(la), (np.median(sa) if len(sa) else None)



## === cell 3
random.seed(0)
np.random.seed(0)

yn_words = set(
    [
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
)

sub_eids = sub["example_id"].astype(str).to_numpy()
eid_to_row = {eid: idx for idx, eid in enumerate(sub_eids)}
pred_arr = np.empty(len(sub_eids), dtype=object)
pred_arr[:] = ""  # default fill = blank prediction (valid "no-answer" submission)

P_FIRST_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def _first_paragraph_token_span(doc_text: str):
    """
    Keep the same semantic intent: use first <p> content, and locate it in tokenized doc.
    Returns (start_token, end_token) or (None, None) if no paragraph found.
    """
    m = P_FIRST_RE.search(doc_text)
    if not m:
        return None, None

    para0 = TAG_RE.sub(" ", m.group(1))
    para0 = " ".join(para0.split())
    if not para0:
        return None, None

    doc_tokens = doc_text.split()
    para_tokens = para0.split()
    if not para_tokens or not doc_tokens:
        return None, None

    L = len(para_tokens)
    max_i = len(doc_tokens) - L
    for i in range(max_i + 1):
        if doc_tokens[i : i + L] == para_tokens:
            return i, i + L

    return None, None


def _pick_candidate_by_overlap(cands, p_start, p_end):
    """
    Choose a valid long_answer_candidate span that best overlaps the first-paragraph span.
    """
    best = None
    best_overlap = -1

    for c in cands:
        if not c.get("top_level", False):
            continue
        s = int(c["start_token"])
        e = int(c["end_token"])
        if e <= s:
            continue

        if p_start is None:
            overlap = 0
        else:
            overlap = max(0, min(e, p_end) - max(s, p_start))

        if overlap > best_overlap:
            best_overlap = overlap
            best = (s, e)

    if best is not None:
        return best

    return None, None


KEEP_HEURISTIC_FRAC = (
    0.000001  # 0.0001% kept (down from 0.002%) to move score toward 0.00776
)

test_count = 0
with open(test_path, "r") as f:
    for line in f:
        row = json.loads(line)
        exid = str(row["example_id"])
        doc_text = row["document_text"]
        q_text = row["question_text"].lower()
        cands = row.get("long_answer_candidates", [])

        gate_val = (abs(hash(exid)) % 10_000) / 10_000.0
        use_heuristic = gate_val < KEEP_HEURISTIC_FRAC

        long_key = f"{exid}_long"
        short_key = f"{exid}_short"

        if not use_heuristic:
            pred_arr[eid_to_row[long_key]] = ""
            pred_arr[eid_to_row[short_key]] = ""
            test_count += 1
            continue

        p_start, p_end = _first_paragraph_token_span(doc_text)
        c_start, c_end = _pick_candidate_by_overlap(cands, p_start, p_end)

        if c_start is not None:
            r = c_start
            long_len = c_end - c_start
            long_answer = f"{c_start}:{c_end}"
        else:
            toks = doc_text.split()
            try:
                r = random.randrange(390, max(391, len(toks)))
            except ValueError:
                r = 7
            long_len = 114
            long_answer = f"{r}:{r + 114}"

        if len(yn_words.intersection(q_text.split())) > 0:
            short_answer = random.choice(["YES", "NO"])
        else:
            span_len = max(2, min(114, int(long_len) if long_len is not None else 114))
            try:
                r2 = random.randrange(r, r + span_len)
            except ValueError:
                r2 = r
            short_answer = f"{r2}:{r2 + 2}"

        pred_arr[eid_to_row[long_key]] = long_answer
        pred_arr[eid_to_row[short_key]] = short_answer

        test_count += 1

sub["PredictionString"] = pred_arr
sub["PredictionString"] = sub["PredictionString"].fillna("")

assert (
    len(sub) == 2 * test_count
), f"Submission length {len(sub)} != 2 * test length {2 * test_count}"
assert sub["example_id"].is_unique, "example_id must be unique in submission"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("KEEP_HEURISTIC_FRAC used:", KEEP_HEURISTIC_FRAC)
