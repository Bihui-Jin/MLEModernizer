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

0.00572

# 6. Current score

0.49746

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49748) has done: 'I make your submission generator reliably produce a valid `submission.csv` that matches Kaggle’s required schema and IDs, since your current code is likely failing due to wrong input paths and by appending `_long/_short` twice. I keep the same core “random span / YES-NO heuristic” logic, but constrain the sampled token ranges so `start:end` stays within the document length (avoids invalid indices that can zero out matches). I also stop installing/downgrading protobuf at runtime (your environment already has TF 2.18 + protobuf 6, and the forced downgrade can break imports and prevent any CSV from being written). These minimal fixes should yield a valid submission and move score above “not yielded” toward a small nonzero baseline.'
- What this solution (achieved 0.4974) has done: 'The crash happens immediately when importing TensorFlow in cell 0, before any notebook logic runs. This is a known incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, where TensorFlow (via generated proto bindings) expects the older `MessageFactory.GetPrototype` API that was removed in protobuf 6. To unblock execution without changing the ML code, we can force protobuf to use its pure-Python implementation and ensure this environment variable is set *before* TensorFlow is imported. This keeps the rest of the notebook intact and deterministic, and preserves all variables/interfaces used by cell 1.'
- What this solution (achieved 0.49751) has done: 'Diagnosis: The crash happens in cell 0 while importing TensorFlow, before any model/data code runs. The environment has `protobuf==6.33.0`, and TensorFlow 2.18 can raise `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when the pure-Python protobuf implementation is forced (via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`). This setting triggers an incompatible protobuf code path during TensorFlow import.  
Patch summary: Remove the forced pure-Python protobuf implementation and instead explicitly prefer the faster C++ implementation by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp` (only if not already set), which avoids the missing `GetPrototype` attribute path and lets TensorFlow import cleanly.  
Updated cells: Only cell 0 is modified, keeping the rest unchanged.  
Compatibility notes for cell k+1: `tf` and `tf.__version__` remain available exactly as before; no downstream variables/interfaces change.  
Assumptions: The runtime has the protobuf C++ backend available (typical in Kaggle/TF images); if not, it still fall back to default behavior without forcing the problematic Python backend.'
- What this solution (achieved 0.49745) has done: 'The crash happens during `import tensorflow as tf` because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes TensorFlow try to use the C++ protobuf extension (`google.protobuf.pyext._message`) that is not available in this environment. The minimal fix is to stop forcing the C++ protobuf backend and instead force the Python protobuf implementation, which is compatible and avoids importing the missing extension. This change is localized to cell 0 and keeps the rest of the notebook logic unchanged. The `tf.__version__` print remains as-is for later debugging/compatibility checks.'
- What this solution (achieved 0.49753) has done: 'Diagnosis: The crash happens immediately when importing/initializing TensorFlow in cell 0. With protobuf==6.33.0, TensorFlow 2.18 can trigger an incompatibility where the pure-Python protobuf backend exposes a `MessageFactory` without `GetPrototype`, causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Forcing the protobuf implementation to the C++ backend (when available) avoids this code path and resolves the import-time failure.  

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"cpp"` (instead of `"python"`) before importing TensorFlow, keeping everything else unchanged.  

Updated cells: Only cell 0 is modified.  

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 are changed; `tf` is still imported and `tf.__version__` is printed as before.  

Assumptions: The environment has the protobuf C++ runtime available (typical for Kaggle/TensorFlow images); if not, TensorFlow would fail differently and would require a different dependency-level fix outside notebook scope.'
- What this solution (achieved 0.49741) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` because the cell forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes protobuf try to load the C++ extension (`google.protobuf.pyext._message`). In this environment that extension is not available/compatible (protobuf 6.x), so TensorFlow import fails with `cannot import name '_message'`.  
Patch summary: Remove the forced `"cpp"` protobuf implementation and instead force the pure-Python implementation before importing TensorFlow, which avoids the missing C++ extension and allows TensorFlow 2.18.0 to import deterministically. The rest of the cell remains unchanged.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: Variables `os` and `sys` are still imported and available; no interfaces used by cell 1 are changed. TensorFlow is successfully imported so later TF-dependent cells can run.  
Assumptions: Using protobuf’s Python implementation is acceptable for this notebook and does not change model logic, only fixes the import-time crash.'
- What this solution (achieved 0.49751) has done: 'Diagnosis: The crash happens immediately on importing TensorFlow in cell 0, before any notebook logic runs. With TensorFlow 2.18.0 and protobuf 6.33.0, forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` triggers a known incompatibility where TensorFlow (via protobuf) expects `MessageFactory.GetPrototype`, which is no longer present in newer protobuf versions. The root cause is the environment variable override that forces the pure-Python protobuf backend. Removing that override (and optionally forcing the default C++ backend) restores compatibility and allows TensorFlow to import normally.

Patch summary: Modify only cell 0 by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` override (and setting it to `"cpp"` if the variable is present) before importing TensorFlow. Keep the rest of the cell intact so downstream cells remain unchanged.

Updated cells: Only cell 0 is changed below.

Compatibility notes for cell k+1: Cell 1 depends on `os` being imported (it is, in cell 0). TensorFlow import and version print remain, so no variables or interfaces used later are altered.

Assumptions: The environment has the protobuf C++ implementation available (standard in Kaggle/TF builds); using the default/CPP backend is required for TensorFlow 2.18 with protobuf 6.x.'
- What this solution (achieved 0.49751) has done: 'The crash happens before any notebook logic runs: importing TensorFlow triggers a protobuf compatibility issue because the environment has `protobuf==6.33.0`, while TF 2.18 expects the Python-protobuf runtime API that still provides `MessageFactory.GetPrototype`. The quickest, deterministic fix is to force protobuf to use the pure-Python implementation (which preserves the expected API) *before* importing TensorFlow. Cell 0 currently removes that setting if present, which guarantees the crash; we invert that behavior by setting the env var instead. No other cells need changes, and this keeps TensorFlow usage and downstream variables exactly the same.'
- What this solution (achieved 0.49746) has done: 'The crash happens immediately when importing TensorFlow in cell 0, before any of your notebook logic runs. This is a known incompatibility between TensorFlow 2.18 and `protobuf==6.x`, where TensorFlow expects older protobuf APIs (leading to `'MessageFactory' object has no attribute 'GetPrototype'`). Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` does not resolve this API mismatch. The minimal fix is to force the pure-Python protobuf backend *and* pin protobuf to the compatible major version (5.x) at runtime before importing TensorFlow in this cell.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is None or _pb_major >= 6:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<6"])
    for _m in list(sys.modules.keys()):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import tensorflow as tf

print(tf.__version__)


## === cell 1
import numpy as np
import pandas as pd
import json


def read_lines_m(path, max_limit=4000):
    rlm = []
    ml = max_limit
    with open(path, "r") as f:
        for l in f:
            rlm.append(json.loads(l))
            ml -= 1
            if ml <= 0:
                break
    return pd.DataFrame(rlm)


p = "/kaggle/input/"

train = read_lines_m(os.path.join(p, "simplified-nq-train.jsonl"))
train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)

test = read_lines_m(os.path.join(p, "simplified-nq-test.jsonl")).reset_index(drop=True)

sub = pd.read_csv(os.path.join(p, "sample_submission.csv"))
train.shape, test.shape, sub.shape



## === cell 2
i = 99
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
np.median(la), np.median(sa)



## === cell 4
import random

aux_verbs = set(
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

pred_map = {}

for i in range(len(test)):
    tokens = (
        test.document_text[i].split() if isinstance(test.document_text[i], str) else []
    )
    n = len(tokens)

    long_pred = ""
    short_pred = ""

    if n >= 10:
        long_len = 114
        long_len = min(long_len, max(1, n - 1))
        start_min = 0
        start_max = max(start_min, n - long_len)
        r = random.randrange(start_min, start_max + 1)
        long_start = r
        long_end = min(n, r + long_len)
        if long_end > long_start:
            long_pred = f"{long_start}:{long_end}"

        q_words = set(str(test.question_text[i]).lower().split())
        if len(aux_verbs.intersection(q_words)) > 0:
            short_pred = random.choice(["YES", "NO"])
        else:
            if long_pred != "":
                short_len = 2
                s_start_min = long_start
                s_start_max = max(s_start_min, long_end - short_len)
                s = random.randrange(s_start_min, s_start_max + 1)
                s_end = min(n, s + short_len)
                if s_end > s:
                    short_pred = f"{s}:{s_end}"

    exid = test.example_id[i]
    pred_map[f"{exid}_long"] = long_pred
    pred_map[f"{exid}_short"] = short_pred

submission = sub[["example_id"]].copy()
submission["PredictionString"] = submission["example_id"].map(pred_map).fillna("")

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Non-empty predictions:",
    (submission["PredictionString"].astype(str).str.len() > 0).sum(),
)
