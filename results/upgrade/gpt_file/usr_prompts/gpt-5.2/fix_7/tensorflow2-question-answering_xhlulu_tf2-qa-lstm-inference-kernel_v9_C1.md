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

No external packages required in the script and installed.

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

0.1814550641940085

# 6. Current score

0.57117

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57117) has done: 'I fix the import/runtime crash by removing the notebook-only `tqdm_notebook` usage and guarding TensorFlow imports that can trigger protobuf issues, switching to a plain `tqdm` progress bar. I also fix the missing model/tokenizer paths by locating the correct files inside the provided `/kaggle/input/tensorflow2-question-answering/` dataset (or fall back to a safe baseline that still produces a valid submission if the files truly don’t exist). Finally, I ensure the submission is created with the exact required columns/IDs (`*_long` and `*_short`) and always written to `submission.csv` even if no model is available (score be low in that fallback, but it run end-to-end).'
- What this solution (achieved 0.57117) has done: 'I fix the TensorFlow/protobuf crash by preventing TensorFlow from importing at module import time and instead loading it lazily only if a model file is found (and falling back cleanly otherwise). I also ensure the submission always contains both `*_long` and `*_short` rows by merging long predictions and explicitly filling short predictions with blanks, matching the required format. Because your current score (0.57117) is far above the target (0.18145), I nudge performance downward toward the target band using a more conservative decision threshold (raising it from 0.5 to 0.9), which preserves the same model and inference semantics but reduces recall/overall F1. All paths and I/O remain the same, and the script always write `submission.csv`.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is well above the target (0.18146), so the smallest way to move closer is to intentionally reduce recall by making the decision threshold more conservative while keeping the exact same model, preprocessing, and submission semantics. I raise `THRESH` substantially (and make it configurable via an environment variable) so far fewer long answers are emitted, which should drop Micro F1 toward the target band without changing architecture/training. I also make the “which candidate to choose” aggregation deterministic by sorting then taking the last row per `example_id` (instead of `.max()` across mixed dtypes), which is a minimal correctness/stability fix and keeps the core logic intact. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is far above the target (0.18146), so to move closer we should *reduce* performance with the smallest possible change. The safest lever that preserves your model/inference core logic is the decision threshold; we raise it further so far fewer long answers are emitted, dropping micro-F1 toward the target band. To make the effect stable across runs, we also make the “pick best candidate per example” step deterministic (sort descending by target and take the first), without changing the underlying semantics. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is much higher than the target (0.181455), so to move closer we should intentionally reduce Micro-F1 with the smallest, safest lever that doesn’t change the model or preprocessing: make the long-answer emission threshold even more conservative. I only adjust the default `NQ_THRESH` upward (while keeping the environment override) so far fewer long answers are predicted, which should drop recall and overall F1 toward the target band. I also add a tiny safety clamp to keep the threshold within (0,1) and keep the rest of the inference and submission construction identical to preserve semantics and stability. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.57117) has done: 'Your current score (0.57117) is far above the target (0.181455), so we should *reduce* performance with the smallest safe lever that preserves the exact model/inference core logic. The minimal and most controllable change is to make the long-answer emission threshold much more conservative (raise `NQ_THRESH` default), which drastically reduce predicted long answers and thus reduce micro-F1 toward the target band. I keep the same model, tokenizer, preprocessing, candidate selection, and submission construction, only adjusting the default threshold value and keeping the existing environment override. The script still run end-to-end and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import pickle
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

TF_AVAILABLE = False
TF_IMPORT_ERROR = None




## === cell 1
def build_test(test_path):
    processed_rows = []
    with open(test_path, "r") as f:
        for line in tqdm(f, desc="Reading test jsonl"):
            ex = json.loads(line)

            doc_tokens = ex["document_text"].split(" ")
            question = ex["question_text"]
            example_id = str(ex["example_id"])

            for candidate in ex["long_answer_candidates"]:
                if candidate.get("top_level", False):
                    start = int(candidate["start_token"])
                    end = int(candidate["end_token"])
                    processed_rows.append(
                        {
                            "text": " ".join(doc_tokens[start:end]),
                            "question": question,
                            "example_id": example_id,
                            "PredictionString": f"{start}:{end}",
                        }
                    )
    return pd.DataFrame(processed_rows)




## === cell 2
directory = "/kaggle/input/tensorflow2-question-answering/"
test_path = os.path.join(directory, "simplified-nq-test.jsonl")
sample_sub_path = os.path.join(directory, "sample_submission.csv")

if not os.path.exists(test_path):
    test_path = "/kaggle/input/simplified-nq-test.jsonl"
if not os.path.exists(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"

test = build_test(test_path)
submission = pd.read_csv(sample_sub_path)

test.head()




## === cell 3
def compute_text_and_questions(test_df, tokenizer, sequence_module):
    test_text = tokenizer.texts_to_sequences(test_df.text.values)
    test_questions = tokenizer.texts_to_sequences(test_df.question.values)

    test_text = sequence_module.pad_sequences(test_text, maxlen=400)
    test_questions = sequence_module.pad_sequences(test_questions)

    return test_text, test_questions




## === cell 4
def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


candidate_model_paths = [
    os.path.join(directory, "model.h5"),
    os.path.join(directory, "model", "model.h5"),
]
candidate_tokenizer_paths = [
    os.path.join(directory, "tokenizer.pickle"),
    os.path.join(directory, "tokenizer.pkl"),
    os.path.join(directory, "tokenizer", "tokenizer.pickle"),
]

candidate_model_paths += glob.glob(
    os.path.join(directory, "**", "*.h5"), recursive=True
)
candidate_tokenizer_paths += glob.glob(
    os.path.join(directory, "**", "*.pickle"), recursive=True
)
candidate_tokenizer_paths += glob.glob(
    os.path.join(directory, "**", "*.pkl"), recursive=True
)

model_path = find_first_existing(candidate_model_paths)
tokenizer_path = find_first_existing(candidate_tokenizer_paths)

model_path, tokenizer_path



## === cell 5
load_model = None
sequence = None

if model_path is not None:
    try:
        from tensorflow.keras.models import load_model as _load_model
        from tensorflow.keras.preprocessing import sequence as _sequence

        load_model = _load_model
        sequence = _sequence
        TF_AVAILABLE = True
    except Exception as e:
        TF_AVAILABLE = False
        TF_IMPORT_ERROR = repr(e)

model = None
tokenizer = None

if TF_AVAILABLE and model_path is not None:
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        model = None
        MODEL_LOAD_ERROR = repr(e)

if tokenizer_path is not None:
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        tokenizer = None
        TOKENIZER_LOAD_ERROR = repr(e)

(model is not None), (tokenizer is not None), TF_AVAILABLE



## === cell 6
if model is None or tokenizer is None or not TF_AVAILABLE:
    final_submission = submission.copy()
    final_submission["PredictionString"] = ""
    final_submission.to_csv("submission.csv", index=False)

    debug_info = {
        "TF_AVAILABLE": TF_AVAILABLE,
        "TF_IMPORT_ERROR": TF_IMPORT_ERROR,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "MODEL_LOAD_ERROR": globals().get("MODEL_LOAD_ERROR", None),
        "TOKENIZER_LOAD_ERROR": globals().get("TOKENIZER_LOAD_ERROR", None),
    }
    debug_info



## === cell 7
if model is not None and tokenizer is not None and TF_AVAILABLE:
    test_text, test_questions = compute_text_and_questions(test, tokenizer, sequence)

    test_for_pred = test.drop(columns=["text", "question"]).copy()

    test_target = model.predict([test_text, test_questions], batch_size=256, verbose=1)
    test_target = np.asarray(test_target).reshape(-1)
    test_for_pred["target"] = test_target

    THRESH = float(os.environ.get("NQ_THRESH", "0.999999"))
    THRESH = min(max(THRESH, 0.0), 1.0)

    filtered = test_for_pred.loc[
        test_for_pred["target"] > THRESH, ["example_id", "PredictionString", "target"]
    ]

    if filtered.shape[0] > 0:
        filtered = filtered.sort_values(
            ["example_id", "target", "PredictionString"], ascending=[True, False, True]
        )
        best = filtered.groupby("example_id", as_index=False).head(1)
        result_long = best.loc[:, ["example_id", "PredictionString"]].assign(
            example_id=lambda df: df["example_id"].astype(str) + "_long"
        )
    else:
        result_long = pd.DataFrame({"example_id": [], "PredictionString": []})

    final_submission = submission.copy()
    final_submission = final_submission.merge(
        result_long, on="example_id", how="left", suffixes=("", "_pred")
    )

    pred = final_submission["PredictionString_pred"].fillna("")
    base = final_submission["PredictionString"].copy()

    is_long = final_submission["example_id"].astype(str).str.endswith("_long")
    is_short = final_submission["example_id"].astype(str).str.endswith("_short")

    base.loc[is_long] = pred.loc[is_long]
    base.loc[is_short] = ""

    final_submission["PredictionString"] = base.astype(str)
    final_submission = final_submission.drop(columns=["PredictionString_pred"])

    final_submission.to_csv("submission.csv", index=False)

final_submission.head()



## === cell 8
out = pd.read_csv("submission.csv")
assert list(out.columns) == ["example_id", "PredictionString"], out.columns
assert out.shape[0] == submission.shape[0], (out.shape, submission.shape)
assert out["example_id"].nunique() == submission["example_id"].nunique()
out.head()
