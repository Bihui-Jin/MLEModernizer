# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.4563716948032379

# 6. Current score

0.50744

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50744) has done: 'I fix the crash in the Transformers question-answering pipeline caused by an incompatible `protobuf` runtime by forcing the pure-Python protobuf implementation *before* importing anything from `transformers`, and I add a safe fallback if the SQuAD2 model path isn’t available. Then I ensure we always generate exactly one prediction per test row (even if the pipeline returns fewer results) so the submission row count matches `test.csv`. Finally, I write `submission.csv` with the required `id,PredictionString` columns and add sanity checks to guarantee the file is valid.'
- What this solution (achieved 0.50744) has done: 'I fix the crash by forcing the pure-Python protobuf implementation *and* the known-compatible protobuf API version **before any transformers import**, which addresses the `MessageFactory.GetPrototype` AttributeError in newer protobufs. I keep the same Hugging Face QA pipeline approach and batching logic, only adding a small safety fallback to still write a valid submission (empty strings) if the model cannot be loaded for any reason. I also adjust the cell numbering to start at 1 (your current cell 0 becomes cell 1) so the notebook/script runs cleanly in the provided “cells” format. These changes are runtime/stability focused and should not intentionally improve score beyond your current level.'
- What this solution (achieved 0.50744) has done: 'I fix the `protobuf`/Transformers compatibility crash that triggers `MessageFactory.GetPrototype` by forcing the pure-Python protobuf implementation *and* pinning the legacy Python protobuf API version **before any `transformers` import**, plus removing any already-imported `google.protobuf` modules to ensure the env vars take effect. I keep the exact same Hugging Face QA pipeline approach and batching/prediction logic, only making the minimum changes needed to successfully instantiate the pipeline. Since your current score (0.50744) is already within ±10% of the target (0.45637), I won’t make any score-improving changes—this is a stability/runtime fix to reliably produce a valid `submission.csv`. The script still fall back to empty predictions if model loading fails, ensuring a valid CSV is always written.'
- What this solution (achieved 0.50744) has done: 'I fix the `protobuf`/Transformers crash by setting the correct environment variable (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) before any Transformers-related import, and by removing the invalid/unsupported protobuf env vars that are triggering the `MessageFactory.GetPrototype` failure. I also move the protobuf module cleanup to occur before importing `transformers` to ensure the env change actually takes effect in-kernel. The QA pipeline logic, batching, and submission formatting stay the same to avoid intentional score changes (your current score is already within ±10% of the target). Finally, the script still reliably write a valid `submission.csv` even if model loading fails.'
- What this solution (achieved 0.50744) has done: 'I fix the `protobuf`/Transformers crash by forcing both the pure-Python protobuf backend and the legacy protobuf Python implementation, and by purging any already-imported `google.protobuf` modules before importing `transformers`. This is a runtime/stability change only; the QA pipeline approach, batching, and prediction handling remain the same to avoid intentionally changing your score (you’re already within the ±10% target band). I also keep the model path fallback exactly as-is and ensure a valid `submission.csv` is always written even if model loading still fails for any reason.'
- What this solution (achieved 0.50744) has done: 'I fix the `protobuf`/Transformers incompatibility that triggers `MessageFactory.GetPrototype` by ensuring the pure-Python protobuf backend is enforced *before* any indirect protobuf import, and by additionally disabling Transformers’ fast tokenizers (which can pull in protobuf-dependent components in some environments). I keep the exact same QA pipeline approach, batching, and prediction extraction so scoring behavior remains essentially unchanged (you’re already within the ±10% target band, so no intentional score improvements). I also make the data path robust to both Kaggle directory layouts while keeping the same default, so the notebook runs reliably end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.50744) has done: 'I fix the Transformers/protobuf crash by ensuring the protobuf environment variables are set before any potential indirect import, and by removing the unsupported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` setting that can trigger the `MessageFactory.GetPrototype` error in newer protobuf runtimes. I also proactively purge any already-imported protobuf modules (and related `google` namespace entries) before importing `transformers`, so the env change actually takes effect. The QA pipeline, batching, prediction extraction, and submission formatting remain the same to avoid intentional score changes (your current score is already within the ±10% target band). The script still fall back to empty predictions and always write a valid `submission.csv`.'
- What this solution (achieved 0.50744) has done: 'I fix the Transformers/protobuf crash by enforcing a protobuf configuration that works with this Kaggle image before importing `transformers`, and by purging already-imported protobuf modules so the env vars actually take effect. To keep score behavior essentially unchanged (you’re already within the ±10% target band), I preserve the same QA pipeline approach, batching, and prediction extraction logic, only making the minimal compatibility adjustments needed to get the pipeline to instantiate. I also add a second safe fallback: if the pipeline still can’t be created, we still generate a valid `submission.csv` with the correct row count and columns. Paths and submission formatting remain the same.'
- What this solution (achieved 0.50744) has done: 'I fix the `MessageFactory.GetPrototype` crash by enforcing a protobuf configuration that is compatible with newer protobuf runtimes *before* importing `transformers`, and by purging already-imported protobuf modules so the env vars actually take effect. I keep the exact same Hugging Face QA pipeline approach, batching, and prediction extraction logic; this is a runtime/stability fix and should be score-neutral (your current score is already within the ±10% target band). I also keep the existing safe fallback to empty predictions so a valid `submission.csv` is always written. Finally, I renumber the cells to start at 1 to match the required format.'
- What this solution (achieved 0.50744) has done: 'I fix the Transformers/protobuf crash by forcing the pure-Python protobuf backend and purging already-imported protobuf modules *before* importing `transformers`, and by also disabling optional audio dependencies that can trigger protobuf imports in this environment. This is a runtime/stability fix only (your current score is already within the ±10% target band, so I won’t intentionally change modeling or post-processing). I keep the same QA pipeline, batching, and prediction extraction logic, but ensure the pipeline creation is wrapped safely so the notebook always completes and writes a valid `submission.csv`. Finally, I keep the existing row-count alignment and submission sanity checks.'
- What this solution (achieved 0.50744) has done: 'I fix the `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf backend *and* explicitly forcing Transformers to avoid loading TensorFlow/Flax (which is a common trigger for protobuf descriptor issues) before importing anything from `transformers`. I also add a small, safe fallback: if the pipeline creation still fails, we try loading via `AutoModelForQuestionAnswering/AutoTokenizer` with `use_fast=False` to keep the same QA inference semantics. These changes are runtime/stability focused and should keep scoring behavior essentially unchanged (your current score is already within the ±10% target band). The rest of the batching, prediction extraction, and submission writing logic be preserved, and a valid `submission.csv` always be produced.'
- What this solution (achieved 0.50744) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing a protobuf configuration that is compatible with this Kaggle runtime *before* importing `transformers`, and by purging already-imported `google.protobuf` modules so the setting actually takes effect. This is a runtime/stability fix only (your current score is already within the ±10% target band vs 0.45637), so I not change the QA pipeline logic, batching, or post-processing beyond what’s required to run. I also make the model-loading fallback keep the same semantics while avoiding paths that re-trigger protobuf issues, and ensure the notebook always writes a valid `submission.csv` with the correct rows/columns.'
- What this solution (achieved 0.50744) has done: 'I fix the crash happening before `transformers` loads by enforcing a protobuf configuration that avoids the `MessageFactory.GetPrototype` incompatibility: set the pure-Python protobuf backend early, and purge any already-imported protobuf modules before importing `transformers`. I also make the pipeline creation explicitly torch-only (`framework="pt"`) to prevent transformers from attempting TF/Flax side paths that can trigger protobuf descriptor issues in this environment. These are runtime/stability changes only; the QA pipeline approach, batching, and prediction extraction stay the same so scoring behavior should remain essentially unchanged (and your current score is already within the ±10% band of the target). The script still always produce a valid `submission.csv` with the correct columns and row count.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_DIR = "../input/chaii-hindi-and-tamil-question-answering"
alt_DATA_DIR = "/kaggle/input/chaii-hindi-and-tamil-question-answering"
if not os.path.exists(DATA_DIR) and os.path.exists(alt_DATA_DIR):
    DATA_DIR = alt_DATA_DIR

test_path = os.path.join(DATA_DIR, "test.csv")

test_df = pd.read_csv(test_path)
print("Loaded test:", test_df.shape)
print(test_df.head())



## === cell 1
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCH_AUDIO"] = "1"
os.environ["TORCH_AUDIO_BACKEND"] = "soundfile"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

for m in list(sys.modules.keys()):
    if (
        m == "google"
        or m.startswith("google.")
        or m == "protobuf"
        or m.startswith("protobuf.")
        or m == "google.protobuf"
        or m.startswith("google.protobuf.")
    ):
        sys.modules.pop(m, None)

import torch
from transformers import pipeline

primary_model_name = (
    "../input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-base-squad2"
)
fallback_model_name = "deepset/xlm-roberta-base-squad2"

model_name = (
    primary_model_name if os.path.exists(primary_model_name) else fallback_model_name
)
print("Using model:", model_name)

device_id = 0 if torch.cuda.is_available() else -1
print("device_id:", device_id)

qa_pl = None
try:
    qa_pl = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=device_id,
        framework="pt",
    )
except Exception as e:
    print("WARNING: failed to create QA pipeline; will try a safer torch-only load.")
    print("Exception:", repr(e))

    try:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        mdl = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pl = pipeline(
            "question-answering",
            model=mdl,
            tokenizer=tok,
            device=device_id,
            framework="pt",
        )
        print("Fallback torch-only pipeline created successfully.")
    except Exception as e2:
        print(
            "WARNING: fallback torch-only load also failed; will output empty predictions."
        )
        print("Exception:", repr(e2))
        qa_pl = None



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
predictions = []

if qa_pl is None:
    predictions = [""] * len(test_df)
else:
    batch_size = 16  # keep identical logic
    n = len(test_df)

    for start in range(0, n, batch_size):
        batch = test_df.iloc[start : start + batch_size]
        inputs = [
            {"context": str(ctx), "question": str(q)}
            for ctx, q in batch[["context", "question"]].to_numpy()
        ]

        results = qa_pl(inputs)
        if isinstance(results, dict):
            results = [results]

        batch_preds = [
            r.get("answer", "") if isinstance(r, dict) else "" for r in results
        ]
        if len(batch_preds) < len(inputs):
            batch_preds.extend([""] * (len(inputs) - len(batch_preds)))
        elif len(batch_preds) > len(inputs):
            batch_preds = batch_preds[: len(inputs)]

        predictions.extend(batch_preds)

print("Predictions:", len(predictions))
print(predictions[:3])



## === cell 3
if len(predictions) != len(test_df):
    if len(predictions) < len(test_df):
        predictions = predictions + [""] * (len(test_df) - len(predictions))
    else:
        predictions = predictions[: len(test_df)]

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str).values,
        "PredictionString": pd.Series(predictions, dtype="string"),
    }
)

submission_df["PredictionString"] = submission_df["PredictionString"].fillna("")

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())



## === cell 4
assert submission_df.shape[0] == test_df.shape[0], "Row count mismatch vs test.csv"
assert list(submission_df.columns) == [
    "id",
    "PredictionString",
], "Wrong submission columns"
assert submission_df["id"].isna().sum() == 0, "Missing ids"
assert submission_df["PredictionString"].isna().sum() == 0, "Missing predictions"

print("Submission shape:", submission_df.shape)
print(submission_df.sample(5, random_state=42))
