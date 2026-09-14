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

# 8. Previous improvement plan

- What this solution (achieved 0.50744) has done: 'I fix the crash in the Transformers question-answering pipeline caused by an incompatible `protobuf` runtime by forcing the pure-Python protobuf implementation *before* importing anything from `transformers`, and I add a safe fallback if the SQuAD2 model path isn’t available. Then I ensure we always generate exactly one prediction per test row (even if the pipeline returns fewer results) so the submission row count matches `test.csv`. Finally, I write `submission.csv` with the required `id,PredictionString` columns and add sanity checks to guarantee the file is valid.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_DIR = "../input/chaii-hindi-and-tamil-question-answering"
test_path = os.path.join(DATA_DIR, "test.csv")

test_df = pd.read_csv(test_path)
print("Loaded test:", test_df.shape)
test_df.head()



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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

qa_pl = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name,
    device=device_id,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
predictions = []

batch_size = 16  # small, safe default; avoids GPU/CPU memory issues
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

    batch_preds = [r.get("answer", "") if isinstance(r, dict) else "" for r in results]
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
submission_df.head()



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
