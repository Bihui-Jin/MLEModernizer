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

0.5439859628677368

# 6. Current score

0.64781

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.64781) has done: 'We fix the `protobuf`/`transformers` incompatibility causing the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `transformers`. Then we fix the tokenizer batching error by removing `return_tensors="pt"` (which can’t work with variable-length overflowed features unless padded) and instead manually convert each overflow feature to tensors for inference. Finally, we ensure we generate exactly one prediction per test row and write a valid `submission.csv` with the required `id,PredictionString` columns.'
- What this solution (achieved 0.64781) has done: 'We fix the `MessageFactory.GetPrototype` crash by strengthening the protobuf workaround: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also force a safe protobuf version path *before* importing `transformers`, then hard-fail with a clear message if the runtime still uses the incompatible C++ backend. This change is execution-only (score-neutral) and preserves your model/inference logic. We also keep your existing per-feature tensor conversion (avoids padding/overflow batching issues) and ensure the submission is always written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import torch

try:
    from google.protobuf.internal import api_implementation as _pb_api_impl

    _pb_type = _pb_api_impl.Type()
    if _pb_type != "python":
        raise RuntimeError(
            "Incompatible protobuf backend detected. Expected pure-Python protobuf implementation "
            f"but got '{_pb_type}'. Ensure PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set "
            "before importing transformers."
        )
except Exception as e:
    print("Warning: could not validate protobuf backend:", repr(e))

CANDIDATE_BASE_PATHS = [
    "/kaggle/input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data/chaii-hindi-and-tamil-question-answering",
    "../input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/input",
    "/kaggle/data",
]

BASE_PATH = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.isfile(os.path.join(p, "test.csv")) and os.path.isfile(
        os.path.join(p, "train.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in expected competition input directories. "
        f"Tried: {CANDIDATE_BASE_PATHS}"
    )

test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

required_cols = {"id", "context", "question"}
missing = required_cols - set(test_df.columns)
if missing:
    raise ValueError(
        f"test.csv is missing required columns: {missing}. Columns: {list(test_df.columns)}"
    )

print("BASE_PATH:", BASE_PATH)
print("test_df shape:", test_df.shape)
print("sample_submission shape:", sample_sub.shape)




## === cell 1
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "/kaggle/input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
    "/kaggle/input/pretrained-xlm-models-for-squad",
    "../input/pretrained-xlm-models-for-squad",
]

MODEL_PATH = None
for p in CANDIDATE_MODEL_DIRS:
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "config.json")):
            MODEL_PATH = p
            break
        try:
            for name in os.listdir(p):
                cand = os.path.join(p, name)
                if os.path.isdir(cand) and os.path.isfile(
                    os.path.join(cand, "config.json")
                ):
                    MODEL_PATH = cand
                    break
        except Exception:
            pass
    if MODEL_PATH is not None:
        break

FALLBACK_HF_MODEL = "deepset/xlm-roberta-large-squad2"
LOCAL_ONLY = MODEL_PATH is not None
MODEL_ID = MODEL_PATH if LOCAL_ONLY else FALLBACK_HF_MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, local_files_only=LOCAL_ONLY
)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_ID, local_files_only=LOCAL_ONLY
)

model.to(device)
model.eval()
torch.set_grad_enabled(False)

print("MODEL_ID:", MODEL_ID)
print("device:", device)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
predictions = []

max_length = 384
doc_stride = 128
n_best = 20
max_answer_length = 30

for ctx, q in test_df[["context", "question"]].to_numpy():
    enc = tokenizer(
        q,
        ctx,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    input_ids_list = enc["input_ids"]
    attention_mask_list = enc["attention_mask"]
    token_type_ids_list = enc.get("token_type_ids", None)

    best_score = -1e18
    best_span = (0, 0)

    num_features = len(input_ids_list)
    for feat_idx in range(num_features):
        input_ids = torch.tensor(
            [input_ids_list[feat_idx]], dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            [attention_mask_list[feat_idx]], dtype=torch.long, device=device
        )

        if token_type_ids_list is not None:
            token_type_ids = torch.tensor(
                [token_type_ids_list[feat_idx]], dtype=torch.long, device=device
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits.detach().cpu().numpy()[0]  # (seq_len,)
        end_logits = outputs.end_logits.detach().cpu().numpy()[0]  # (seq_len,)

        seq_ids = enc.sequence_ids(feat_idx)
        ctx_token_mask = np.array([sid == 1 for sid in seq_ids], dtype=bool)

        s_logits = np.where(ctx_token_mask, start_logits, -1e9)
        e_logits = np.where(ctx_token_mask, end_logits, -1e9)

        start_indexes = np.argsort(s_logits)[-n_best:][::-1]
        end_indexes = np.argsort(e_logits)[-n_best:][::-1]

        offsets = enc["offset_mapping"][feat_idx]

        for s in start_indexes:
            s_off = offsets[s]
            if s_off is None:
                continue
            for e in end_indexes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                e_off = offsets[e]
                if e_off is None:
                    continue

                start_char, _ = s_off
                _, end_char = e_off
                if start_char is None or end_char is None:
                    continue
                if start_char == 0 and end_char == 0:
                    continue

                score = float(s_logits[s] + e_logits[e])
                if score > best_score:
                    best_score = score
                    best_span = (int(start_char), int(end_char))

    if best_score <= -1e17:
        pred = ""
    else:
        pred = ctx[best_span[0] : best_span[1]]

    predictions.append(pred)

print("n_predictions:", len(predictions))
print("first_pred:", predictions[0] if predictions else None)




## === cell 3
assert len(predictions) == len(
    test_df
), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(str), "PredictionString": predictions}
)

if set(sample_sub["id"].astype(str)) != set(submission_df["id"].astype(str)):
    print("Warning: submission ids do not match sample_submission ids as a set.")

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
