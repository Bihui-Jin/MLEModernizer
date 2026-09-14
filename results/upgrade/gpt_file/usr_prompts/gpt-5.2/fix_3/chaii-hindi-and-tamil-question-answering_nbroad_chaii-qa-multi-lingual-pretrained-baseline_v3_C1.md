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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

BASE_PATH = "/kaggle/input/chaii-hindi-and-tamil-question-answering"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "../input/chaii-hindi-and-tamil-question-answering"

test_path = os.path.join(BASE_PATH, "test.csv")
test_df = pd.read_csv(test_path)
test_df.head()



## === cell 1
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "/kaggle/input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2",
    "../input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2",
]

MODEL_PATH = None
for p in CANDIDATE_MODEL_DIRS:
    if os.path.isdir(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find local model directory. Looked in:\n"
        + "\n".join(CANDIDATE_MODEL_DIRS)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, local_files_only=True
)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(device)
model.eval()

torch.set_grad_enabled(False)

predictions = []

max_length = 384
doc_stride = 128

for ctx, q in test_df[["context", "question"]].to_numpy():
    enc = tokenizer(
        q,
        ctx,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")[0].cpu().numpy()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits[0].detach().cpu().numpy()
    end_logits = outputs.end_logits[0].detach().cpu().numpy()

    seq_ids = enc.sequence_ids(0)
    ctx_token_mask = np.array([sid == 1 for sid in seq_ids], dtype=bool)

    start_logits_masked = np.where(ctx_token_mask, start_logits, -1e9)
    end_logits_masked = np.where(ctx_token_mask, end_logits, -1e9)

    start_idx = int(start_logits_masked.argmax())
    end_idx = int(end_logits_masked.argmax())

    if end_idx < start_idx:
        end_idx = start_idx

    start_char, _ = offset_mapping[start_idx]
    _, end_char = offset_mapping[end_idx]

    if (start_char, end_char) == (0, 0):
        pred = ""
    else:
        pred = ctx[int(start_char) : int(end_char)]

    predictions.append(pred)

len(predictions), predictions[0] if predictions else None



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/443490104.py in <cell line: 0>()
     18 
     19 if MODEL_PATH is None:
---> 20     raise FileNotFoundError(
     21         "Could not find local model directory. Looked in:\n"
     22         + "\n".join(CANDIDATE_MODEL_DIRS)

FileNotFoundError: Could not find local model directory. Looked in:
/kaggle/input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2
/kaggle/input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2
../input/pretrained-xlm-models-for-squad/deepset/xlm-roberta-large-squad2
../input/pretrained-xlm-models-for-squad/xlm-roberta-large-squad2

## === cell 2
assert len(predictions) == len(
    test_df
), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(str), "PredictionString": predictions}
)

submission_df.to_csv("submission.csv", index=False)

submission_df.head()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2090420292.py in <cell line: 0>()
      1 # Ensure predictions length matches test rows; if any mismatch, fail fast (prevents invalid submissions).
----> 2 assert len(predictions) == len(
      3     test_df
      4 ), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"
      5 

NameError: name 'predictions' is not defined
