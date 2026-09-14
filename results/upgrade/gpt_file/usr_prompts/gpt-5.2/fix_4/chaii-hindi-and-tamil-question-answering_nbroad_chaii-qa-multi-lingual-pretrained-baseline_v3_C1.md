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
        return_tensors="pt",
        padding=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offset_mapping = enc["offset_mapping"].cpu().numpy()  # (num_features, seq_len, 2)

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

    start_logits = (
        outputs.start_logits.detach().cpu().numpy()
    )  # (num_features, seq_len)
    end_logits = outputs.end_logits.detach().cpu().numpy()

    best_score = -1e18
    best_span = (0, 0)

    for feat_idx in range(start_logits.shape[0]):
        seq_ids = enc.sequence_ids(feat_idx)
        ctx_token_mask = np.array([sid == 1 for sid in seq_ids], dtype=bool)

        s_logits = np.where(ctx_token_mask, start_logits[feat_idx], -1e9)
        e_logits = np.where(ctx_token_mask, end_logits[feat_idx], -1e9)

        start_indexes = np.argsort(s_logits)[-n_best:][::-1]
        end_indexes = np.argsort(e_logits)[-n_best:][::-1]

        for s in start_indexes:
            for e in end_indexes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                start_char, _ = offset_mapping[feat_idx, s]
                _, end_char = offset_mapping[feat_idx, e]
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

len(predictions), predictions[0] if predictions else None



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/241550667.py in <cell line: 0>()
      1 # Ensure predictions length matches test rows; if any mismatch, fail fast (prevents invalid submissions).
----> 2 assert len(predictions) == len(
      3     test_df
      4 ), f"Predictions ({len(predictions)}) != test rows ({len(test_df)})"
      5 

AssertionError: Predictions (0) != test rows (112)
