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

0.007682021241635

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The notebook fails because it tries to load a Hugging Face QA model from a local Kaggle dataset that isn’t present, and outbound downloads are disabled, so the model can’t be fetched and `predictions` is never created. I fix this by making model discovery robust: search common `/kaggle/input/**` locations for an already-packaged QA model snapshot and load it strictly offline; if none is found, fall back to a safe baseline that still produces a valid submission (empty strings). I also make the tokenization call compatible across tokenizer implementations by not requesting offset mappings when we can’t run QA, and ensure the submission is always written with the required columns and `.csv` suffix. This keeps the core QA inference logic intact when a local model is available, and otherwise unblocks end-to-end execution.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

TEST_PATH = "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "../input/chaii-hindi-and-tamil-question-answering/test.csv"

test_df = pd.read_csv(TEST_PATH)
test_df.head()



## === cell 1
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


PRIMARY_LOCAL_MODEL_DIR = (
    "/kaggle/input/pretrained-xlm-models-for-squad/mrm8488/xlm-multi-finetuned-xquadv1"
)
FALLBACK_LOCAL_MODEL_ID = (
    "deepset/xlm-roberta-large-squad2"  # only if already cached locally
)


def _iter_local_candidate_model_dirs():
    yield PRIMARY_LOCAL_MODEL_DIR

    roots = ["/kaggle/input", "../input"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = dirpath[len(root) :].count(os.sep)
            if rel_depth > 6:
                dirnames[:] = []
                continue

            if "config.json" in filenames:
                yield dirpath

            dirnames[:] = [d for d in dirnames if not d.startswith(".")]


def _try_load_from_dir(model_dir):
    try:
        if not os.path.isdir(model_dir):
            return None
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True, local_files_only=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_dir, local_files_only=True
        )
        return tokenizer, model, model_dir
    except Exception:
        return None


def _load_qa_model_offline():
    seen = set()
    for cand in _iter_local_candidate_model_dirs():
        if cand in seen:
            continue
        seen.add(cand)
        out = _try_load_from_dir(cand)
        if out is not None:
            return out

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            FALLBACK_LOCAL_MODEL_ID, use_fast=True, local_files_only=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            FALLBACK_LOCAL_MODEL_ID, local_files_only=True
        )
        return tokenizer, model, FALLBACK_LOCAL_MODEL_ID + " (cached)"
    except Exception:
        return None


loaded = _load_qa_model_offline()
tokenizer = model = None
model_source = None
if loaded is not None:
    tokenizer, model, model_source = loaded

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_seq_length = 384
doc_stride = 128
max_answer_length = 30

predictions = []

if model is None:
    predictions = [""] * len(test_df)
else:
    model.eval()
    model.to(device)

    with torch.no_grad():
        for ctx, q in test_df[["context", "question"]].to_numpy():
            enc = tokenizer(
                q,
                ctx,
                truncation="only_second",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=False,
                return_offsets_mapping=True,
                padding=False,
                return_tensors="pt",
            )

            offset_mapping = enc.pop("offset_mapping")[0].tolist()
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

            start_logits = outputs.start_logits[0].detach().cpu().numpy()
            end_logits = outputs.end_logits[0].detach().cpu().numpy()

            sequence_ids = enc.sequence_ids(0)
            valid_token_idxs = [
                i
                for i, sid in enumerate(sequence_ids)
                if sid == 1
                and offset_mapping[i] is not None
                and offset_mapping[i][0] is not None
            ]

            if not valid_token_idxs:
                predictions.append("")
                continue

            best_score = -1e18
            best_span = (valid_token_idxs[0], valid_token_idxs[0])

            start_candidates = np.argsort(start_logits)[-50:][::-1]
            end_candidates = np.argsort(end_logits)[-50:][::-1]

            valid_set = set(valid_token_idxs)
            for s in start_candidates:
                if s not in valid_set:
                    continue
                for e in end_candidates:
                    if e not in valid_set:
                        continue
                    if e < s:
                        continue
                    if (e - s + 1) > max_answer_length:
                        continue

                    score = float(start_logits[s] + end_logits[e])
                    if score > best_score:
                        best_score = score
                        best_span = (s, e)

            s, e = best_span
            start_char, end_char = offset_mapping[s][0], offset_mapping[e][1]

            if start_char is None or end_char is None or start_char >= end_char:
                pred_text = ""
            else:
                pred_text = ctx[start_char:end_char].strip()

            predictions.append(pred_text)

len(predictions), predictions[:3], model_source



## === cell 2
if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} does not match test_df length {len(test_df)}"
    )

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str).values,
        "PredictionString": pd.Series(predictions, dtype="string").fillna("").values,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head()
