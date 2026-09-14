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

0.3464473485946655

# 6. Current score

0.53065

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53065) has done: 'I fix the crash by resolving the Hugging Face path-vs-repo-id issue: your `from_pretrained` call is treating a non-existent local directory as a Hub repo ID. The minimal robust fix is to search for an actually-existing local model directory and, if none is present, fall back to a public model ID while forcing `local_files_only=True` first and then allowing online download only if needed. This unblock inference so `predictions` is created, and then the submission-writing cell run and produce a valid `submission.csv` with the required columns. I also add a tiny safeguard to ensure predictions length matches the test set and no `None` values end up in the CSV.'
- What this solution (achieved 0.0) has done: 'I fix the failure in model loading by (1) forcing fully-offline behavior, (2) broadening the search for an actually-present local QA model under `/kaggle/input`, and (3) adding a safe fallback that still produces a valid submission even if no model is available (score drop, but the run complete). I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that triggers the `MessageFactory.GetPrototype` AttributeError in this environment. These changes keep the inference logic identical when a local model is found, and only change behavior when the environment can’t load any model offline. Finally, I ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.53065) has done: 'Your current 0.0 score is consistent with the fallback branch being used (no model loaded), which outputs essentially a 1-character “answer” and Jaccard to ~0 on most rows. The smallest score-improving change is to make model loading robust in this environment by (1) allowing an online Hub download only if no local model exists, and (2) restricting the candidate Hub models to known multilingual QA-finetuned checkpoints that work well for chaii. This preserves your exact inference logic (same tokenizer/model API, same span scoring, same max_length/stride, same single-window behavior), but makes it far more likely that `model_loaded=True` so predictions become meaningful. I also add a tiny post-processing guard to never output empty answers (falls back to first non-empty token of context), which generally improves Jaccard slightly without changing the core approach.'
- What this solution (achieved 0.53065) has done: 'The crash comes from an incompatibility between the installed `protobuf` runtime and what `transformers` pulls in during model/tokenizer loading; it manifests as `MessageFactory.GetPrototype` missing. The minimal, score-neutral fix is to force the pure-Python protobuf implementation *before* importing `transformers`, which avoids the failing compiled API path in this environment. I keep your model/inference logic identical, just adjust the environment variable handling and import order so the pipeline runs end-to-end and writes a valid `submission.csv`. No score-tuning changes are introduced since your current score (0.53065) is already well above the target.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

TEST_PATH_CANDIDATES = [
    "../input/chaii-hindi-and-tamil-question-answering/test.csv",
    "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv",
    "../kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv",
    "../kaggle/data/chaii-hindi-and-tamil-question-answering/test.csv",
    "/kaggle/data/chaii-hindi-and-tamil-question-answering/test.csv",
]
test_path = next((p for p in TEST_PATH_CANDIDATES if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any known locations: {TEST_PATH_CANDIDATES}"
    )

test_df = pd.read_csv(test_path)
test_df.head()



## === cell 1
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = torch.device("cpu")

MODEL_DIR_CANDIDATES = [
    "../input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "/kaggle/input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "../kaggle/input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "../kaggle/data/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "/kaggle/data/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
]


def find_local_hf_qa_model_dir():
    for p in MODEL_DIR_CANDIDATES:
        if os.path.isdir(p):
            return p

    search_roots = ["/kaggle/input", "../input", "/kaggle/data", "../kaggle/data"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for cfg in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
            d = os.path.dirname(cfg)
            has_weights = (
                os.path.exists(os.path.join(d, "pytorch_model.bin"))
                or len(glob.glob(os.path.join(d, "*.safetensors"))) > 0
            )
            if has_weights:
                return d
    return None


local_model_dir = find_local_hf_qa_model_dir()

max_length = 384
doc_stride = 128
max_answer_tokens = 30
batch_size = 8


def load_tokenizer_and_model(name_or_path: str, local_files_only: bool):
    tok = AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True, local_files_only=local_files_only
    )
    mdl = AutoModelForQuestionAnswering.from_pretrained(
        name_or_path, local_files_only=local_files_only
    )
    return tok, mdl


tokenizer = None
model = None
model_loaded = False
load_error = None

HUB_FALLBACK_MODELS = [
    "deepset/xlm-roberta-base-squad2",
    "deepset/xlm-roberta-large-squad2",
    "sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1",  # last-resort (may fail as QA)
]

if local_model_dir is not None:
    try:
        tokenizer, model = load_tokenizer_and_model(
            local_model_dir, local_files_only=True
        )
        model.to(device)
        model.eval()
        model_loaded = True
    except Exception as e:
        load_error = e
        model_loaded = False
else:
    for name in HUB_FALLBACK_MODELS:
        try:
            tokenizer, model = load_tokenizer_and_model(name, local_files_only=True)
            model.to(device)
            model.eval()
            model_loaded = True
            load_error = None
            break
        except Exception as e:
            load_error = e
            model_loaded = False

    if not model_loaded:
        for name in HUB_FALLBACK_MODELS:
            try:
                tokenizer, model = load_tokenizer_and_model(
                    name, local_files_only=False
                )
                model.to(device)
                model.eval()
                model_loaded = True
                load_error = None
                break
            except Exception as e:
                load_error = e
                model_loaded = False


def predict_batch(contexts, questions):
    enc = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=False,  # keep single window per example (minimal + fast)
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offset_mapping = enc.pop("offset_mapping").cpu().numpy()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        start_logits = out.start_logits.detach().cpu().numpy()
        end_logits = out.end_logits.detach().cpu().numpy()

    answers = []
    for i in range(len(contexts)):
        offsets = offset_mapping[i]
        s_logits = start_logits[i]
        e_logits = end_logits[i]

        valid = offsets[:, 1] > offsets[:, 0]

        s_logits_masked = np.where(valid, s_logits, -1e9)
        e_logits_masked = np.where(valid, e_logits, -1e9)

        start_indexes = np.argsort(s_logits_masked)[-20:][::-1]
        end_indexes = np.argsort(e_logits_masked)[-20:][::-1]

        best_score = -1e18
        best_span = (0, 0)
        for s_idx in start_indexes:
            for e_idx in end_indexes:
                if e_idx < s_idx:
                    continue
                if (e_idx - s_idx + 1) > max_answer_tokens:
                    continue
                score = s_logits_masked[s_idx] + e_logits_masked[e_idx]
                if score > best_score:
                    best_score = score
                    best_span = (s_idx, e_idx)

        s_idx, e_idx = best_span
        start_char, _ = offsets[s_idx]
        _, end_char = offsets[e_idx]
        if end_char <= start_char:
            answers.append("")
        else:
            answers.append(contexts[i][int(start_char) : int(end_char)].strip())

    return answers


def nonempty_fallback_from_context(c: str) -> str:
    if not isinstance(c, str):
        return ""
    toks = c.strip().split()
    return toks[0] if len(toks) else ""


predictions = []
contexts = test_df["context"].astype(str).tolist()
questions = test_df["question"].astype(str).tolist()

if model_loaded:
    for i in range(0, len(test_df), batch_size):
        batch_ctx = contexts[i : i + batch_size]
        batch_q = questions[i : i + batch_size]
        predictions.extend(predict_batch(batch_ctx, batch_q))
    predictions = [
        (
            p
            if isinstance(p, str) and len(p.strip()) > 0
            else nonempty_fallback_from_context(contexts[i])
        )
        for i, p in enumerate(predictions)
    ]
else:
    predictions = [nonempty_fallback_from_context(c) for c in contexts]

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} != test length {len(test_df)}"
    )

len(predictions), predictions[:3], (
    "MODEL_DIR",
    local_model_dir,
    "MODEL_LOADED",
    model_loaded,
    "LOAD_ERROR",
    str(load_error)[:200] if load_error is not None else "",
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str),
        "PredictionString": pd.Series(predictions, dtype="string").fillna(""),
    }
)

if submission_df.shape[0] != test_df.shape[0]:
    raise RuntimeError("Submission row count mismatch.")
submission_df["PredictionString"] = submission_df["PredictionString"].fillna("")

submission_df.to_csv("submission.csv", index=False)
submission_df.head()
