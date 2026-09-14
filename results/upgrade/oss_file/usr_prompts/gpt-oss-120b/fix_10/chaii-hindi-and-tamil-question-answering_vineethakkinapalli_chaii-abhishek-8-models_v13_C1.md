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

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
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

0.7053247094154358

# 6. Current score

0.5689

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52976) has done: 'I fix the undefined variables, replace the missing model‑loading utilities with a standard HuggingFace pipeline, and ensure the script reads the test file, generates answer strings, cleans them, and writes a proper `submission.csv`. This resolves all runtime errors and produces a valid submission; using the pre‑trained XLM‑RoBerta fine‑tuned on SQuAD2 should bring the Jaccard score close to the target.'
- What this solution (achieved 0.5689) has done: 'The fix avoids the protobuf import error by not loading `AutoModelForQuestionAnswering` directly; instead the pipeline loads the model and tokenizer internally. The model checkpoint is switched to the larger *xlm-roberta‑large* version, which is expected to raise the Jaccard score toward the target while keeping the original inference logic unchanged. All other processing and the submission file creation remain the same.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import numpy as np
import pandas as pd
import torch
import string
from pathlib import Path
from transformers import pipeline

APEX_INSTALLED = False


class Config:
    model_type = "xlm-roberta"
    model_name_or_path = "deepset/xlm-roberta-large-squad2"
    tokenizer_name = "deepset/xlm-roberta-large-squad2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    logging_steps = 10

    output_dir = "output"
    seed = 2021




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_file(relative_path: str) -> Path:
    possible_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("./data"),
        Path("./input"),
        Path("."),
    ]
    for root in possible_roots:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {relative_path} in any known locations.")


test_csv_path = find_file("chaii-hindi-and-tamil-question-answering/test.csv")
sample_submission_path = find_file(
    "chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)

test_df = pd.read_csv(test_csv_path)

device_id = 0 if torch.cuda.is_available() else -1
qa_pipe = pipeline(
    "question-answering",
    model=Config.model_name_or_path,
    tokenizer=Config.tokenizer_name,
    device=device_id,
    framework="pt",
)

raw_predictions = {}
for idx, row in test_df.iterrows():
    question = str(row["question"])
    context = str(row["context"])
    if not question or not context:
        answer = ""
    else:
        try:
            result = qa_pipe(question=question, context=context)
            answer = result.get("answer", "")
        except Exception:
            answer = ""
    raw_predictions[row["id"]] = answer




## === cell 2
bad_starts = [".", ",", "(", ")", "-", "–", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ई.पू"

cleaned = []
for pid, pred in raw_predictions.items():
    pred = " ".join(pred.split())
    pred = pred.strip(string.punctuation)

    while any(pred.startswith(ch) for ch in bad_starts):
        pred = pred[1:]

    while any(pred.endswith(ch) for ch in bad_endings):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1]

    context = test_df.loc[test_df["id"] == pid, "context"].values[0]
    if (
        any(
            pred.endswith(suffix)
            for suffix in (tamil_ad, tamil_bc, tamil_km, hindi_ad, hindi_bc)
        )
        and (pred + ".") in context
    ):
        pred = pred + "."

    cleaned.append((pid, pred))

submission_df = pd.DataFrame(cleaned, columns=["id", "PredictionString"])
submission_path = Path("submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path.resolve()}")
