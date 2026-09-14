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

0.727562665939331

# 6. Current score

0.00955

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00773) has done: 'The changes speed up data preparation by using a faster iterator and pre‑allocating the feature list, shrink DataLoader overhead by using a single worker, and move tensors to the GPU only once per batch. Inference now reuses the same device tensors and avoids repeated `.cuda()` calls, while still loading each fold’s checkpoint and averaging logits exactly as before, preserving full model logic and prediction accuracy.'
- What this solution (achieved 0.0) has done: 'I added the missing imports, defined the simple `APEX_INSTALLED` flag, and replaced the undefined functions with a straightforward `transformers` question‑answering pipeline that runs the pretrained XLM‑RoBERTa model on each test example. The pipeline directly produces answer strings, which are then cleaned and written to a CSV file named `submission.csv` with the required column names. This fixes the runtime errors and provides a functional baseline that should achieve a reasonable Jaccard score toward the target.'
- What this solution (achieved 0.00955) has done: 'I added the missing `torch` import, removed the failing custom model loader, and created the QA pipeline directly from the model name/path. This avoids the protobuf error and the undefined `torch` reference, letting the script run end‑to‑end and correctly write a `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import string
import torch  # added import for device handling

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)

APEX_INSTALLED = False


class Config:
    model_type = "xlm_roberta"
    model_name_or_path = (
        "../input/xlm-roberta-large-squad-v2"
        if os.path.isdir("../input/xlm-roberta-large-squad-v2")
        else "xlm-roberta-large"
    )
    config_name = (
        "../input/xlm-roberta-large-squad-v2"
        if os.path.isdir("../input/xlm-roberta-large-squad-v2")
        else "xlm-roberta-large"
    )
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = (
        "../input/xlm-roberta-large-squad-v2"
        if os.path.isdir("../input/xlm-roberta-large-squad-v2")
        else "xlm-roberta-large"
    )
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
def make_model(args):
    """
    Load config, tokenizer and model, falling back to the hub model if a local path is missing.
    """
    try:
        config = AutoConfig.from_pretrained(args.config_name)
    except Exception:
        config = AutoConfig.from_pretrained("xlm-roberta-large")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path, config=config
    )
    return config, tokenizer, model




## === cell 2
test_path = "../input/chaii-hindi-and-tamil-question-answering/test.csv"
test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

qa_pipeline = pipeline(
    "question-answering",
    model=Config().model_name_or_path,
    tokenizer=Config().tokenizer_name,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
)




## === cell 3
predictions = {}
for idx, row in tqdm(test.iterrows(), total=len(test), desc="Running QA pipeline"):
    context = row["context"]
    question = row["question"]
    try:
        result = qa_pipeline(question=question, context=context, top_k=1)
        answer = result["answer"]
    except Exception:
        answer = ""
    answer = " ".join(answer.split())
    answer = answer.strip(string.punctuation)
    predictions[row["id"]] = answer

submission = pd.DataFrame(list(predictions.items()), columns=["id", "PredictionString"])

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")
