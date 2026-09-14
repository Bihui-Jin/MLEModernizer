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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
rich==14.2.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3
wordcloud==1.9.4

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

0.4375553727149963

# 6. Current score

0.50744

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50744) has done: 'I remove the failing `rich` dependency that triggers the protobuf `MessageFactory` error and replace it with a lightweight `cprint` that works everywhere. Then I fix the Hugging Face pipeline model loading by switching from an invalid local path (that doesn’t exist in your provided inputs) to a valid public multilingual QA checkpoint, using `device=-1` so it runs on CPU reliably in the Kaggle environment. Finally, I ensure predictions are always created and aligned to `test.csv`, and that a valid `submission.csv` with the exact required columns (`id,PredictionString`) is written end-to-end.'
- What this solution (achieved 0.50744) has done: 'I remove the problematic `rich` import path and heavily reduce optional visualization imports that trigger the protobuf `MessageFactory` error in this Kaggle environment. I keep the same QA pipeline model and inference loop (so core logic and score behavior stay essentially unchanged), but make the pipeline creation more robust by explicitly disabling TF usage and ensuring predictions are always strings. Finally, I ensure the submission is written as a valid `submission.csv` with exactly the required columns aligned to `test.csv`.'
- What this solution (achieved 0.50744) has done: 'I fix the runtime crash caused by an incompatible `protobuf` version imported indirectly by `transformers` by forcing Transformers to use the pure-Python protobuf implementation before importing it, and by importing `transformers` only after setting those environment variables. I also make the QA pipeline call more robust across Transformers versions by allowing batched inference (same model/logic, just fewer per-call overheads) while keeping outputs identical in meaning. Finally, I ensure the submission always has exactly `id,PredictionString`, matches `test.csv` row order/length, and is written to `submission.csv` in the working directory.'
- What this solution (achieved 0.50744) has done: 'We fix the runtime crash occurring during `transformers` import by forcing a compatible protobuf mode *before* importing anything that might load protobuf, and by avoiding the code paths that trigger the `MessageFactory.GetPrototype` issue in this environment. We keep the same model (`deepset/xlm-roberta-base-squad2`) and the same QA pipeline inference logic so the score behavior stays essentially the same (your current score is already above target, so we won’t intentionally improve it). We also make dataset path detection include the provided `/kaggle/data/...` layout, and ensure a valid `submission.csv` with `id,PredictionString` is always written aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import sys



## === cell 2
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import pipeline


def _pprint(x):
    print(x)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def cprint(string):
    """
    Utility function for beautiful colored printing.
    In this environment we use plain printing for maximum compatibility.
    """
    print(string)




## === cell 4
DATA_CANDIDATES = [
    "../input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/input/chaii-hindi-and-tamil-question-answering",
    "../kaggle/input/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data/chaii-hindi-and-tamil-question-answering",
    "/kaggle/data",
    "/kaggle/input",
]

DATA_DIR = None
for p in DATA_CANDIDATES:
    if os.path.exists(p) and os.path.isfile(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break

if DATA_DIR is None:
    if os.path.isfile("../input/train.csv"):
        DATA_DIR = "../input"
    elif os.path.isfile("/kaggle/input/train.csv"):
        DATA_DIR = "/kaggle/input"
    elif os.path.isfile("/kaggle/data/train.csv"):
        DATA_DIR = "/kaggle/data"
    else:
        raise FileNotFoundError(
            "Could not locate chaii dataset CSV files in expected input paths."
        )

DATA_DIR



## === cell 5
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_path, test_path, sub_path



## === cell 6
train_file = pd.read_csv(train_path)
test_file = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

train_file.shape, test_file.shape, sample_sub.shape



## === cell 7
train_file.head()



## === cell 8
train_file.info()



## === cell 9
train_file.describe(include="all")



## === cell 10
test_file.head()



## === cell 11
test_file.info()



## === cell 12
test_file.describe(include="all")



## === cell 13
sample_sub.head()



## === cell 14
cprint("Total Training Examples: {}".format(train_file.shape[0]))
cprint("Total Testing Examples: {}".format(test_file.shape[0]))



## === cell 15
train_file["language"].value_counts()



## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
MODEL_NAME = "deepset/xlm-roberta-base-squad2"

qna = pipeline(
    "question-answering",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1,
    framework="pt",
)

predictions = []

qa_inputs = [
    {"question": q, "context": c}
    for q, c in test_file[["question", "context"]].to_numpy()
]

try:
    results = qna(qa_inputs, batch_size=8)
except TypeError:
    results = [qna(context=ex["context"], question=ex["question"]) for ex in qa_inputs]

for result in results:
    ans = ""
    if isinstance(result, dict):
        ans = result.get("answer", "") or ""
    predictions.append(str(ans))

len(predictions), predictions[0] if len(predictions) else None



## === cell 22
submission = pd.DataFrame(
    {
        "id": test_file["id"].astype(str).values,
        "PredictionString": pd.Series(predictions, dtype="string").fillna("").values,
    }
)

assert (
    submission.shape[0] == test_file.shape[0]
), "Submission row count does not match test row count."

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 23
cprint("Done. Wrote submission.csv")
