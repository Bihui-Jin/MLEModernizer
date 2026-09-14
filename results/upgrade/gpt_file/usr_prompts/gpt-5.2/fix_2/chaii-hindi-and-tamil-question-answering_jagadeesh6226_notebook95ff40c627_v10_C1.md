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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.5892518758773804

# 6. Current score

0.03989

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.03989) has done: 'I fix the environment-breaking `transformers.pipeline` import error by pinning a compatible protobuf implementation at runtime, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` issue. Then I remove the hardcoded missing model path (`../input/qa-data-hindi-tamil/...`) and instead load a standard QA model from the local Hugging Face cache if available, with a safe fallback to an extractive heuristic so the notebook always produces predictions. I also fix the row-count mismatch by reading the correct test file (7172 rows) and ensure `predicted_ans` is always created for every row. Finally, I guarantee the submission has exactly the required columns (`id`, `PredictionString`) and is written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd



## === cell 1
TEST_PATH = "../input/chaii-hindi-and-tamil-question-answering/test.csv"
SAMPLE_PATH = "../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"

test_df = pd.read_csv(TEST_PATH)
sample_df = pd.read_csv(SAMPLE_PATH)

print("test_df shape:", test_df.shape)
print("sample_df shape:", sample_df.shape)
test_df.head()



## === cell 2
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



## === cell 3
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import tqdm



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
CANDIDATE_MODELS = [
    "deepset/xlm-roberta-base-squad2",
    "deepset/roberta-base-squad2",
    "distilbert-base-multilingual-cased-distilled-squad",
]

qa_pipe = None
load_errors = []

for m in CANDIDATE_MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
        mod = AutoModelForQuestionAnswering.from_pretrained(m, local_files_only=True)
        qa_pipe = pipeline(
            "question-answering",
            model=mod,
            tokenizer=tok,
            device=-1,
        )
        print(f"Loaded QA model from local cache: {m}")
        break
    except Exception as e:
        load_errors.append((m, repr(e)))

if qa_pipe is None:
    print("WARNING: Could not load any Hugging Face QA model from local cache.")
    print("Will use heuristic fallback. Model load errors (first 2):", load_errors[:2])




## === cell 5
def heuristic_answer(question: str, context: str) -> str:
    """
    Deterministic extractive fallback: pick a short span from the context.
    Keeps core 'extract from context' semantics so submission is valid.
    """
    if not isinstance(context, str) or not context.strip():
        return ""
    words = context.strip().split()
    return " ".join(words[:12])


def predict_answer(question, context):
    if qa_pipe is not None:
        out = qa_pipe(question=question, context=context)
        ans = out.get("answer", "")
        if ans is None:
            ans = ""
        return str(ans)
    return heuristic_answer(question, context)




## === cell 6
context_lst = test_df["context"].astype(str).tolist()
question_lst = test_df["question"].astype(str).tolist()

predict_list = []
for i in tqdm.tqdm(range(len(question_lst))):
    predict_list.append(predict_answer(question_lst[i], context_lst[i]))

len(predict_list), len(test_df)



## === cell 7
test_df["predicted_ans"] = predict_list
test_df[["id", "predicted_ans"]].head()



## === cell 8
regList = [
    r"^[0-9]+\.",
    r"^[0-9]\.",
    r"^[0-9][0-9]\.",
    r"^[0-9][0-9][0-9]\.",
    r"^\([0-9]+\)",
    r"^[0-9]+\)",
]

for reg in regList:
    test_df["predicted_ans"] = test_df["predicted_ans"].str.replace(
        reg, " ", regex=True
    )

test_df["predicted_ans"] = test_df["predicted_ans"].str.replace(
    r"\[\d*]", " ", regex=True
)
test_df["predicted_ans"] = test_df["predicted_ans"].str.replace("(", "", regex=False)
test_df["predicted_ans"] = test_df["predicted_ans"].str.replace(")", "", regex=False)

test_df["predicted_ans"] = (
    test_df["predicted_ans"].str.replace(r"\s+", " ", regex=True).str.strip()
)



## === cell 9
final_test = pd.DataFrame(
    {
        "id": test_df["id"],
        "PredictionString": test_df["predicted_ans"].fillna("").astype(str),
    }
)

assert len(final_test) == len(test_df)
assert list(final_test.columns) == ["id", "PredictionString"]

final_test.head()



## === cell 10
final_test.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_test.shape)
print(final_test.isna().sum())
