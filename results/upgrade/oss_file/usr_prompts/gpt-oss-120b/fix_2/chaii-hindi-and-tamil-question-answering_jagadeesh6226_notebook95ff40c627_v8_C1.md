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

0.467526763677597

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the failing imports, load a publicly‑available QA model (roberta‑base‑squad2) that works with the current Transformers version, adjust the prediction function to handle long inputs safely, make sure the prediction list length matches the test set, clean the text, and finally write a proper `submission.csv` with the required `PredictionString` column.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering



## === cell 1
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")



## === cell 2
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def predict_answer(question: str, context: str) -> str:
    """
    Extract answer span using the loaded QA model.
    Truncation is enabled to keep the total length ≤ 512 tokens.
    """
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        start_scores, end_scores = model(**inputs, return_dict=False)

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1  # inclusive

    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[answer_start:answer_end]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer




## === cell 4
question_lst = test_df["question"].tolist()
context_lst = test_df["context"].tolist()

predict_list = []
for q, c in tqdm(zip(question_lst, context_lst), total=len(question_lst)):
    try:
        pred = predict_answer(q, c)
    except Exception:
        pred = ""  # fallback to empty string on unexpected errors
    predict_list.append(pred)



## === cell 5
test_df["predicted_ans"] = predict_list



## === cell 6
import re

reg_patterns = [
    r"^[0-9]+\.",  # 1. , 12.
    r"^[0-9]\.",  # 1.
    r"^[0-9][0-9]\.",  # 12.
    r"^[0-9][0-9][0-9]\.",  # 123.
    r"^\([0-9]+\)",  # (1)
    r"^[0-9]+\)",  # 1)
]


def clean_text(txt):
    for pat in reg_patterns:
        txt = re.sub(pat, " ", txt)
    txt = txt.replace("(", " ").replace(")", " ")
    return txt.strip()


test_df["predicted_ans"] = test_df["predicted_ans"].apply(clean_text)



## === cell 7
submission = pd.DataFrame(
    {"id": test_df["id"], "PredictionString": test_df["predicted_ans"]}
)



## === cell 8
submission.to_csv("submission.csv", index=False)
