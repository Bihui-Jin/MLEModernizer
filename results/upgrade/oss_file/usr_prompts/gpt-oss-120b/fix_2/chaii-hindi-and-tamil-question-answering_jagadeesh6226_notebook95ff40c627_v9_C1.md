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

0.5884965658187866

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import tqdm
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

if Path("/kaggle/input").exists():
    BASE_INPUT = Path("/kaggle/input")
else:
    BASE_INPUT = Path("input")

test_path = BASE_INPUT / "chaii-hindi-and-tamil-question-answering" / "test.csv"
test_df = pd.read_csv(test_path)
print(f"Test rows loaded: {len(test_df)}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline(
    "question-answering",
    model=AutoModelForQuestionAnswering.from_pretrained(model_name),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    device=0 if torch.cuda.is_available() else -1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/465473829.py in <cell line: 0>()
      7     model=AutoModelForQuestionAnswering.from_pretrained(model_name),
      8     tokenizer=AutoTokenizer.from_pretrained(model_name),
----> 9     device=0 if torch.cuda.is_available() else -1,
     10 )
     11 

NameError: name 'torch' is not defined

## === cell 2
def predict_answer(question: str, context: str) -> str:
    """
    Run the QA pipeline and return the extracted answer.
    If the model returns an empty string, fallback to an empty answer.
    """
    result = qa_pipeline(question=question, context=context)
    answer = result.get("answer", "")
    return answer




## === cell 3
predictions = []
for q, c in tqdm.tqdm(zip(test_df["question"], test_df["context"]), total=len(test_df)):
    pred = predict_answer(str(q), str(c))
    predictions.append(pred)

assert len(predictions) == len(test_df), "Prediction length mismatch"



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3138198520.py in <cell line: 0>()
      2 predictions = []
      3 for q, c in tqdm.tqdm(zip(test_df["question"], test_df["context"]), total=len(test_df)):
----> 4     pred = predict_answer(str(q), str(c))
      5     predictions.append(pred)
      6 

/tmp/ipykernel_55/2629378615.py in predict_answer(question, context)
      4     If the model returns an empty string, fallback to an empty answer.
      5     """
----> 6     result = qa_pipeline(question=question, context=context)
      7     answer = result.get("answer", "")
      8     return answer

NameError: name 'qa_pipeline' is not defined

## === cell 4
import re

cleaned_preds = []
pattern = re.compile(r"^[\(\[]?\d{1,3}[\.\\)]\s*")
for ans in predictions:
    cleaned = pattern.sub("", ans).strip()
    cleaned_preds.append(cleaned)

test_df["predicted_ans"] = cleaned_preds



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4205545659.py in <cell line: 0>()
      9     cleaned_preds.append(cleaned)
     10 
---> 11 test_df["predicted_ans"] = cleaned_preds
     12 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (112)

## === cell 5
submission = pd.DataFrame(
    {"id": test_df["id"], "PredictionString": test_df["predicted_ans"]}
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, encoding="utf-8")
print(f"Submission written to {submission_path}, rows: {len(submission)}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'predicted_ans'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/4009993023.py in <cell line: 0>()
      1 # Build the final submission DataFrame with the exact column names required by Kaggle
      2 submission = pd.DataFrame(
----> 3     {"id": test_df["id"], "PredictionString": test_df["predicted_ans"]}
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'predicted_ans'
