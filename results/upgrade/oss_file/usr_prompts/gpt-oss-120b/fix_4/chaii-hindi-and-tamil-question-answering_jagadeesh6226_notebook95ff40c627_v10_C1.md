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

0.0542

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03454) has done: 'I fix the import paths, remove the failing model loading, and replace it with a simple deterministic predictor that extracts the first short snippet from the context. This ensures the script runs end‑to‑end, creates a CSV with the required `PredictionString` column, and avoids the earlier attribute and repository errors.'
- What this solution (achieved 0.04596) has done: 'I replace the naïve “first‑10‑words” predictor with a lightweight heuristic: the function now slides a window (size 10) over the context and selects the snippet that shares the most words with the question. If no overlap is found it falls back to the original first‑10‑words approach. This modest change keeps the overall pipeline unchanged while expected to raise the Jaccard‑based score toward the target.'
- What this solution (achieved 0.0542) has done: 'I added a regex‑based sentence splitter and changed the predictor to choose the whole sentence from the context that shares the most words with the question (falling back to the original window heuristic when no overlap is found). This richer excerpt should contain more answer tokens, moving the Jaccard score upward toward the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
from pathlib import Path

BASE_INPUT = Path("/kaggle/input/chaii-hindi-and-tamil-question-answering")
test_path = BASE_INPUT / "test.csv"
test_df = pd.read_csv(test_path)




## === cell 1
def simple_predict(question: str, context: str, window_size: int = 10) -> str:
    """
    Heuristic predictor:
    - Split the context into sentences.
    - Choose the sentence that shares the most words with the question.
    - If no sentence shares any word, fall back to the sliding‑window approach.
    """
    q_tokens = set(question.lower().split())

    sentences = re.split(r"[।.!?|\n]+", context)
    sentences = [s.strip() for s in sentences if s.strip()]

    best_overlap = -1
    best_sentence = None

    for sent in sentences:
        sent_tokens = set(sent.lower().split())
        overlap = len(sent_tokens & q_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent
            if overlap == len(q_tokens):  # perfect match, early exit
                break

    if best_overlap > 0 and best_sentence:
        return best_sentence
    words = context.split()
    if len(words) <= window_size:
        return context

    best_overlap = -1
    best_start = 0
    for start in range(len(words) - window_size + 1):
        window = words[start : start + window_size]
        overlap = len({w.lower() for w in window} & q_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_start = start
            if overlap == window_size:
                break

    if best_overlap <= 0:
        return " ".join(words[:window_size])
    else:
        return " ".join(words[best_start : best_start + window_size])




## === cell 2
predictions = []
for q, c in zip(test_df["question"], test_df["context"]):
    pred = simple_predict(q, c)
    predictions.append(pred)

test_df["PredictionString"] = predictions




## === cell 3
submission = test_df[["id", "PredictionString"]].copy()




## === cell 4
output_path = Path("submission.csv")
submission.to_csv(output_path, index=False)
