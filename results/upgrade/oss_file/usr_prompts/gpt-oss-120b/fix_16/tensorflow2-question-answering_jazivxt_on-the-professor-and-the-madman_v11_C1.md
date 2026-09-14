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
For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

## Metric
Micro F1. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

## Submission Format
For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```
`
## Data
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.

# 2. Python version

3.8

# 3. Installed packages

beautifulsoup4==4.13.4
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        input/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        working/
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
```

-> data/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> data/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> working/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

# 5. Target score

0.21367

# 6. Current score

0.0562

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00299) has done: 'I replace the slow BeautifulSoup parsing with a compiled regular‑expression that extracts paragraph text directly from the HTML string, which is far quicker. I also stream the CSV output instead of building a huge DataFrame in memory and use the built‑in csv writer. These changes keep the exact same logic for selecting long and short answers, only making the per‑row processing faster and more memory‑efficient, so the script finish well within the 600‑second limit.'
- What this solution (achieved 0.03484) has done: 'I compute the document length (`doc_len`) before any branching so it is always defined, and I use the same token span for the short answer when the question does not contain an auxiliary verb (which aligns short predictions with the long‑answer span and should modestly improve the F1). These fixes resolve the NameError and ensure a proper two‑row submission per example.'
- What this solution (achieved 0.03143) has done: 'I replace the random and very coarse YES/NO short‑answer heuristic with deterministic span‑based predictions, pick the longest paragraph when no candidate spans are available, and use a fixed fallback start position instead of random numbers. These changes keep the overall model logic unchanged while giving more realistic answer spans, which should raise the micro‑F1 toward the target.'
- What this solution (achieved 0.05251) has done: 'I add a lightweight overlap‑based heuristic when long answer candidates are present: the candidate whose token span shares the most words with the question be chosen (falling back to the longest span if there is no overlap). This keeps the overall logic unchanged while giving a more focused answer, which should raise the micro‑F1 toward the target. The rest of the script remains the same, ensuring a valid CSV is still written.'
- What this solution (achieved 0.05454) has done: 'I add a simple yes/no short‑answer heuristic that activates only for questions containing an auxiliary verb. When such a question appears, the script looks for the words “yes” or “no” in the document text and outputs the corresponding token `YES`/`NO` instead of copying the long‑answer span; otherwise it keeps the original long‑answer as the short answer. This modest change keeps all core logic unchanged while potentially increasing the correct short‑answer count and moving the micro‑F1 toward the target.'
- What this solution (achieved 0.05443) has done: 'I tighten the short‑answer heuristic so that “YES”/“NO” predictions are only emitted when the word actually appears inside the chosen long‑answer span (or, as a fallback, anywhere in the document). This reduces incorrect yes/no labels and should raise the micro‑F1 toward the target while keeping all core logic unchanged.'
- What this solution (achieved 0.05449) has done: 'I tighten the fallback when no long‑answer candidates exist (use the first 200 tokens instead of a fixed offset) and broaden the short‑answer heuristic so that a global “YES/NO” check is also applied when the question lacks an auxiliary verb. These small, deterministic changes keep the original logic but should raise the micro‑F1 score toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 0.05474) has done: 'I tighten the heuristics to reduce false YES/NO predictions and make the overlap metric focus on meaningful words.  
 • Define a small stop‑word set and filter both the question and each candidate span before computing overlap, so the chosen long answer is more likely to contain the key terms.  
 • Limit the short‑answer YES/NO check to the selected long‑answer span (or skip it entirely when the question lacks an auxiliary verb) instead of searching the whole document, which cuts many incorrect YES/NO labels.  
These modest, deterministic changes keep the core logic unchanged while expected to raise the micro‑F1 toward the target.'
- What this solution (achieved 0.0562) has done: 'I add a small token‑normalisation step (lower‑casing and stripping punctuation) and a fallback YES/NO check on the whole document. This keeps the original heuristic flow but makes overlap and short‑answer detection a bit more robust, which should raise the micro‑F1 toward the target without altering the core model logic.'

# 9. Code solution

## === cell 0
import os
import json
import random
import re
import csv
import string
import pandas as pd  # kept for optional sample read
from bs4 import BeautifulSoup as b  # retained but not used in the main loop




## === cell 1
base_dir = "/kaggle/input"

test_path = os.path.join(base_dir, "simplified-nq-test.jsonl")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

if not os.path.isfile(test_path):
    raise FileNotFoundError(f"Test file not found at expected location: {test_path}")

if os.path.isfile(sample_sub_path):
    _sample_sub = pd.read_csv(sample_sub_path, nrows=5)




## === cell 2
p_regex = re.compile(r"<p>(.*?)</p>", re.DOTALL)

aux_verbs = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "has",
    "have",
    "is",
    "may",
    "should",
    "was",
    "were",
    "will",
}

_stopwords = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "of",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
}


def _norm(tok: str) -> str:
    return re.sub(r"\W+", "", tok.lower())


random.seed(42)  # deterministic behaviour for any remaining randomness

submission_path = "submission.csv"
with open(submission_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["example_id", "PredictionString"])

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            example_id = item["example_id"]
            doc_text = item["document_text"]
            question = item["question_text"]

            doc_tokens = doc_text.split()
            doc_len = len(doc_tokens)

            q_tokens = [
                _norm(t) for t in question.split() if _norm(t) not in _stopwords
            ]
            q_tokens_set = set(q_tokens)

            candidates = item.get("long_answer_candidates", [])
            if candidates:
                best_candidate = None
                best_overlap = -1
                best_len = -1
                for c in candidates:
                    start = c.get("start_token", 0)
                    end = c.get("end_token", 0)
                    span_tokens = doc_tokens[start:end]

                    span_filtered = [
                        _norm(t) for t in span_tokens if _norm(t) not in _stopwords
                    ]
                    overlap = len(set(span_filtered) & q_tokens_set)
                    span_len = end - start

                    if overlap > best_overlap or (
                        overlap == best_overlap and span_len > best_len
                    ):
                        best_candidate = c
                        best_overlap = overlap
                        best_len = span_len

                if best_candidate is None:
                    best_candidate = max(
                        candidates,
                        key=lambda c: c.get("end_token", 0) - c.get("start_token", 0),
                    )
                start_token = best_candidate["start_token"]
                end_token = best_candidate["end_token"]
                long_answer = f"{start_token}:{end_token}"
            else:
                paragraphs = [
                    p.strip() for p in p_regex.findall(doc_text) if len(p.strip()) > 50
                ]
                if paragraphs:
                    para = max(paragraphs, key=lambda p: len(p.split()))
                    start_char = doc_text.find(para)
                    start_token = len(doc_text[:start_char].split())
                    end_token = min(start_token + len(para.split()) + 2, doc_len)
                    long_answer = f"{start_token}:{end_token}"
                else:
                    start_token = 0
                    end_token = min(300, doc_len)
                    long_answer = f"{start_token}:{end_token}"

            q_tokens_lower = question.lower().split()
            if any(v in q_tokens_lower for v in aux_verbs):
                span_tokens = doc_tokens[start_token:end_token]
                span_text = " ".join(span_tokens).lower()
                span_words = [_norm(w) for w in span_text.split()]
                if "yes" in span_words:
                    short_answer = "YES"
                elif "no" in span_words:
                    short_answer = "NO"
                else:
                    doc_words = [_norm(w) for w in doc_tokens]
                    if "yes" in doc_words:
                        short_answer = "YES"
                    elif "no" in doc_words:
                        short_answer = "NO"
                    else:
                        short_answer = long_answer
            else:
                short_answer = long_answer

            writer.writerow([f"{example_id}_long", long_answer])
            writer.writerow([f"{example_id}_short", short_answer])

print(f"Submission written to {submission_path}")
