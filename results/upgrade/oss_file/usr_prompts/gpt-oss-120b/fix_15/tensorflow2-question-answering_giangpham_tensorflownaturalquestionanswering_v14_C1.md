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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.378029797642873

# 6. Current score

0.3197

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3197) has done: 'The script was failing because of a stray non‑code line, protobuf import errors, and heavy TensorFlow/transformers usage that isn’t needed for generating a valid submission. I removed the non‑code line, dropped the TensorFlow‑related imports, and replaced the model‑based pipeline with a lightweight “first‑candidate” heuristic: for each test example we read the cleaned document, pick the first long‑answer candidate (adjusted for `<tag>` tokens) as the long answer, and leave the short answer blank. This eliminates the runtime errors, keeps the original data‑processing logic, and still produces reasonably‑sized predictions that should move the score toward the target.'
- What this solution (achieved 0.3197) has done: 'I keep the overall pipeline unchanged but improve the long‑answer heuristic: instead of always taking the first candidate, the code now selects the candidate whose token span shares the most words with the question (a simple relevance measure). This modest change is inexpensive, preserves all existing logic, and is expected to raise the micro‑F1 from 0.3197 closer to the target 0.3780.'
- What this solution (achieved 0.31812) has done: 'I add a safety check in **cell 2** to ensure candidate token indices never exceed the document length, preventing the IndexError. Candidates with out‑of‑range indices are skipped, and if all are discarded the long‑answer fields stay blank. No other logic is changed, so the overall approach and scoring behaviour remain unchanged while guaranteeing a valid `submission.csv` is produced.'
- What this solution (achieved 0.31812) has done: 'I keep the overall pipeline unchanged and only adjust the short‑answer handling.  
For questions that are not explicit YES/NO, the previous code inserted a text snippet, which does not match the expected short‑answer format and can hurt the micro‑F1. Now the short answer be left blank in those cases, while YES/NO answers remain unchanged. This small change keeps the core logic intact and should move the score closer to the target.'
- What this solution (achieved 0.31814) has done: 'I improve the long‑answer heuristic in **cell 3** so that a candidate is chosen only when it shares at least one word with the question; otherwise the prediction is left blank. This reduces false‑positive long answers, which should raise precision and move the micro‑F1 score closer to the target while keeping all other logic unchanged.'
- What this solution (achieved 0.31812) has done: 'I relax the long‑answer selection rule so that a candidate is chosen even when it shares zero words with the question (by removing the “> 0” overlap check). This keeps the core logic unchanged while increasing recall, which should raise the micro‑F1 toward the target score.'
- What this solution (achieved 0.31814) has done: 'I tighten the long‑answer selection so that a candidate is only chosen when it shares **at least one** word with the question (otherwise the prediction stays blank). This reduces false‑positive long answers, improving precision and moving the micro‑F1 closer to the target. No other logic is changed, keeping the pipeline and short‑answer handling intact.'
- What this solution (achieved 0.31798) has done: 'Implemented a more robust short‑answer detector that strips punctuation and uses word‑boundary regex, and added a fallback long‑answer rule: if no candidate shares any word with the question, the shortest candidate (≤ 30 tokens) is chosen to improve recall. These minimal tweaks keep the original pipeline intact while aiming to raise the micro F1 toward the target score.'
- What this solution (achieved 0.31798) has done: 'I slightly relax the long‑answer selection rule: if no candidate shares any word with the question we still use the best‑overlap candidate (even when the overlap is zero) instead of forcing a fallback to a short candidate. This modest change increases recall and moves the micro‑F1 closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.3197) has done: 'Implemented a stricter short‑answer heuristic and refined long‑answer selection.  
- The YES/NO detector now only triggers when “yes” or “no” appears as the final word of the question (reducing false positive short answers).  
- A long‑answer candidate is accepted only if it shares **at least one** word with the question; otherwise the fallback to the shortest candidate (≤ 30 tokens) is used, improving precision and overall micro‑F1.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random
import json
import re
from tqdm import tqdm




## === cell 1
cleanr = re.compile("<.*?>")


def clean_html(raw_html):
    return re.sub(cleanr, "<tag>", raw_html)




## === cell 2
def getRawAndCleanTextDocs(test_file):
    """
    Yields one processed sample at a time.
    Each sample contains:
        - example_id (str)
        - question (str)
        - document_tokens (list of str, tags removed)
        - candidates (list of adjusted candidate dicts)
    """
    with open(test_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading test file"):
            data = json.loads(line)
            example_id = data["example_id"]
            question = data["question_text"]
            doc_text_raw = clean_html(data["document_text"])
            doc_tokens_raw = doc_text_raw.split()
            tag_prefix = [0] * (len(doc_tokens_raw) + 1)
            for i, tok in enumerate(doc_tokens_raw, 1):
                tag_prefix[i] = tag_prefix[i - 1] + (1 if tok == "<tag>" else 0)
            clean_doc = [tok for tok in doc_tokens_raw if tok != "<tag>"]

            adjusted_candidates = []
            for cand in data["long_answer_candidates"]:
                start, stop = cand["start_token"], cand["end_token"]
                if (
                    start < 0
                    or stop < 0
                    or start >= len(tag_prefix)
                    or stop >= len(tag_prefix)
                ):
                    continue
                tags_before_start = tag_prefix[start]
                tags_before_stop = tag_prefix[stop]
                adjusted = {
                    "start_token": start - tags_before_start,
                    "end_token": stop - tags_before_stop,
                    "top_level": cand["top_level"],
                }
                adjusted_candidates.append(adjusted)

            yield {
                "example_id": str(example_id),
                "question": question,
                "document_tokens": clean_doc,
                "candidates": adjusted_candidates,
            }




## === cell 3
def getFinalResult(test_path):
    """
    Choose the candidate whose token span has the highest word overlap with the question.
    Require at least one overlapping word; otherwise fall back to the shortest candidate
    (≤30 tokens).  Short‑answer YES/NO is inferred only when the question ends with the
    word yes or no, reducing spurious YES/NO predictions.
    """
    results = []
    yes_no_pattern = re.compile(r"\b(yes|no)\b\??$", flags=re.IGNORECASE)

    for doc in getRawAndCleanTextDocs(test_path):
        example_id = doc["example_id"]
        lan_start, lan_stop = -1, -1
        short_answer = ""

        if doc["candidates"]:
            doc_tokens = doc["document_tokens"]
            q_tokens = set(doc["question"].lower().split())

            best_overlap = -1
            best_cand = None

            for cand in doc["candidates"]:
                start, stop = cand["start_token"], cand["end_token"]
                if start < 0 or stop > len(doc_tokens) or start >= stop:
                    continue
                span_tokens = doc_tokens[start:stop]
                overlap = len(set(map(str.lower, span_tokens)).intersection(q_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cand = cand

            if best_cand is not None and best_overlap > 0:
                lan_start = best_cand["start_token"]
                lan_stop = best_cand["end_token"]
            else:
                short_candidates = [
                    c
                    for c in doc["candidates"]
                    if 0 <= c["start_token"] < c["end_token"] <= len(doc_tokens)
                ]
                if short_candidates:
                    fallback = min(
                        short_candidates,
                        key=lambda c: (c["end_token"] - c["start_token"]),
                    )
                    length = fallback["end_token"] - fallback["start_token"]
                    if length <= 30:
                        lan_start = fallback["start_token"]
                        lan_stop = fallback["end_token"]

        q_lower = doc["question"].lower()
        match = yes_no_pattern.search(q_lower)
        short_target = "NO_ANSWER"
        if match:
            short_target = match.group(1).upper()  # YES or NO

        short_answer = short_target if short_target in ("YES", "NO") else ""

        results.append(
            {
                "example_id": example_id,
                "lan_start": lan_start,
                "lan_stop": lan_stop,
                "short_answer": short_answer,
                "target": short_target,
            }
        )
    return results




## === cell 4
def getLines(preds):
    lines = []
    for doc in preds:
        long_id = f"{doc['example_id']}_long"
        short_id = f"{doc['example_id']}_short"

        long_line = {
            "example_id": long_id,
            "PredictionString": (
                ""
                if doc["lan_start"] == -1
                else f"{doc['lan_start']}:{doc['lan_stop']}"
            ),
        }
        short_line = {
            "example_id": short_id,
            "PredictionString": doc["short_answer"],
        }
        lines.extend([long_line, short_line])
    return lines




## === cell 5
def getSubmission():
    test_path = "../input/tensorflow2-question-answering/simplified-nq-test.jsonl"
    preds = getFinalResult(test_path)
    lines = getLines(preds)
    df = pd.DataFrame(lines)
    df = df.sort_values("example_id")
    df.to_csv("submission.csv", index=False, columns=["example_id", "PredictionString"])




## === cell 6
getSubmission()
