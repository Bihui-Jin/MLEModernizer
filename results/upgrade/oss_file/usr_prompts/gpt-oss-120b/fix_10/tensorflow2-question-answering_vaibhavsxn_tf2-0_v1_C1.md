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
nltk==3.9.2
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

0.09758

# 6. Current score

0.07481

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00311) has done: 'The changes replace the heavy BeautifulSoup parsing with a lightweight regular‑expression extraction of `<p>` blocks, pre‑compile the regex once, and streamline the word‑overlap computation by tokenizing paragraphs only once. These adjustments keep the heuristic identical while cutting the per‑record runtime dramatically, allowing the full test set to be processed well under the 600‑second limit.'
- What this solution (achieved 0.00615) has done: 'Implemented deterministic handling for answer generation to eliminate unnecessary randomness and avoid spurious predictions when no suitable paragraph is found.  
- If no paragraphs are extracted, the long answer is now left blank instead of a random token window.  
- Short answers are generated deterministically: when the question contains a modal verb, a fixed “YES” is used (removing random guess), otherwise the short span is taken from the start of the predicted long answer.  
- A fixed random seed ensures reproducibility. These changes fix the runtime issues and make the heuristic more consistent, nudging the score toward the target while preserving the core logic.'
- What this solution (achieved 0.00617) has done: 'The changes remove the problematic TensorFlow import that caused an import‑time failure, and refine the heuristic so that the predicted token spans match the exact paragraph length (no extra tokens) and handle yes/no short answers more sensibly. This keeps the core logic unchanged while fixing the runtime error and modestly improving the F1 score.'
- What this solution (achieved 0.07839) has done: 'I replace the paragraph‑only heuristic with one that first examines the provided `long_answer_candidates` (token spans already aligned to the document). For each candidate I extract the text using the same whitespace tokenisation as the original code, compute word‑overlap after removing English stopwords from **both** the question and the candidate, and select the best candidate. If no valid candidate exists I fall back to the previous paragraph method. The short‑answer logic is kept unchanged except that it now uses the exact token indices from the chosen candidate, ensuring a more accurate span while preserving the overall heuristic.'
- What this solution (achieved 0.0745) has done: 'I tighten the short‑answer heuristic: instead of guessing “YES/NO” based on modal verbs (which often produces incorrect predictions), the short answer only be derived from the selected long‑answer span (a brief two‑token excerpt). If no long answer is found, the short answer stays blank. This small, deterministic change reduces many wrong YES/NO guesses and is expected to raise the micro‑F1 toward the target while keeping all core logic intact.'
- What this solution (achieved 0.07481) has done: 'I increase the short‑answer snippet from two to three tokens and add a tiny deterministic YES/NO rule: when the question contains a modal verb, the code looks for the words “yes” or “no” in the selected answer text and outputs the corresponding token (“YES”/“NO”). This keeps the core heuristic unchanged, adds only a small deterministic branch, and is expected to raise the micro‑F1 toward the target without over‑optimising.'

# 9. Code solution

## === cell 0
print("TensorFlow import skipped – not needed for this heuristic solution.")



## === cell 1
import numpy as np
import pandas as pd
import json
import nltk
from bs4 import BeautifulSoup as b
import random
import re

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


def read_lines_m(path, max_limit=None):
    """
    Read a JSONL file into a DataFrame.
    If max_limit is None, read the entire file; otherwise stop after max_limit lines.
    """
    records = []
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            count += 1
            if max_limit is not None and count >= max_limit:
                break
    return pd.DataFrame(records)


p = "../input/tensorflow2-question-answering/"

train_sample = read_lines_m(p + "simplified-nq-train.jsonl", max_limit=5000)
train_sample["D"] = [
    t[0]["long_answer"]["start_token"] for t in train_sample.annotations
]
train_sample = train_sample[train_sample["D"] > -1].reset_index(drop=True)

print(
    "Shapes -> train_sample:",
    train_sample.shape,
    "(full dataset not loaded for speed)",
)

i = 99
if i < len(train_sample):
    print("URL:", train_sample.document_url[i])
    print(train_sample.question_text[i])
    print(train_sample.long_answer_candidates[i][0])
    print(
        " ".join(
            train_sample.document_text[i].split()[
                train_sample.annotations[i][0]["long_answer"][
                    "start_token"
                ] : train_sample.annotations[i][0]["long_answer"]["end_token"]
            ]
        )
    )
    if len(train_sample.annotations[i][0]["short_answers"]) > 0:
        print(
            " ".join(
                train_sample.document_text[i].split()[
                    train_sample.annotations[i][0]["short_answers"][0][
                        "start_token"
                    ] : train_sample.annotations[i][0]["short_answers"][0]["end_token"]
                ]
            )
        )

la = [
    t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
    for t in train_sample.annotations
]
sa = [
    t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
    for t in train_sample.annotations
    if len(t[0]["short_answers"]) > 0
]
print("Median long answer length:", np.median(la))
print("Median short answer length:", np.median(sa))



## === cell 2
p_tag_regex = re.compile(r"<p[^>]*>(.*?)</p>", flags=re.DOTALL | re.IGNORECASE)

stop_words_set = set(stopwords.words("english"))
modal_verbs = {
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
random.seed(42)


def clean_para(text):
    """Remove any remaining HTML tags inside a paragraph."""
    return re.sub(r"<[^>]+>", "", text)


def qa_word_match(q, a_list):
    """
    Choose the string from a_list with highest word overlap with the question.
    Stop‑words are removed from BOTH question and candidate strings.
    """
    q_tokens = set(w for w in q.lower().split() if w not in stop_words_set)
    best_match = a_list[0] if a_list else ""
    max_overlap = -1
    for a in a_list:
        a_tokens = set(w for w in a.lower().split() if w not in stop_words_set)
        overlap = len(q_tokens & a_tokens)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = a
    return best_match


result = []

test_path = p + "simplified-nq-test.jsonl"
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        doc_text = obj["document_text"]
        question = obj["question_text"]
        example_id = obj["example_id"]

        doc_tokens = doc_text.split()

        candidates = obj.get("long_answer_candidates", [])
        candidate_spans = []
        candidate_texts = []
        for cand in candidates:
            start = cand.get("start_token", -1)
            end = cand.get("end_token", -1)
            if start is None or end is None or start < 0 or end <= start:
                continue
            if start >= len(doc_tokens) or end > len(doc_tokens):
                continue
            text = " ".join(doc_tokens[start:end])
            candidate_spans.append((start, end))
            candidate_texts.append(text)

        selected_text = ""
        if candidate_texts:
            best_text = qa_word_match(question, candidate_texts)
            idx = candidate_texts.index(best_text)
            start_token, end_token = candidate_spans[idx]
            long_answer = f"{start_token}:{end_token}"
            selected_text = best_text
        else:
            raw_paras = p_tag_regex.findall(doc_text)
            paragraphs = [
                clean_para(p).strip()
                for p in raw_paras
                if len(clean_para(p).strip()) > 50
            ]

            if paragraphs:
                chosen_para = qa_word_match(question, paragraphs)
                start_char = doc_text.find(chosen_para)
                if start_char == -1:
                    start_token = 0
                else:
                    start_token = len(doc_text[:start_char].split())
                end_token = start_token + len(chosen_para.split())
                long_answer = f"{start_token}:{end_token}"
                selected_text = chosen_para
            else:
                long_answer = ""

        if long_answer:
            lt_start, lt_end = map(int, long_answer.split(":"))
            q_words = set(question.lower().split())
            if modal_verbs & q_words:
                lowered_selected = selected_text.lower()
                if "yes" in lowered_selected.split():
                    short_answer = "YES"
                elif "no" in lowered_selected.split():
                    short_answer = "NO"
                else:
                    short_end = min(lt_start + 3, lt_end)
                    short_answer = f"{lt_start}:{short_end}"
            else:
                short_end = min(lt_start + 3, lt_end)  # three‑token snippet
                short_answer = f"{lt_start}:{short_end}"
        else:
            short_answer = ""

        result.append([f"{example_id}_long", long_answer])
        result.append([f"{example_id}_short", short_answer])

submission_df = pd.DataFrame(result, columns=["example_id", "PredictionString"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission_df)} rows.")
