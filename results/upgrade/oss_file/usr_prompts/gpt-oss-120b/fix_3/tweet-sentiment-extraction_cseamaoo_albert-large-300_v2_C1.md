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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.7034798860549927

# 6. Current score

0.59099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'The fix removes the unavailable `transformers` and `torch` dependencies, replaces notebook‑specific magic and shell commands with standard Python, and implements a simple fallback `run_prediction` that returns the whole tweet (or the context itself) as the selected text. This eliminates the import error, ensures `run_prediction` is defined, and correctly writes a CSV submission file, allowing the notebook to run end‑to‑end.'
- What this solution (achieved 0.59099) has done: 'I add a lightweight keyword‑based heuristic to `run_prediction` so that for positive or negative tweets it returns a short phrase around a sentiment‑related word instead of the whole tweet. This keeps the original fallback for neutral sentiment, uses only built‑in libraries, and is expected to raise the Jaccard score toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

is_inference_flag = True
try:
    tweet_models_dir = os.listdir("/kaggle/input/albert-xl-300/")
    if len(tweet_models_dir) > 0:
        is_inference_flag = True
except Exception:
    is_inference_flag = False




## === cell 1
print("Inference flag status :", is_inference_flag)




## === cell 2
if not is_inference_flag:
    pass  # placeholder for cloning logic if required




## === cell 3
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")




## === cell 4
def jaccard(str1, str2):
    a = str1.lower().split()
    b = str2.lower().split()
    c = set(a).intersection(set(b))
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 5
train_df.dropna(inplace=True)




## === cell 6
X_train, X_test = train_test_split(train_df, test_size=0.10, random_state=42)




## === cell 7
train = np.array(X_train)
val = np.array(X_test)
test = np.array(test_df)




## === cell 8
os.makedirs("data", exist_ok=True)
os.makedirs("data/models/albert", exist_ok=True)
os.makedirs("data/models/roberta", exist_ok=True)




## === cell 9
def find_all(input_str, search_str):
    """Return all start indices of search_str within input_str."""
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def do_qa_train(train_arr):
    """Create a minimal SQuAD‑style JSON structure (kept for compatibility)."""
    output = {"version": "v1.0", "data": []}
    paragraphs = []
    for line in train_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        answer = line[2]
        if not all(isinstance(x, str) for x in (answer, context, question)):
            continue
        answer_starts = find_all(context, answer)
        answers = [
            {
                "answer_start": answer_starts[0] if answer_starts else 0,
                "text": answer.lower(),
            }
        ]
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})
    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output


if not is_inference_flag:
    qa_X_train = do_qa_train(train)
    qa_X_test = do_qa_train(val)
    import json

    with open("data/train.json", "w") as f:
        json.dump(qa_X_train, f)
    with open("data/val.json", "w") as f:
        json.dump(qa_X_test, f)




## === cell 10
def convert_test_qa_json(test_arr):
    """Create a minimal test JSON (kept for compatibility)."""
    output = {"version": "v1.0", "data": []}
    paragraphs = []
    for line in test_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if not all(isinstance(x, str) for x in (context, question)):
            continue
        answers = [{"answer_start": 1000000, "text": "__None__"}]
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})
    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output


qa_test = convert_test_qa_json(test)
import json

with open("data/test.json", "w") as f:
    json.dump(qa_test, f)




## === cell 11
from collections import OrderedDict
import re

POSITIVE_KEYWORDS = {
    "good",
    "great",
    "awesome",
    "amazing",
    "fantastic",
    "love",
    "nice",
    "perfect",
    "best",
    "wonderful",
    "excellent",
    "happy",
    "liked",
    "like",
    "positive",
}
NEGATIVE_KEYWORDS = {
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "hated",
    "disappointed",
    "sad",
    "negative",
    "poor",
    "sucks",
    "shit",
    "ugly",
    "angry",
    "unhappy",
}


def _extract_phrase(context, keywords, window=2):
    """
    Return a short phrase (up to 2 words before and after) that contains a keyword.
    If no keyword is found, return the whole context (fallback).
    """
    words = re.findall(r"\w+|\S", context)  # keep punctuation as separate tokens
    lowered = [w.lower() for w in words]

    for i, w in enumerate(lowered):
        if w in keywords:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            return "".join(
                [
                    " " + token if re.match(r"\w+", token) and idx != start else token
                    for idx, token in enumerate(words[start:end])
                ]
            ).strip()
    return context


def run_prediction(question_texts, context_text):
    """
    Heuristic prediction:
    - Neutral sentiment → return the whole tweet (unchanged).
    - Positive/Negative → return a short phrase around a sentiment‑related keyword.
    Returns an OrderedDict with a dummy key to keep the original interface.
    """
    sentiment = question_texts[0].lower()
    if sentiment == "neutral":
        return OrderedDict({0: context_text})

    if sentiment == "positive":
        pred = _extract_phrase(context_text, POSITIVE_KEYWORDS)
    elif sentiment == "negative":
        pred = _extract_phrase(context_text, NEGATIVE_KEYWORDS)
    else:
        pred = context_text

    return OrderedDict({0: pred})




## === cell 12
jaccard_scores = []
predictions_x_test = []
for idx, row in X_test.head(10).iterrows():
    context = row["text"]
    selected_text = row["selected_text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict:
        predicted_text = preds_dict[key]
        predictions_x_test.append(
            {
                "selected_text": selected_text,
                "predicted_text": predicted_text,
                "sentiment": row["sentiment"],
                "textID": row["textID"],
            }
        )
        jaccard_scores.append(jaccard(selected_text, predicted_text))
print("Jaccard Score (sample):", np.mean(jaccard_scores))




## === cell 13
predictions = []
for _, row in test_df.iterrows():
    context = row["text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict:
        predicted_text = preds_dict[key]
        predictions.append({"textID": row["textID"], "selected_text": predicted_text})

predictions_df = pd.DataFrame.from_dict(predictions)

submission_path = "submission.csv"
predictions_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
