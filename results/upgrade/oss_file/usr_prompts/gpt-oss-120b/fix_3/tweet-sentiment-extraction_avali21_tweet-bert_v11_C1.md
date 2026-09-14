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

0.4605951011180877

# 6. Current score

0.18107

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17712) has done: 'The fix removes the problematic transformers and torch imports, replaces the BERT‑based model with a lightweight heuristic that selects, for each test tweet, the training selected_text of the same sentiment that has the highest Jaccard similarity to the tweet (using the cleaned text). This avoids the import errors, ensures a valid submission.csv with the required columns, and improves the expected score by leveraging learned text patterns while keeping the overall pipeline structure simple.'
- What this solution (achieved 0.18107) has done: 'I add a small keyword‑based fallback that extracts a contiguous phrase from each tweet containing sentiment‑related words (positive or negative). This phrase is compared using the same Jaccard metric and, if it scores better than the best training‑based candidate, it replaces it. The change keeps the overall heuristic structure intact while giving a more targeted prediction that should raise the score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import re
import string




## === cell 1
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sample_submission = pd.read_csv(
    "../input/tweet-sentiment-extraction/sample_submission.csv"
)




## === cell 2
def clean_text(text):
    text = str(text).lower()
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text.strip()


train["text_raw"] = train["text"]
train["selected_text_raw"] = train["selected_text"]
test["text_raw"] = test["text"]

train["text"] = train["text"].apply(clean_text)
train["selected_text"] = train["selected_text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)




## === cell 3
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 1.0
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 4
sentiment_candidates = {}
for sentiment in train["sentiment"].unique():
    df_sent = train[train["sentiment"] == sentiment]
    mapping = {}
    for _, row in df_sent.iterrows():
        clean_sel = row["selected_text"]
        if clean_sel not in mapping:
            mapping[clean_sel] = row["selected_text_raw"]
    sentiment_candidates[sentiment] = list(mapping.keys())
    if "raw_map" not in locals():
        raw_map = {}
    raw_map.update(mapping)




## === cell 5
positive_words = {
    "good",
    "great",
    "nice",
    "excellent",
    "awesome",
    "amazing",
    "fantastic",
    "love",
    "lovely",
    "best",
    "wonderful",
    "perfect",
    "sweet",
    "nice",
    "happy",
    "delight",
    "delighted",
    "pleased",
    "positive",
    "yay",
}
negative_words = {
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "horrible",
    "sad",
    "depress",
    "depressed",
    "angry",
    "negative",
    "poor",
    "disappointed",
    "unhappy",
    "pain",
    "sucks",
    "sucked",
    "gross",
    "disgust",
}

predicted_text = {}
for idx, row in test.iterrows():
    sentiment = row["sentiment"]
    tweet_clean = row["text"]
    candidates = sentiment_candidates.get(sentiment, [])
    best_candidate = (
        row["selected_text_raw"] if "selected_text_raw" in row else row["text_raw"]
    )
    best_score = -1.0

    for cand_clean in candidates:
        score = jaccard(tweet_clean, cand_clean)
        if score > best_score:
            best_score = score
            best_candidate = raw_map[cand_clean]  # use raw version

    if sentiment in ["positive", "negative"]:
        raw_tokens = row["text_raw"].split()
        clean_tokens = [
            re.sub("[%s]" % re.escape(string.punctuation), "", t).lower()
            for t in raw_tokens
        ]
        keyword_set = positive_words if sentiment == "positive" else negative_words
        idxs = [i for i, tok in enumerate(clean_tokens) if tok in keyword_set]
        if idxs:
            start, end = idxs[0], idxs[-1]
            span_raw = " ".join(raw_tokens[start : end + 1])
            span_clean = clean_text(span_raw)
            span_score = jaccard(tweet_clean, span_clean)
            if span_score > best_score:
                best_score = span_score
                best_candidate = span_raw

    predicted_text[row["textID"]] = best_candidate




## === cell 6
submission_df = pd.DataFrame(
    {"textID": test["textID"], "selected_text": test["textID"].map(predicted_text)}
)




## === cell 7
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
