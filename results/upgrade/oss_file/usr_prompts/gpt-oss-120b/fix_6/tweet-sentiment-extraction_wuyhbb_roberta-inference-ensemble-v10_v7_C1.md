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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7193046808242798

# 6. Current score

0.57682

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60506) has done: 'I replace the missing utility imports and the faulty model‑based pipeline with a compact, self‑contained implementation that reads the test data, applies a simple rule‑based extractor (returning the whole tweet for neutral sentiment and the first sentiment‑related word for positive/negative), and writes a correctly‑named `submission.csv`. This eliminates the import errors, missing files, and tokenizer/model loading problems while still producing a valid submission that can achieve a reasonable Jaccard score.'
- What this solution (achieved 0.6014) has done: 'I enhance the rule‑based extractor so that, for positive and negative tweets, it returns a short phrase surrounding the first sentiment keyword (up to two words before and after) instead of only the single keyword. This larger, more contextual snippet usually matches the true selected text better, which should raise the Jaccard score toward the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.59683) has done: 'The rule‑based extractor is expanded to capture a more natural phrase around the sentiment keyword: after locating the first positive/negative word, the snippet is grown outward until a punctuation token (.,!?) is met, while still limiting the window to at most two words on each side. This richer context usually matches the true selected text better, moving the Jaccard score upward toward the target. No core architecture changes are made; the script still reads the test set, generates `selected_text`, and writes a valid `submission.csv`.'
- What this solution (achieved 0.59728) has done: 'I extend the positive and negative keyword lists with common sentiment terms and adjust the snippet expansion so the ending punctuation token is included in the returned text. This keeps the overall rule‑based approach unchanged while giving the extractor more chances to locate the correct sentiment word and capture the full surrounding phrase, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.57682) has done: 'I extend the keyword lists with the most frequent words found in the training selected_text for each sentiment, then reuse the same rule‑based extractor. Adding these data‑driven keywords should capture more relevant phrases and raise the Jaccard score toward the target while keeping the original logic intact.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np
from collections import Counter




## === cell 1
test_path = os.path.join("..", "input", "tweet-sentiment-extraction", "test.csv")
test = pd.read_csv(test_path)




## === cell 2
POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "amazing",
    "nice",
    "love",
    "awesome",
    "fantastic",
    "best",
    "happy",
    "wonderful",
    "perfect",
    "like",
    "cool",
    "okay",
    "ok",
    "well",
    "satisfied",
    "liked",
    "pleased",
    "enjoy",
    "enjoyed",
    "delight",
    "delighted",
]
NEGATIVE_WORDS = [
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "sad",
    "angry",
    "poor",
    "disappointed",
    "negative",
    "sucks",
    "cry",
    "horrible",
    "unhappy",
    "depressed",
    "annoyed",
    "disgusted",
    "disgusting",
    "dislike",
    "disliked",
    "painful",
    "pain",
    "regret",
]


def _is_punct_token(tok: str) -> bool:
    """Return True if the token ends with a typical sentence‑ending punctuation."""
    return bool(re.search(r"[.!?]$", tok))




## === cell 3
train_path = os.path.join("..", "input", "tweet-sentiment-extraction", "train.csv")
train = pd.read_csv(train_path)


def _clean(tok: str) -> str:
    """Lower‑case token and strip surrounding punctuation."""
    return re.sub(r"^\W+|\W+$", "", tok).lower()


pos_counter = Counter()
neg_counter = Counter()
stop_words = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
}

for _, row in train.iterrows():
    sentiment = str(row["sentiment"]).lower()
    if sentiment not in {"positive", "negative"}:
        continue
    words = [_clean(w) for w in str(row["selected_text"]).split()]
    words = [w for w in words if w and w not in stop_words and len(w) > 2]
    if sentiment == "positive":
        pos_counter.update(words)
    else:
        neg_counter.update(words)

TOP_N = 30
for w, _ in pos_counter.most_common():
    if w not in POSITIVE_WORDS:
        POSITIVE_WORDS.append(w)
    if len([x for x in POSITIVE_WORDS if x not in pos_counter]) >= TOP_N:
        break

for w, _ in neg_counter.most_common():
    if w not in NEGATIVE_WORDS:
        NEGATIVE_WORDS.append(w)
    if len([x for x in NEGATIVE_WORDS if x not in neg_counter]) >= TOP_N:
        break




## === cell 4
def extract_selected_text(row):
    """Return a contextual selected_text for a row.

    - Neutral sentiment ⇒ whole tweet.
    - Positive/negative ⇒ locate first sentiment keyword (now from the enriched lists)
      and expand outward until a punctuation token is hit, keeping at most two
      tokens on each side of the keyword. The ending punctuation token is included.
    """
    text = str(row["text"])
    sentiment = str(row["sentiment"]).lower()
    if sentiment == "neutral":
        return text

    keywords = POSITIVE_WORDS if sentiment == "positive" else NEGATIVE_WORDS
    tokens = text.split()
    keyword_idx = None

    for idx, token in enumerate(tokens):
        clean_token = re.sub(r"^\W+|\W+$", "", token).lower()
        if clean_token in keywords:
            keyword_idx = idx
            break

    if keyword_idx is None:
        return text

    start = max(0, keyword_idx - 2)
    end = min(len(tokens), keyword_idx + 3)  # exclusive

    while start > 0 and not _is_punct_token(tokens[start - 1]):
        start -= 1

    while end < len(tokens) and not _is_punct_token(tokens[end]):
        end += 1

    if end < len(tokens) and _is_punct_token(tokens[end]):
        end += 1

    return " ".join(tokens[start:end])


test["selected_text"] = test.apply(extract_selected_text, axis=1)




## === cell 5
submission_path = "submission.csv"
test[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
