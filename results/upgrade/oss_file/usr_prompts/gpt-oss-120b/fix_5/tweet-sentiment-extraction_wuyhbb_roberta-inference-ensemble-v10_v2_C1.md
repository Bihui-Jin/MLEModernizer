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

0.7257987260818481

# 6. Current score

0.59358

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60845) has done: 'I replace the failing imports and missing utility files with a lightweight, rule‑based extraction that works without external models. The new script loads the test CSV, selects a sentiment‑related word (using simple positive/negative word lists) or defaults to the full tweet, and writes a correctly‑formatted `submission.csv`. This removes the protobuf/transformers errors, fixes missing module imports, and guarantees a valid submission file.'
- What this solution (achieved 0.52543) has done: 'I expand the simple word‑list heuristic by (1) loading the training data to enrich the positive and negative word sets with tokens that actually appear in the labelled selected_text, and (2) when a sentiment word is found in a tweet, return the whole contiguous phrase around it (up to surrounding punctuation) instead of just the single word. These minimal, rule‑based changes keep the original workflow intact while giving the model more relevant vocabulary and longer, more accurate spans, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.52647) has done: 'I expand the phrase extraction so that, after locating a sentiment‑related token, the heuristic also pulls in any preceding words up to the nearest punctuation (rather than stopping at the previous space). This typically yields a longer, more context‑rich selected text and should raise the Jaccard score toward the target while keeping the overall rule‑based approach unchanged.'
- What this solution (achieved 0.59358) has done: 'I augment the heuristic by first trying to reuse exact selected‑text phrases that appear in the training set for the same sentiment.  
For each sentiment I collect the unique training phrases, sort them by length (longest first), and if a phrase is a substring of the tweet I return that exact slice (preserving the original case).  
Only when no training phrase matches do we fall back to the original token‑based extraction.  
These small rule‑based additions keep the core logic intact while giving the model more realistic spans, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np



## === cell 1
POSITIVE_WORDS = {
    "good",
    "great",
    "awesome",
    "fantastic",
    "amazing",
    "love",
    "nice",
    "excellent",
    "best",
    "wonderful",
    "perfect",
    "happy",
    "delight",
    "pleasant",
    "positive",
    "cool",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "worst",
    "sad",
    "negative",
    "poor",
    "horrible",
    "disappoint",
    "unhappy",
    "angry",
    "sucks",
    "dislike",
    "lousy",
    "gross",
    "ugly",
    "painful",
}


def _enrich_word_sets():
    train_path = os.path.join("..", "input", "tweet-sentiment-extraction", "train.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("..", "input", "train.csv")
    if not os.path.exists(train_path):
        return  # safe fallback if training file is missing

    train_df = pd.read_csv(train_path)
    for _, row in train_df.iterrows():
        sentiment = str(row["sentiment"]).lower()
        tokens = re.findall(
            r"\w+['’]?\w*|[^\w\s]", str(row["selected_text"]), flags=re.UNICODE
        )
        if sentiment == "positive":
            POSITIVE_WORDS.update(tok.lower() for tok in tokens if tok.isalpha())
        elif sentiment == "negative":
            NEGATIVE_WORDS.update(tok.lower() for tok in tokens if tok.isalpha())


_enrich_word_sets()

POS_SELECTED_PHRASES = []
NEG_SELECTED_PHRASES = []


def _populate_selected_phrases():
    train_path = os.path.join("..", "input", "tweet-sentiment-extraction", "train.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("..", "input", "train.csv")
    if not os.path.exists(train_path):
        return

    train_df = pd.read_csv(train_path)
    for _, row in train_df.iterrows():
        sentiment = str(row["sentiment"]).lower()
        phrase = str(row["selected_text"]).strip()
        if not phrase:
            continue
        if sentiment == "positive":
            POS_SELECTED_PHRASES.append(phrase)
        elif sentiment == "negative":
            NEG_SELECTED_PHRASES.append(phrase)


_populate_selected_phrases()

POS_SELECTED_PHRASES = sorted(set(POS_SELECTED_PHRASES), key=len, reverse=True)
NEG_SELECTED_PHRASES = sorted(set(NEG_SELECTED_PHRASES), key=len, reverse=True)




## === cell 2
def extract_selected_text(row):
    """
    Heuristic:
    1. Try to match a full training phrase (same sentiment) that appears
       in the tweet; return the exact substring (preserves original case).
    2. Fallback to token‑based extraction:
       • locate the first token that belongs to the sentiment‑specific word set;
       • return the contiguous phrase around it, stopping at punctuation.
    3. If nothing matches, return the whole tweet.
    """
    text = str(row["text"])
    sentiment = str(row["sentiment"]).lower()
    text_lower = text.lower()

    if sentiment == "positive":
        candidates = POS_SELECTED_PHRASES
    elif sentiment == "negative":
        candidates = NEG_SELECTED_PHRASES
    else:
        candidates = []

    for phrase in candidates:
        phrase_lower = phrase.lower()
        if phrase_lower in text_lower:
            start = text_lower.find(phrase_lower)
            end = start + len(phrase_lower)
            return text[start:end].strip()

    if sentiment == "positive":
        word_set = POSITIVE_WORDS
    elif sentiment == "negative":
        word_set = NEGATIVE_WORDS
    else:
        word_set = set()

    for match in re.finditer(r"\w+['’]?\w*|[^\w\s]", text, flags=re.UNICODE):
        token = match.group()
        if token.lower() in word_set:
            left = match.start()
            while left > 0 and text[left - 1] not in ".,!?;:":
                left -= 1
            while left > 0 and text[left - 1] == " ":
                left -= 1
            right = match.end()
            while right < len(text) and text[right] not in ".,!?;:":
                right += 1
            return text[left:right].strip()
    return text.strip()




## === cell 3
test_path = os.path.join("..", "input", "tweet-sentiment-extraction", "test.csv")
if not os.path.exists(test_path):
    test_path = os.path.join("..", "input", "test.csv")
test = pd.read_csv(test_path)



## === cell 4
test["selected_text"] = test.apply(extract_selected_text, axis=1)



## === cell 5
submission_path = "submission.csv"
test[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(test)}")
