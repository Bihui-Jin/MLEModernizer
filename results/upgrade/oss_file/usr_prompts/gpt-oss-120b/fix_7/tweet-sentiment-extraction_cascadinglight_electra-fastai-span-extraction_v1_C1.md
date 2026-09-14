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

fastai==2.8.5
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
tokenizers==0.21.2
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

0.7094413042068481

# 6. Current score

0.57693

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61626) has done: 'I fixed the import errors, removed the unusable fastai components, and replaced the whole training pipeline with a lightweight rule‑based predictor that runs end‑to‑end. The script now loads the data, applies a simple sentiment‑aware heuristic to choose a span, and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.59995) has done: 'I keep the overall structure but improve the span‑extraction heuristic: instead of returning just the first sentiment word, the new rule expands to a surrounding phrase up to punctuation marks. This modest change keeps the original logic while providing longer, more context‑rich predictions, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.59995) has done: 'The patch fixes the `selected_text` NaNs that broke the length calculation, replaces the mean span length with a more robust median length, and ensures the global `AVG_LEN` dictionary is defined before it is used. These changes let the script run end‑to‑end and produce a correctly formatted `submission.csv`, while the median‑based span length modestly improves the Jaccard score toward the target.'
- What this solution (achieved 0.56339) has done: 'I enrich the sentiment word lists with the most frequent tokens from the gold “selected_text” of each sentiment and use a blended average of median and mean span lengths as the target length. These small, data‑driven tweaks keep the original rule‑based extractor while nudging the predicted spans closer to the true distribution, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.57693) has done: 'I adjust the heuristic so it uses a more robust median span length (which better matches the distribution of gold spans) and, instead of always taking the first sentiment word, it evaluates all sentiment‑word positions and picks the phrase whose length is closest to the target length. This modest change keeps the original rule‑based approach while giving predictions that are typically longer and more aligned with the gold excerpts, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path



## === cell 1
base_dir = Path("/kaggle/input/tweet-sentiment-extraction")
train_df = pd.read_csv(base_dir / "train.csv")
test_df = pd.read_csv(base_dir / "test.csv")

train_df["selected_text"] = train_df["selected_text"].fillna("")


def _token_len(s):
    return len(s.split()) if isinstance(s, str) else 0


train_df["selected_len"] = train_df["selected_text"].apply(_token_len)

median_len_pos = train_df.loc[
    train_df["sentiment"] == "positive", "selected_len"
].median()
median_len_neg = train_df.loc[
    train_df["sentiment"] == "negative", "selected_len"
].median()
median_len_pos = median_len_pos if pd.notnull(median_len_pos) else 1
median_len_neg = median_len_neg if pd.notnull(median_len_neg) else 1

AVG_LEN = {
    "positive": median_len_pos,
    "negative": median_len_neg,
}



## === cell 2
positive_words = {
    "good",
    "great",
    "love",
    "nice",
    "excellent",
    "happy",
    "awesome",
    "fantastic",
    "amazing",
    "wonderful",
}
negative_words = {
    "bad",
    "hate",
    "terrible",
    "worst",
    "sad",
    "awful",
    "poor",
    "horrible",
    "angry",
    "disappointed",
}

from collections import Counter


def _top_tokens(sentiment, top_n=20):
    tokens = (
        train_df.loc[train_df["sentiment"] == sentiment, "selected_text"]
        .str.lower()
        .str.split()
        .explode()
    )
    tokens = tokens.str.strip(".,!?\"'")
    tokens = tokens[tokens != ""]
    return [w for w, _ in Counter(tokens).most_common(top_n)]


positive_words.update(_top_tokens("positive"))
negative_words.update(_top_tokens("negative"))


def _expand_to_target_len(tokens, left, right, target_len):
    """Grow the window [left, right] while staying inside punctuation
    boundaries until it reaches roughly target_len tokens."""
    target_len = int(round(target_len))
    while (right - left + 1) < target_len:
        expanded = False
        if left > 0 and not any(p in tokens[left - 1] for p in ".,!?"):
            left -= 1
            expanded = True
        if (right - left + 1) >= target_len:
            break
        if right + 1 < len(tokens) and not any(p in tokens[right + 1] for p in ".,!?"):
            right += 1
            expanded = True
        if not expanded:
            break
    return " ".join(tokens[left : right + 1])


def _candidate_bounds(tokens, idx):
    """Return the left‑most and right‑most indices of the sentence fragment
    that contains the token at position idx (stopping at punctuation)."""
    l = idx
    while l > 0 and not any(p in tokens[l - 1] for p in ".,!?"):
        l -= 1
    r = idx
    while r + 1 < len(tokens) and not any(p in tokens[r + 1] for p in ".,!?"):
        r += 1
    return l, r


def _extract_phrase(text, target_set, sentiment):
    """
    Evaluate all occurrences of sentiment words and return the phrase whose
    length is closest to the median gold span length for that sentiment.
    """
    tokens = text.split()
    match_idxs = [
        i for i, tok in enumerate(tokens) if tok.lower().strip(".,!?\"'") in target_set
    ]

    if not match_idxs:
        return text

    target_len = AVG_LEN.get(sentiment, len(tokens))

    best_phrase = None
    best_len_diff = float("inf")

    for idx in match_idxs:
        l, r = _candidate_bounds(tokens, idx)
        phrase = _expand_to_target_len(tokens, l, r, target_len)
        phrase_len = len(phrase.split())
        len_diff = abs(phrase_len - target_len)

        if len_diff < best_len_diff:
            best_len_diff = len_diff
            best_phrase = phrase

    return best_phrase if best_phrase is not None else text


def predict_span(row):
    text = row["text"]
    sentiment = str(row["sentiment"]).lower()

    if sentiment == "neutral" or len(text) < 3:
        return text

    if sentiment == "positive":
        return _extract_phrase(text, positive_words, "positive")
    elif sentiment == "negative":
        return _extract_phrase(text, negative_words, "negative")
    else:
        return text




## === cell 3
test_df["selected_text"] = test_df.apply(predict_span, axis=1)



## === cell 4
submission = test_df[["textID", "selected_text"]].copy()
submission.to_csv("submission.csv", index=False)
