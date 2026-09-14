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
joblib==1.5.2
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

0.7075726985931396

# 6. Current score

0.37882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I replace the broken tokenization and model code with a simple baseline that returns the whole tweet as the selected text, ensuring the script runs end‑to‑end and creates a correctly sized `submission.csv`. This fixes the missing imports, undefined variables, and length mismatches while keeping the overall structure unchanged.'
- What this solution (achieved 0.60116) has done: 'I add a lightweight rule‑based extractor that looks for sentiment‑related cue words (positive or negative) and returns a short surrounding phrase; for neutral tweets it keeps the whole tweet. This modest change keeps the overall pipeline unchanged while providing more precise selected text, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.51801) has done: 'I replace the simple cue‑word window heuristic with a lightweight similarity‑based selector. For each tweet we generate short contiguous substrings (up to 6 tokens), embed them with a small Sentence‑Transformer model and pick the substring whose embedding is most similar to the sentiment label (“positive”, “negative”, “neutral”). This keeps the overall pipeline unchanged, adds only a few lines, and usually yields a tighter selected text, moving the Jaccard score upward toward the target.'
- What this solution (achieved 0.59913) has done: 'The fix removes the failing SentenceTransformer import and replaces it with a lightweight rule‑based extractor that looks for sentiment‑related cue words and returns a short surrounding window (or the whole tweet for neutral). This avoids the protobuf error, keeps the pipeline structure, and typically yields a higher Jaccard score than the previous “return whole tweet” baseline while staying within the original logic constraints. The submission file is still written correctly.'
- What this solution (achieved 0.59981) has done: 'I expand the cue‑word lists, compute the typical selected‑text length from the training set and use that length to set a sentiment‑specific window size. This keeps the overall rule‑based pipeline intact while giving a more appropriate window around the cue word, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.53135) has done: 'I keep the existing rule‑based pipeline but add a small fallback that, when no cue word is found, searches all short contiguous substrings (up to the average length for the sentiment) and picks the one whose sentence‑transformer embedding is most similar to the sentiment label. I also modestly expand the selected window to include trailing punctuation, which helps the Jaccard match. These changes are lightweight, preserve the original logic, and are aimed at raising the score toward the target.'
- What this solution (achieved 0.56763) has done: 'Implemented robust fixes and modest heuristic enhancements:
- Safely disable the SentenceTransformer model to avoid protobuf import errors.
- Enriched positive/negative cue word sets with the most frequent words from the training‑selected text.
- Corrected the trailing‑punctuation extension logic so the extracted window more often includes the proper ending punctuation.
- Added protective guards for cases where the training data fails to load.

These adjustments keep the original pipeline intact while improving selected‑text extraction and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.56558) has done: 'Implemented a refined window extraction: the selected text now starts at the first cue word and preferentially expands forward, only pulling preceding tokens when needed to meet the typical length. This improves relevance of the extracted phrase while keeping the original rule‑based pipeline intact and ensures the submission CSV is correctly written.'
- What this solution (achieved 0.5116) has done: 'Implemented a neutral‑sentiment fallback that extracts a short middle snippet instead of the whole tweet, and added a safe default window length when the training‑derived average is zero. This keeps the original rule‑based logic, avoids any model loading issues, and provides more appropriate selected‑text candidates for neutral cases, helping raise the Jaccard score toward the target while ensuring a valid CSV submission is written.'
- What this solution (achieved 0.5116) has done: 'Implemented a safe fallback for the similarity‑based selector that avoids the problematic `sentence_transformers` import by using a lightweight word‑overlap heuristic when the model cannot be loaded. Added comprehensive error‑handling around the import and model creation, ensuring the script always proceeds with the rule‑based extractor. The new heuristic selects the substring with the highest overlap with sentiment cue words, which improves relevance and pushes the Jaccard score closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.56558) has done: 'We prevent the protobuf‑related crash by always skipping the SentenceTransformer model (so the fallback logic is used) and improve the neutral‑sentiment handling: when the sentiment is neutral we now return the whole tweet (or the full text if the average length is zero), which matches the usual ground‑truth and raises the Jaccard score toward the target. The rest of the pipeline stays unchanged and a proper `submission.csv` is written.'
- What this solution (achieved 0.56959) has done: 'Implemented a safe guard around the `sentence_transformers` import and model usage to prevent the protobuf‑related crash. The code now forces the fallback heuristic path by disabling the transformer model entirely, ensuring the extraction runs without errors and the submission CSV is correctly produced.'
- What this solution (achieved 0.37422) has done: 'Implemented a safe import guard that skips loading `sentence_transformers` to avoid the protobuf `MessageFactory` error, and refined the rule‑based extractor:
* Use median selected‑text length per sentiment (more representative than the mean).
* For neutral sentiment, fall back to the overlap‑based selector instead of always returning the whole tweet.
* Minor code cleanup and comments for clarity.

These fixes ensure the script runs end‑to‑end, produces a valid `submission.csv`, and modestly improve the Jaccard score toward the target.'
- What this solution (achieved 0.37882) has done: 'I remove the problematic sentence‑transformers import (so it never raises a protobuf error) and strengthen the rule‑based extractor. For positive/negative sentiments the window be centered on the first cue word and respect the median length observed in the training data, then expand to the next punctuation. This keeps the original heuristic structure while giving a more appropriate span, which should raise the Jaccard score toward the target. The script also reliably write a valid submission.csv file.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np

SentenceTransformer = None
util = None

INPUT_PATH = "/kaggle/input/"
DATA_ROOT = os.path.join(INPUT_PATH, "tweet-sentiment-extraction")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")

try:
    df_train = pd.read_csv(TRAIN_CSV)
    df_train["selected_len"] = (
        df_train["selected_text"].astype(str).apply(lambda x: len(x.split()))
    )
    avg_len_by_sent = (
        df_train.groupby("sentiment")["selected_len"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )
except Exception:
    df_train = None
    avg_len_by_sent = {"positive": 6, "negative": 6, "neutral": 0}

positive_words = {
    "good",
    "great",
    "nice",
    "love",
    "awesome",
    "excellent",
    "fantastic",
    "happy",
    "amazing",
    "best",
    "pleasant",
    "terrific",
    "fine",
    "satisfied",
    "wonderful",
    "delightful",
    "splendid",
}
negative_words = {
    "bad",
    "terrible",
    "hate",
    "worst",
    "awful",
    "poor",
    "sad",
    "angry",
    "disappointed",
    "horrible",
    "unhappy",
    "dislike",
    "dreadful",
    "horrendous",
    "lousy",
    "dejected",
}


def _clean_token(tok: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return re.sub(r"[^\w']", "", tok.lower())


def _add_freq_cues(df, sentiment, top_n=30):
    """Add most frequent words from selected_text for a given sentiment."""
    words = (
        df[df["sentiment"] == sentiment]["selected_text"]
        .astype(str)
        .str.lower()
        .str.split()
        .explode()
    )
    words = words.apply(_clean_token)
    freq = words.value_counts()
    return set(freq.head(top_n).index)


if df_train is not None:
    positive_words.update(_add_freq_cues(df_train, "positive"))
    negative_words.update(_add_freq_cues(df_train, "negative"))


def _default_substring(text: str, length: int) -> str:
    """Return a middle snippet of `length` tokens (or whole text if shorter)."""
    tokens = text.split()
    if length <= 0 or len(tokens) <= length:
        return text
    start = (len(tokens) - length) // 2
    return " ".join(tokens[start : start + length]).strip()


def _best_substring_by_overlap(text: str, sentiment: str, max_len: int) -> str:
    """
    Choose the substring (≤max_len tokens) with the most cue‑word overlap.
    If none match, fall back to a middle snippet.
    """
    cue_set = positive_words if sentiment == "positive" else negative_words
    tokens = text.split()
    best_sub = ""
    best_score = -1
    for length in range(1, max_len + 1):
        for start in range(0, len(tokens) - length + 1):
            cand = " ".join(tokens[start : start + length])
            cand_words = {_clean_token(t) for t in cand.split()}
            score = len(cand_words & cue_set)
            if score > best_score or (
                score == best_score and len(cand_words) < len(best_sub.split())
            ):
                best_score = score
                best_sub = cand
    if best_score > 0:
        return best_sub.strip()
    return _default_substring(text, max_len)


def _best_substring_by_similarity(text: str, sentiment: str, max_len: int) -> str:
    """
    Compatibility wrapper – with the transformer disabled we always fall back
    to the overlap heuristic.
    """
    return _best_substring_by_overlap(text, sentiment, max_len)


def extract_selected(text: str, sentiment: str) -> str:
    """
    Rule‑based extractor:
    • Neutral → overlap heuristic (often returns a concise relevant span).
    • Positive/negative → locate first cue word and expand a balanced window
      whose size follows the median length for that sentiment; then extend
      to the next punctuation mark.
    """
    text = text.strip()
    if not text:
        return text

    target_len = avg_len_by_sent.get(sentiment, 6)

    if sentiment == "neutral" or target_len <= 0:
        return _best_substring_by_similarity(text, sentiment, max_len=target_len or 6)

    cue_set = positive_words if sentiment == "positive" else negative_words
    tokens = text.split()
    idx = None
    for i, tok in enumerate(tokens):
        if _clean_token(tok) in cue_set:
            idx = i
            break

    if idx is None:
        return _best_substring_by_similarity(text, sentiment, max_len=target_len)

    half = target_len // 2
    start = max(0, idx - half)
    end = start + target_len
    if end > len(tokens):
        end = len(tokens)
        start = max(0, end - target_len)

    while end < len(tokens) and not re.search(r"[.,!?]$", tokens[end - 1]):
        end += 1

    return " ".join(tokens[start:end]).strip()


df_test = pd.read_csv(TEST_CSV)
final_output = [
    extract_selected(row.text, row.sentiment) for row in df_test.itertuples()
]



## === cell 1
submission = pd.read_csv(SAMPLE_SUBMISSION)
assert len(final_output) == len(submission), "Prediction length mismatch."
submission["selected_text"] = final_output
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(submission), "rows.")
