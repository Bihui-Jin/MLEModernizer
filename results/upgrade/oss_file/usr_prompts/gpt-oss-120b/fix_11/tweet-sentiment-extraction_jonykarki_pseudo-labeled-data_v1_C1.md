# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5116

# 7. Whether higher score is better

Higher is better

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

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
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
        .mean()
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

if SentenceTransformer is not None:
    try:
        _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _sentiment_emb = {
            "positive": _st_model.encode("positive", convert_to_tensor=True),
            "negative": _st_model.encode("negative", convert_to_tensor=True),
            "neutral": _st_model.encode("neutral", convert_to_tensor=True),
        }
    except Exception:
        _st_model = None
        _sentiment_emb = {}
else:
    _st_model = None
    _sentiment_emb = {}


def _default_substring(text: str, length: int) -> str:
    """Return a middle snippet of `length` tokens (or whole text if shorter)."""
    tokens = text.split()
    if len(tokens) <= length:
        return text
    start = (len(tokens) - length) // 2
    return " ".join(tokens[start : start + length]).strip()


def _best_substring_by_overlap(text: str, sentiment: str, max_len: int) -> str:
    """
    Simple fallback: choose the substring (≤max_len tokens) that shares the most
    cue words with the sentiment. If no overlap is found, return a middle snippet.
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
    Choose the most similar substring using SentenceTransformer if available;
    otherwise fall back to the lightweight overlap heuristic.
    """
    if _st_model is None:
        return _best_substring_by_overlap(text, sentiment, max_len)

    tokens = text.split()
    candidates = []
    for length in range(1, max_len + 1):
        for start in range(0, len(tokens) - length + 1):
            candidates.append(" ".join(tokens[start : start + length]))

    cand_emb = _st_model.encode(
        candidates, convert_to_tensor=True, normalize_embeddings=True
    )
    sentiment_emb = _sentiment_emb.get(sentiment)
    if sentiment_emb is None:
        return _best_substring_by_overlap(text, sentiment, max_len)

    sims = util.cos_sim(cand_emb, sentiment_emb).squeeze(1)  # (num_cand,)
    best_idx = int(sims.argmax().item())
    return candidates[best_idx]


def extract_selected(text: str, sentiment: str) -> str:
    """
    Rule‑based extractor with sentiment‑specific window size.
    - Neutral → short middle snippet (fallback when model unavailable).
    - Positive/negative → locate first cue word and build a forward‑biased window.
      If no cue word is found, fall back to a similarity‑based selector.
    """
    text = text.strip()
    if not text:
        return text

    if sentiment == "neutral":
        target_len = avg_len_by_sent.get(sentiment, 0)
        if target_len <= 0:
            target_len = 3
        return _default_substring(text, target_len)

    cue_set = positive_words if sentiment == "positive" else negative_words
    tokens = text.split()
    idx = None
    for i, tok in enumerate(tokens):
        if _clean_token(tok) in cue_set:
            idx = i
            break

    if idx is None:
        target_len = avg_len_by_sent.get(sentiment, 6)
        return _best_substring_by_similarity(text, sentiment, max_len=target_len)

    target_len = avg_len_by_sent.get(sentiment, 6)
    start = idx
    end = min(len(tokens), idx + target_len)

    while (end - start) < target_len and start > 0:
        start -= 1

    while end < len(tokens) and not re.search(r"[.,!?]$", tokens[end - 1]):
        end += 1

    return " ".join(tokens[start:end]).strip()


df_test = pd.read_csv(TEST_CSV)

final_output = [
    extract_selected(row.text, row.sentiment) for row in df_test.itertuples()
]



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
submission = pd.read_csv(SAMPLE_SUBMISSION)
assert len(final_output) == len(submission), "Prediction length mismatch."

submission["selected_text"] = final_output
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created with", len(submission), "rows.")
