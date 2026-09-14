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

0.7258532643318176

# 6. Current score

0.47366

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38352) has done: 'I fixed the path handling (removing the undefined `__file__`) and added a lightweight heuristic that chooses, for each tweet, the longest known `selected_text` from the training set that actually appears in the tweet (grouped by sentiment). If no known phrase is found it falls back to using the whole tweet, which matches the required CSV format and improves the Jaccard baseline toward the target score. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.38352) has done: 'I add a lightweight sentiment‑keyword heuristic to the `pick_selected_text` function: after trying the known training phrases, if none match, the code look for the longest contiguous n‑gram (up to 5 words) that contains a word from a small positive or negative keyword list matching the tweet’s sentiment. This simple rule often captures the part of the tweet that conveys the sentiment, improving the Jaccard score while keeping the original logic intact. I also import `itertools` for n‑gram generation.'
- What this solution (achieved 0.33013) has done: 'I make the matching case‑insensitive when searching for known training phrases (so more phrases are found) and expand the keyword‑based fallback: after the n‑gram search, if nothing matches I locate the first sentiment keyword in the tweet and return a short window of surrounding words (up to three before and after). I also add a few extra common positive/negative keywords. These small heuristics keep the original logic but should raise the Jaccard score toward the target.'
- What this solution (achieved 0.24234) has done: 'I add a lightweight similarity step that, when no exact phrase is found in the tweet, selects the training phrase of the same sentiment with the highest Jaccard overlap of word tokens. This keeps the original heuristic but makes matching far less strict, which should raise the Jaccard score substantially toward the target while preserving the overall logic and output format.'
- What this solution (achieved 0.23568) has done: 'I improve the token‑matching step by normalising words (lower‑casing and stripping punctuation) so the Jaccard similarity finds more overlaps, and I add a simple fallback that returns the most frequent training phrase for the tweet’s sentiment when no better match is found. These small, targeted changes keep the original workflow while expected to raise the Jaccard score toward the target.'
- What this solution (achieved 0.59196) has done: 'I add a few targeted heuristic upgrades that stay within the original workflow but should lift the Jaccard score toward the target:  
1) Immediately return the whole tweet for neutral sentiment (the official baseline often does this).  
2) Expand the positive/negative keyword lists and enlarge the n‑gram window to capture longer sentiment‑bearing phrases.  
3) Widen the surrounding‑word window when a single keyword is found.  
These small, focused tweaks keep the core logic unchanged while giving the model a better chance to pick the correct span.'
- What this solution (achieved 0.59196) has done: 'I keep the overall workflow unchanged but add a lightweight span‑search step that, after trying the existing heuristics, looks for a contiguous window of words in the tweet whose token Jaccard similarity to any training phrase of the same sentiment is maximal. This often recovers the correct sub‑string when the exact phrase does not appear verbatim. The change is tiny, preserves the original logic, and is expected to raise the Jaccard score toward the target without introducing new models or heavy computation.'
- What this solution (achieved 0.47298) has done: 'I add a lightweight cleaning step that strips punctuation from both the tweet and candidate phrases before checking for containment. This lets the heuristic match phrases that differ only by punctuation or extra spaces, which should capture more correct spans and push the Jaccard score upward toward the target, while keeping the overall workflow unchanged.'
- What this solution (achieved 0.47366) has done: 'I enhance the heuristic by (1) expanding the window search to consider up to 15 words, and (2) when a training phrase has the highest Jaccard similarity with the whole tweet, I extract the best contiguous sub‑span of that phrase using the Jaccard window search instead of returning the full phrase. This keeps the original workflow intact while giving a higher‑precision selected text, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os
import re
from pathlib import Path
import pandas as pd
import itertools


def _normalize_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    return set(cleaned.split())


def _clean_text(text: str) -> str:
    """
    Return a lower‑cased version of `text` with punctuation replaced by spaces.
    This is used only for containment checks so that phrases that differ only
    by punctuation can still be matched.
    """
    return re.sub(r"[^a-zA-Z0-9']", " ", text.lower()).strip()


base_dir = Path.cwd() / "input" / "tweet-sentiment-extraction"
if not (base_dir / "test.csv").exists():
    base_dir = Path("../input/tweet-sentiment-extraction")
if not (base_dir / "test.csv").exists():
    raise FileNotFoundError("Test file not found in expected locations.")

test_path = base_dir / "test.csv"
train_path = base_dir / "train.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)




## === cell 1
sentiment_to_phrases = {}
for sentiment, group in train_df.groupby("sentiment"):
    sentiment_to_phrases[sentiment] = list(group["selected_text"].dropna().unique())




## === cell 2
sentiment_phrase_tokens = {}
sentiment_most_common = {}
for sentiment, phrases in sentiment_to_phrases.items():
    info = []
    freq = {}
    for ph in phrases:
        freq[ph] = freq.get(ph, 0) + 1
        tokens = _normalize_tokens(ph)
        info.append((ph, tokens))
    sentiment_phrase_tokens[sentiment] = info
    most_common = max(freq.items(), key=lambda x: x[1])[0] if freq else ""
    sentiment_most_common[sentiment] = most_common




## === cell 3
POSITIVE_KEYWORDS = {
    "good",
    "great",
    "happy",
    "love",
    "excellent",
    "fantastic",
    "awesome",
    "nice",
    "like",
    "enjoy",
    "wonderful",
    "perfect",
    "amazing",
    "delight",
    "pleased",
    "best",
    "positive",
    "sweet",
    "cool",
    "smile",
}
NEGATIVE_KEYWORDS = {
    "bad",
    "terrible",
    "hate",
    "sad",
    "awful",
    "worst",
    "poor",
    "horrible",
    "dislike",
    "unhappy",
    "disappointed",
    "angry",
    "depressed",
    "negative",
    "sick",
    "pain",
    "cry",
    "fail",
    "ugly",
}


def _best_window_by_jaccard(words: list[str], phrase_tokens: set[str]) -> str:
    """
    Return the contiguous sub‑list of `words` (as a string) that yields the highest
    Jaccard similarity with `phrase_tokens`. The window size is limited to the
    length of the phrase tokens plus a small buffer (max 15 words) to keep it fast.
    """
    if not words:
        return ""
    best_win = ""
    best_score = 0.0
    max_len = min(15, len(words))  # expanded from 12 to allow longer spans
    for win_len in range(1, max_len + 1):
        for start in range(len(words) - win_len + 1):
            window = words[start : start + win_len]
            win_tokens = _normalize_tokens(" ".join(window))
            inter = win_tokens & phrase_tokens
            union = win_tokens | phrase_tokens
            if not union:
                continue
            score = len(inter) / len(union)
            if score > best_score or (
                score == best_score and len(window) > len(best_win.split())
            ):
                best_score = score
                best_win = " ".join(window)
    return best_win


def pick_selected_text(row):
    """
    Heuristic to choose a span that best matches the sentiment.
    The original steps are kept; an extra step (after the most‑common fallback)
    searches for the best token‑Jaccard window inside the tweet.
    """
    text = str(row["text"])
    sentiment = row["sentiment"]
    lower_text = text.lower()
    cleaned_text = _clean_text(text)

    if sentiment == "neutral":
        return text

    candidates = sentiment_to_phrases.get(sentiment, [])
    best_match = ""
    for cand in candidates:
        if cand:
            cleaned_cand = _clean_text(cand)
            if (
                cleaned_cand
                and cleaned_cand in cleaned_text
                and len(cand) > len(best_match)
            ):
                best_match = cand
    if best_match:
        return best_match

    tweet_tokens = _normalize_tokens(text)
    best_phrase = ""
    best_tokens = set()
    best_score = 0.0
    for ph, ph_tokens in sentiment_phrase_tokens.get(sentiment, []):
        if not ph_tokens:
            continue
        inter = tweet_tokens & ph_tokens
        union = tweet_tokens | ph_tokens
        if not union:
            continue
        sim = len(inter) / len(union)
        if sim > best_score or (sim == best_score and len(ph) > len(best_phrase)):
            best_score = sim
            best_phrase = ph
            best_tokens = ph_tokens
    if best_score > 0:
        words = text.split()
        window = _best_window_by_jaccard(words, best_tokens)
        return window if window else best_phrase

    keywords = (
        POSITIVE_KEYWORDS
        if sentiment == "positive"
        else NEGATIVE_KEYWORDS if sentiment == "negative" else set()
    )
    words = text.split()
    longest = ""
    max_n = min(8, len(words))
    for n in range(max_n, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if any(kw in phrase.lower() for kw in keywords):
                if len(phrase) > len(longest):
                    longest = phrase
        if longest:
            break
    if longest:
        return longest

    lower_words = [w.lower() for w in words]
    for idx, w in enumerate(lower_words):
        if w in keywords:
            start = max(0, idx - 5)
            end = min(len(words), idx + 6)  # idx inclusive, +5 after
            return " ".join(words[start:end])

    most_common = sentiment_most_common.get(sentiment, "")
    if most_common:
        return most_common

    best_window = ""
    best_win_score = 0.0
    for ph, ph_tokens in sentiment_phrase_tokens.get(sentiment, []):
        window = _best_window_by_jaccard(words, ph_tokens)
        if window:
            win_score = len(_normalize_tokens(window) & ph_tokens) / len(
                _normalize_tokens(window) | ph_tokens
            )
            if win_score > best_win_score or (
                win_score == best_win_score and len(window) > len(best_window)
            ):
                best_win_score = win_score
                best_window = window
    if best_window:
        return best_window

    return text


test_df["selected_text"] = test_df.apply(pick_selected_text, axis=1)




## === cell 4
submission_path = Path("submission.csv")
test_df[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
