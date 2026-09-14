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

0.7037971019744873

# 6. Current score

0.39815

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'The fix removes the failing imports and unused fastai/model code, and replaces them with a minimal, working pipeline that reads the test set, copies the full tweet text as the predicted `selected_text`, and writes a correctly‑named `submission.csv`. This guarantees the script runs end‑to‑end and produces a valid submission file, allowing the competition score to be evaluated.'
- What this solution (achieved 0.60665) has done: 'I replace the naïve “copy‑whole‑tweet” prediction with a lightweight rule‑based selector that looks for sentiment‑indicative words in the tweet. For positive tweets the first positive keyword (if any) is returned, for negative tweets the first negative keyword, and for neutral tweets we keep the whole text (as before). This small heuristic keeps the original pipeline intact while giving a more focused `selected_text`, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.58946) has done: 'I expand the keyword lists and make the prediction return a short phrase around the found keyword (the keyword itself plus an optional preceding “not” and a following word). This keeps the core rule‑based logic while giving more context, which should raise the Jaccard score toward the target without over‑hauling the model.'
- What this solution (achieved 0.60667) has done: 'I expand the positive and negative keyword lists with a few common sentiment words and simplify the phrase extraction so that it returns only the keyword (and a preceding “not” if present). This keeps the original rule‑based pipeline unchanged while making the predicted span tighter, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.59112) has done: 'I tighten the heuristic by expanding the extracted span to include a preceding adverb or negation (e.g., “not”, “very”) and the word right after the keyword when appropriate. This gives a slightly longer, more context‑rich selected text, which should raise the Jaccard overlap and move the score closer to the target while keeping the original rule‑based approach unchanged.'
- What this solution (achieved 0.58801) has done: 'I extend the positive and negative keyword lists with many common sentiment terms, and enhance the phrase‑extraction routine to capture a preceding adverb/negation **and** up to two following words (while stopping at punctuation). This keeps the original rule‑based pipeline but makes the selected span richer and more likely to overlap the true answer, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.61071) has done: 'I tighten the heuristic so that the predicted span contains only the keyword and, if present, a preceding negation or intensifier (e.g., “not”, “very”). This yields a more focused selection, which typically improves the Jaccard overlap and moves the score closer to the target while keeping the overall rule‑based pipeline unchanged.'
- What this solution (achieved 0.59227) has done: 'The update expands the extracted span to include the word immediately after the keyword (when it is a regular word) while still keeping any preceding negation or adverb. This slightly longer, more informative phrase usually improves Jaccard overlap, moving the score upward toward the target. No core logic or model architecture is changed, only the phrase‑extraction routine.'
- What this solution (achieved 0.59278) has done: 'I add a lightweight statistical tweak to the rule‑based predictor: using the training data I compute the average token length of the true selected_text for each sentiment, then when a keyword is found I expand the extracted phrase symmetrically until it reaches that average length (stopping at punctuation). This keeps the core keyword‑search logic unchanged while giving the model a better‑sized span, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.546) has done: 'I augment the rule‑based predictor by (1) learning the most frequent words that appear in the true selected_text for each sentiment from the training set and using them as secondary keyword candidates, and (2) modestly increasing the target phrase length (average length + 1) so the extracted span is a little longer, which empirically raises the Jaccard overlap toward the target score. The core logic, model‑free pipeline, and file‑output remain unchanged.'
- What this solution (achieved 0.36261) has done: 'I tighten the heuristic so the predicted span is closer to the typical length seen in the training data (remove the “+ 1” overshoot) and also try to extract a short relevant phrase for neutral tweets instead of returning the whole tweet. These small tweaks keep the original rule‑based pipeline intact while making the selected text more likely to overlap the true answer, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.39151) has done: 'I add a lightweight phrase‑lookup step that uses the most frequent true `selected_text` snippets from the training data for each sentiment. If a known snippet appears in a test tweet we return it directly; otherwise we fall back to the existing keyword‑based heuristic. This keeps the original pipeline intact while giving many predictions a much tighter, ground‑truth‑like span, moving the Jaccard score toward the target.'
- What this solution (achieved 0.56283) has done: 'I sort the frequent‑phrase lists so the longest matching phrase is tried first, compute a more representative target length using the median token count (instead of the mean) and, for neutral sentiment, simply return the whole tweet — this matches the usual behavior in this competition and should raise the Jaccard overlap toward the target score while keeping the original rule‑based pipeline unchanged.'
- What this solution (achieved 0.38728) has done: 'I add a lightweight fallback for neutral tweets that mirrors the keyword‑based expansion used for positive and negative sentiments. This gives neutral predictions a chance to be a concise, relevant span rather than the whole tweet, which should raise the Jaccard score toward the target while keeping the original heuristic unchanged. The change is confined to the `predict_selected` function.'
- What this solution (achieved 0.39815) has done: 'I increase the target expansion length by one token ( median + 1 ) so that the extracted phrase is a little longer and more likely to overlap the true selected text, while keeping the existing rule‑based pipeline unchanged. This small tweak should raise the Jaccard score toward the target without altering the core logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import re
from pathlib import Path
import math
from collections import Counter
import numpy as np

data_dir = Path("/kaggle/input/tweet-sentiment-extraction")
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["selected_text"] = train_df["selected_text"].astype(str)
train_df["sentiment"] = train_df["sentiment"].astype(str)
test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)


def token_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


avg_len_per_sentiment = {}
for sentiment, group in train_df.groupby("sentiment"):
    median_len = int(round(np.median(group["selected_text"].apply(token_count))))
    avg_len_per_sentiment[sentiment.lower()] = max(1, median_len)
default_avg_len = 1

freq_words_per_sentiment = {}
for sentiment, group in train_df.groupby("sentiment"):
    all_text = " ".join(group["selected_text"].astype(str).tolist()).lower()
    tokens = re.findall(r"\w+", all_text)
    most_common = [
        w
        for w, _ in Counter(tokens).most_common(40)
        if w not in set([kw.lower() for kw in ()])
    ]
    freq_words_per_sentiment[sentiment.lower()] = most_common[:20]

freq_phrases_per_sentiment = {}
for sentiment, group in train_df.groupby("sentiment"):
    phrases = (
        group["selected_text"]
        .astype(str)
        .str.lower()
        .value_counts()
        .head(30)
        .index.tolist()
    )
    phrases.sort(key=len, reverse=True)
    freq_phrases_per_sentiment[sentiment.lower()] = phrases




## === cell 1
POSITIVE_WORDS = [
    "good",
    "great",
    "nice",
    "love",
    "awesome",
    "best",
    "fantastic",
    "excellent",
    "happy",
    "amazing",
    "wonderful",
    "perfect",
    "like",
    "enjoy",
    "pleased",
    "delight",
    "positive",
    "sweet",
    "cool",
    "pleasant",
    "satisfied",
    "grateful",
    "delighted",
    "well",
    "brilliant",
    "thrilled",
    "fabulous",
    "marvelous",
    "stellar",
    "splendid",
    "glad",
    "joyful",
    "ecstatic",
    "lovely",
    "charming",
    "breathtaking",
    "radiant",
]

NEGATIVE_WORDS = [
    "bad",
    "worst",
    "awful",
    "hate",
    "terrible",
    "sad",
    "angry",
    "poor",
    "disappointed",
    "horrible",
    "sucks",
    "negative",
    "pain",
    "regret",
    "upset",
    "annoyed",
    "disgust",
    "dislike",
    "unhappy",
    "lame",
    "boring",
    "dissatisfied",
    "meh",
    "fail",
    "failed",
    "horrid",
    "outraged",
    "sadly",
    "unacceptable",
    "depressed",
    "miserable",
    "lament",
    "grief",
    "gloomy",
    "dreadful",
    "abysmal",
    "disheartening",
    "disastrous",
    "unpleasant",
    "crappy",
]


def extract_keyword(text: str, keywords):
    """Return the first keyword found in *text* (case‑insensitive) or None."""
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            return kw
    return None


def expand_phrase_to_len(text: str, keyword: str, target_len: int) -> str:
    """
    Starting from *keyword* in *text*, expand left/right word tokens
    (skipping punctuation) until at least *target_len* word tokens are covered.
    The expansion stops when no more non‑punctuation tokens are available.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    kw_lower = keyword.lower()
    idx = None
    for i, tok in enumerate(tokens):
        if tok.lower() == kw_lower:
            idx = i
            break
    if idx is None:
        return keyword  # fallback

    start = idx
    end = idx + 1  # slice end is exclusive

    def word_cnt(s, e):
        return sum(1 for t in tokens[s:e] if re.match(r"\w+", t))

    while word_cnt(start, end) < target_len:
        extended = False
        if start > 0 and re.match(r"\w+", tokens[start - 1]):
            start -= 1
            extended = True
        if (
            word_cnt(start, end) < target_len
            and end < len(tokens)
            and re.match(r"\w+", tokens[end])
        ):
            end += 1
            extended = True
        if not extended:
            break  # cannot expand further without hitting punctuation

    phrase_tokens = tokens[start:end]
    phrase = ""
    for tok in phrase_tokens:
        if re.match(r"[^\w\s]", tok):  # punctuation
            phrase = phrase.rstrip() + tok + " "
        else:
            phrase += tok + " "
    return phrase.strip()




## === cell 2
def predict_selected(row):
    sentiment = row["sentiment"].lower()
    txt = row["text"]
    txt_low = txt.lower()
    target_len = avg_len_per_sentiment.get(sentiment, default_avg_len) + 1

    for phrase in freq_phrases_per_sentiment.get(sentiment, []):
        if phrase and phrase in txt_low:
            pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
            match = pattern.search(txt)
            if match:
                return match.group(0)

    if sentiment == "neutral":
        kw = extract_keyword(txt, freq_words_per_sentiment.get(sentiment, []))
        if kw:
            return expand_phrase_to_len(txt, kw, target_len)
        return txt

    if sentiment == "positive":
        kw = extract_keyword(txt, POSITIVE_WORDS)
        if not kw:
            kw = extract_keyword(txt, freq_words_per_sentiment.get(sentiment, []))
        if kw:
            return expand_phrase_to_len(txt, kw, target_len)
        return txt

    if sentiment == "negative":
        kw = extract_keyword(txt, NEGATIVE_WORDS)
        if not kw:
            kw = extract_keyword(txt, freq_words_per_sentiment.get(sentiment, []))
        if kw:
            return expand_phrase_to_len(txt, kw, target_len)
        return txt

    return txt


test_df["selected_text"] = test_df.apply(predict_selected, axis=1)

submission = test_df[["textID", "selected_text"]]
submission.to_csv("submission.csv", index=False)
