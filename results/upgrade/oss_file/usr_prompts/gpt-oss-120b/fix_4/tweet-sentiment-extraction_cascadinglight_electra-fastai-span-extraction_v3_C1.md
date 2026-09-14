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

0.58946

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'The fix removes the failing imports and unused fastai/model code, and replaces them with a minimal, working pipeline that reads the test set, copies the full tweet text as the predicted `selected_text`, and writes a correctly‑named `submission.csv`. This guarantees the script runs end‑to‑end and produces a valid submission file, allowing the competition score to be evaluated.'
- What this solution (achieved 0.60665) has done: 'I replace the naïve “copy‑whole‑tweet” prediction with a lightweight rule‑based selector that looks for sentiment‑indicative words in the tweet. For positive tweets the first positive keyword (if any) is returned, for negative tweets the first negative keyword, and for neutral tweets we keep the whole text (as before). This small heuristic keeps the original pipeline intact while giving a more focused `selected_text`, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.58946) has done: 'I expand the keyword lists and make the prediction return a short phrase around the found keyword (the keyword itself plus an optional preceding “not” and a following word). This keeps the core rule‑based logic while giving more context, which should raise the Jaccard score toward the target without over‑hauling the model.'

# 9. Code solution

## === cell 0
import pandas as pd
import re
from pathlib import Path




## === cell 1
data_dir = Path("/kaggle/input/tweet-sentiment-extraction")
test_path = data_dir / "test.csv"

test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)




## === cell 2
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
]


def extract_keyword(text: str, keywords):
    """Return the first keyword found in *text* (case‑insensitive) or None."""
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            return kw
    return None


def extract_phrase_around_keyword(text: str, keyword: str):
    """
    Return a short phrase containing *keyword*:
    - Include a preceding 'not' if present.
    - Include the following word if it is not punctuation.
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
    if idx > 0 and tokens[idx - 1].lower() == "not":
        start = idx - 1

    end = idx + 1
    if end < len(tokens) and re.match(r"\w+", tokens[end]):
        end += 1

    phrase = " ".join(tokens[start:end])
    return phrase


def predict_selected(row):
    sentiment = row["sentiment"].lower()
    txt = row["text"]
    if sentiment == "positive":
        kw = extract_keyword(txt, POSITIVE_WORDS)
        if kw is not None:
            return extract_phrase_around_keyword(txt, kw)
        else:
            return txt
    elif sentiment == "negative":
        kw = extract_keyword(txt, NEGATIVE_WORDS)
        if kw is not None:
            return extract_phrase_around_keyword(txt, kw)
        else:
            return txt
    else:  # neutral or any other label
        return txt


test_df["selected_text"] = test_df.apply(predict_selected, axis=1)

submission = test_df[["textID", "selected_text"]]
submission.to_csv("submission.csv", index=False)
