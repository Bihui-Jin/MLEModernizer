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

0.59995

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61626) has done: 'I fixed the import errors, removed the unusable fastai components, and replaced the whole training pipeline with a lightweight rule‑based predictor that runs end‑to‑end. The script now loads the data, applies a simple sentiment‑aware heuristic to choose a span, and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.59995) has done: 'I keep the overall structure but improve the span‑extraction heuristic: instead of returning just the first sentiment word, the new rule expands to a surrounding phrase up to punctuation marks. This modest change keeps the original logic while providing longer, more context‑rich predictions, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path




## === cell 1
base_dir = Path("/kaggle/input/tweet-sentiment-extraction")

train_df = pd.read_csv(base_dir / "train.csv")
test_df = pd.read_csv(base_dir / "test.csv")




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


def _extract_phrase(text, target_set):
    """Return a short phrase containing the first matching sentiment word.

    The phrase is expanded left/right until a punctuation token is hit,
    providing a more complete sentiment snippet than a single word.
    """
    tokens = text.split()
    for i, tok in enumerate(tokens):
        cleaned = tok.lower().strip(".,!?\"'")
        if cleaned in target_set:
            l = i
            while l > 0 and not any(p in tokens[l - 1] for p in ".,!?"):
                l -= 1
            r = i
            while r + 1 < len(tokens) and not any(p in tokens[r + 1] for p in ".,!?"):
                r += 1
            return " ".join(tokens[l : r + 1])
    return text


def predict_span(row):
    text = row["text"]
    sentiment = row["sentiment"].lower()

    if sentiment == "neutral" or len(text) < 3:
        return text

    if sentiment == "positive":
        return _extract_phrase(text, positive_words)
    elif sentiment == "negative":
        return _extract_phrase(text, negative_words)
    else:
        return text




## === cell 3
test_df["selected_text"] = test_df.apply(predict_span, axis=1)




## === cell 4
submission = test_df[["textID", "selected_text"]].copy()
submission.to_csv("submission.csv", index=False)
