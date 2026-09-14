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

0.727962076663971

# 6. Current score

0.57806

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59357) has done: 'I replace the missing utility imports and the failing model code with a lightweight, rule‑based approach that extracts the sentiment word (or the whole tweet for neutral sentiment). This fixes the import and path errors, guarantees a `.csv` submission file, and provides a reasonable Jaccard score that should move the result toward the target without altering the core modeling logic of the original notebook.'
- What this solution (achieved 0.59429) has done: 'I replace the simple “first‑occurrence‑of‑sentiment‑word” heuristic with a lightweight sentiment‑lexicon rule: for positive and negative tweets the code looks for any known positive/negative word in the tweet and returns the first matching word (or the whole tweet if none are found). This keeps the overall structure unchanged while providing a more relevant selected text, which should raise the Jaccard score toward the target. The rest of the pipeline—loading data, applying the function, and writing the CSV—remains the same.'
- What this solution (achieved 0.61084) has done: 'I enhance the heuristic by expanding the matched sentiment word to include the surrounding word(s) — so phrases like “very good” or “not good” are captured instead of only the single token. This modest change keeps the original rule‑based structure while providing richer selected text, which should raise the Jaccard score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.57972) has done: 'I keep the overall rule‑based pipeline unchanged but improve the phrase‑expansion heuristic: after locating a sentiment word, the code now also pulls surrounding intensifier tokens (e.g., “very”, “really”) and includes up to two following tokens, stopping before punctuation. This gives richer selected‑text spans such as “very good” or “not happy”, which should raise the Jaccard score toward the target while preserving the original logic.'
- What this solution (achieved 0.57806) has done: 'I expand the heuristic slightly to capture more sentiment context: add a few common sentiment words, treat “not” as an intensifier so negations like “not good” are selected, and allow the right‑hand expansion to include up to three following tokens (still stopping at punctuation). These modest changes keep the rule‑based core intact while improving the Jaccard overlap, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import re
import pandas as pd

POSITIVE_WORDS = {
    "good",
    "great",
    "wonderful",
    "amazing",
    "nice",
    "love",
    "excellent",
    "best",
    "fantastic",
    "happy",
    "awesome",
    "like",
    "enjoy",
    "well",
    "cool",
    "delicious",
    "perfect",
    "sweet",
    "joyful",
    "pleased",
    "positive",
    "fabulous",
    "okay",
    "ok",
    "fine",
    "satisfied",
    "liked",
    "lovely",
    "brilliant",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "horrible",
    "awful",
    "worst",
    "hate",
    "disappointed",
    "sad",
    "angry",
    "annoyed",
    "poor",
    "disgusting",
    "sucks",
    "sick",
    "pain",
    "problem",
    "negative",
    "unhappy",
    "not",
    "no",
    "hard",
    "hardly",
    "fail",
    "failed",
    "failure",
    "depress",
    "lonely",
    "dislike",
    "poorly",
    "hated",
    "unpleasant",
    "mistake",
}

INTENSIFIERS = {
    "very",
    "extremely",
    "so",
    "really",
    "quite",
    "too",
    "absolutely",
    "super",
    "almost",
    "barely",
    "hardly",
    "somewhat",
    "rather",
    "not",
}


## === cell 1
test_path = "../input/tweet-sentiment-extraction/test.csv"
test = pd.read_csv(test_path)




## === cell 2
def extract_selected_text(text: str, sentiment: str) -> str:
    """
    Heuristic with modest phrase expansion:
    - Neutral → whole tweet.
    - Positive/Negative → first lexicon match (case‑insensitive).
      Expand left to include preceding intensifiers or a preceding “not”.
      Expand right to include up to three following tokens,
      stopping before punctuation. If no match → whole tweet.
    """
    if sentiment.lower() == "neutral":
        return text.strip()

    tokens = list(re.finditer(r"\b\w+\b", text))
    lexicon = POSITIVE_WORDS if sentiment.lower() == "positive" else NEGATIVE_WORDS

    for idx, match in enumerate(tokens):
        word = match.group().lower()
        if word in lexicon:
            start_idx = idx
            while start_idx > 0:
                prev_word = tokens[start_idx - 1].group().lower()
                if prev_word in INTENSIFIERS:
                    start_idx -= 1
                else:
                    break

            end_idx = idx
            while (end_idx + 1) < len(tokens) and (end_idx - idx) < 3:
                next_char_pos = tokens[end_idx].end()
                if next_char_pos < len(text) and text[next_char_pos] in ",.!?;:":
                    break
                end_idx += 1

            expanded_start = tokens[start_idx].start()
            expanded_end = tokens[end_idx].end()
            return text[expanded_start:expanded_end].strip()

    return text.strip()




## === cell 3
test["selected_text"] = test.apply(
    lambda row: extract_selected_text(row["text"], row["sentiment"]), axis=1
)


## === cell 4
submission_path = "submission.csv"
test[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
