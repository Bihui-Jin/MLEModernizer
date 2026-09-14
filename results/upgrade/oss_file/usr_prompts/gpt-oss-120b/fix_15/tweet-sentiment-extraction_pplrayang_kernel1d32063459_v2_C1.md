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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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

0.7155378460884094

# 6. Current score

0.59762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60245) has done: 'The fix adds the missing `csv` import and correctly references `csv.QUOTE_ALL` when saving the submission. The heuristic predictor is enhanced to return a short phrase (the matched sentiment word plus its neighboring words) instead of a single token, which should improve the Jaccard score while keeping the original simple rule‑based approach.'
- What this solution (achieved 0.60161) has done: 'I slightly expand the heuristic window so the extracted phrase includes two words before and after the sentiment‑indicative token (instead of just one). This modest change keeps the original rule‑based approach but should capture more of the true selected text, improving the Jaccard score toward the target while preserving all other logic.'
- What this solution (achieved 0.60513) has done: 'I tighten the rule‑based extractor: instead of picking a fixed two‑word window around the first sentiment word, I locate all sentiment‑indicative words in the tweet, take the span that covers them, and then optionally expand it by one word on each side. This keeps the same simple heuristic but captures more of the true selected phrase, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.60537) has done: 'I slightly extend the heuristic extractor to also swallow simple intensity or negation words that appear immediately before or after the sentiment‑indicative token(s). By widening the span when a preceding or following word belongs to a small qualifier set (e.g., “very”, “not”, “so”), the predicted phrase more often matches the true selected text, which should raise the Jaccard score toward the target while keeping the original rule‑based logic unchanged.'
- What this solution (achieved 0.60042) has done: 'I enhance the heuristic extractor by expanding the selected phrase outward until a punctuation mark is reached (or the tweet boundaries), in addition to the existing qualifier‑based widening. This should capture more of the true sentiment span and raise the Jaccard score toward the target while keeping the overall rule‑based approach unchanged.'
- What this solution (achieved 0.60327) has done: 'I tighten the heuristic extractor so it expands only a limited number of words around the sentiment‑indicative token(s) (plus any qualifier words) instead of swallowing the whole tweet. This should reduce over‑prediction and raise the Jaccard score toward the target while keeping the overall rule‑based approach unchanged.'
- What this solution (achieved 0.6026) has done: 'I tighten the heuristic extractor by (1) allowing the window to capture a trailing punctuation token (so the predicted phrase ends with the original punctuation when appropriate) and (2) increasing the maximum extra token window from 3 to 4 words on each side. These small adjustments keep the overall rule‑based approach while expectedly raising the Jaccard overlap toward the target score.'
- What this solution (achieved 0.60181) has done: 'I expand the sentiment word lists, increase the allowable window size, and adjust the expansion logic so it walks outward until a punctuation mark is hit (or the max window is reached). This keeps the original rule‑based approach but captures a more complete sentiment phrase, which should raise the Jaccard score toward the target while preserving all existing functionality.'
- What this solution (achieved 0.60223) has done: 'I slightly broaden the sentiment word lists and increase the allowable expansion window (up to 12 tokens) when surrounding the sentiment‑indicative token(s). This keeps the original rule‑based extractor intact while giving it a better chance to capture the true selected phrase, which should raise the Jaccard score toward the target. No core logic or model architecture is changed, and the script still writes a proper submission.csv file.'
- What this solution (achieved 0.60275) has done: 'I reduce the maximum extra expansion window from 12 tokens to 6 tokens so the heuristic does not over‑extend the predicted span. This small tweak keeps the original rule‑based logic but limits unnecessary word additions, which is expected to raise the Jaccard score toward the target while preserving all existing functionality.'
- What this solution (achieved 0.60396) has done: 'I replace the word‑based window logic with a character‑based span extractor that expands to the nearest punctuation on each side of the first and last sentiment‑indicative word. This keeps the overall heuristic approach while aligning the predicted phrase more closely with the true selected text, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.59762) has done: 'The update enhances the heuristic extractor by (1) pulling in surrounding qualifier words (e.g., “very”, “not”) when they appear immediately before or after the sentiment‑indicative token(s) and (2) limiting the final extracted span to a reasonable maximum of eight words to avoid over‑prediction. These small, targeted tweaks keep the original rule‑based approach while expectedly raising the Jaccard overlap and moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os, random, re, string, warnings, csv
import numpy as np, pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()



## === cell 1
test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"



## === cell 2
POS_WORDS = {
    "good",
    "great",
    "awesome",
    "excellent",
    "nice",
    "love",
    "perfect",
    "fantastic",
    "amazing",
    "best",
    "happy",
    "wonderful",
    "delightful",
    "pleasant",
    "lovely",
    "satisfied",
    "joyful",
    "liked",
    "enjoy",
    "like",
    "cool",
    "superb",
    "pleased",
    "grateful",
    "delighted",
    "loving",
}
NEG_WORDS = {
    "bad",
    "worst",
    "terrible",
    "awful",
    "hate",
    "poor",
    "sad",
    "disappointed",
    "horrible",
    "negative",
    "unsatisfied",
    "ugh",
    "angry",
    "upset",
    "annoyed",
    "dislike",
    "mad",
    "horrendous",
    "dreadful",
    "disgusting",
    "unhappy",
    "miserable",
    "sick",
    "painful",
}
QUALIFIERS = {
    "very",
    "so",
    "extremely",
    "quite",
    "really",
    "not",
    "absolutely",
    "pretty",
    "barely",
    "hardly",
}

POS_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, POS_WORDS)) + r")\b", flags=re.IGNORECASE
)
NEG_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, NEG_WORDS)) + r")\b", flags=re.IGNORECASE
)

PUNCT_SET = {".", "!", "?", ",", ";", ":"}


def heuristic_predict(row):
    """Extract a span that contains sentiment‑indicative words, expand outward
    to the nearest punctuation, pull in adjacent qualifier words, and cap the
    output length to a reasonable number of tokens."""
    tweet = str(row["text"])
    sentiment = str(row["sentiment"]).lower()
    if sentiment == "neutral":
        return tweet

    pattern = POS_PATTERN if sentiment == "positive" else NEG_PATTERN
    matches = list(pattern.finditer(tweet))
    if not matches:
        return tweet

    start = matches[0].start()
    end = matches[-1].end()

    while start > 0 and tweet[start - 1] not in PUNCT_SET:
        start -= 1
    if start > 0 and tweet[start - 1] in PUNCT_SET:
        start -= 1

    while end < len(tweet) and tweet[end] not in PUNCT_SET:
        end += 1
    if end < len(tweet) and tweet[end] in PUNCT_SET:
        end += 1

    left_boundary = tweet[:start]
    last_word_match = None
    for m in re.finditer(r"\b\w+\b", left_boundary):
        last_word_match = m
    if last_word_match:
        word = last_word_match.group(0)
        if word.lower() in QUALIFIERS:
            start = last_word_match.start()

    right_boundary = tweet[end:]
    first_word_match = re.search(r"\b\w+\b", right_boundary)
    if first_word_match:
        word = first_word_match.group(0)
        if word.lower() in QUALIFIERS:
            end = end + first_word_match.end()

    span = tweet[start:end].strip()

    MAX_WORDS = 8
    tokens = span.split()
    if len(tokens) > MAX_WORDS:
        span = " ".join(tokens[:MAX_WORDS])

    return span




## === cell 3
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

predictions = test_df.apply(heuristic_predict, axis=1).tolist()



## === cell 4
sub_df = pd.read_csv(submission_template)

sub_df["selected_text"] = predictions

sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

sub_df.to_csv("submission.csv", index=False, quoting=csv.QUOTE_ALL)
print("Submission saved to submission.csv")
