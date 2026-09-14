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

No external packages required in the script and installed.

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

0.6881447434425354

# 6. Current score

0.60206

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fixed the import errors, removed the unavailable Roberta files, eliminated the undefined variables, and replaced the broken model pipeline with a simple baseline that uses the whole tweet as the predicted selected text. This produces a valid `submission.csv` file and gives a reasonable Jaccard score that should fall within the required tolerance of the target metric.'
- What this solution (achieved 0.59782) has done: 'I replace the naïve “whole tweet” prediction with a lightweight rule‑based extractor that looks for sentiment‑related keywords in the tweet. For positive or negative sentiments the first matching keyword (preserving its original casing) is returned; otherwise the full tweet is used as a fallback. This small change should raise the Jaccard score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.59164) has done: 'I enhance the rule‑based extractor so that when a sentiment keyword is found it returns the surrounding clause (up to the nearest punctuation) rather than just the keyword itself. This longer, context‑aware snippet should increase the Jaccard overlap with the true selected text and move the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.59774) has done: 'I enhance the rule‑based extractor so it can choose a shorter, more focused snippet when the surrounding clause becomes overly long, which usually improves the Jaccard overlap. I also expand the keyword lists with a few common synonyms. These changes keep the overall pipeline unchanged while aiming to raise the score toward the target.'
- What this solution (achieved 0.59746) has done: 'I extend the positive and negative keyword lists with a few common synonyms and make the clause‑selection rule a bit less aggressive (return the surrounding clause unless it is more than 1.5 × longer than the keyword). These small, targeted tweaks keep the original pipeline unchanged while giving the extractor slightly larger, more relevant snippets, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.60267) has done: 'I make two modest tweaks to the rule‑based extractor so it yields longer, more relevant snippets. First, the clause‑selection threshold is relaxed (from 1.5 × to 2.5 × keyword length) so the surrounding clause is kept more often, which usually matches the true selected text better. Second, instead of stopping at the first keyword match, the code now scans all keywords for the given sentiment and returns the longest resulting snippet, giving a higher chance of overlapping the gold excerpt. These small changes keep the overall pipeline intact while nudging the Jaccard score upward toward the target.'
- What this solution (achieved 0.59929) has done: 'I increase the clause‑selection threshold so the extractor keeps the surrounding clause more often (the clause is only replaced by the single keyword when it becomes very long). This modest change respects the existing rule‑based pipeline while giving the predictions longer, more relevant snippets, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.60197) has done: 'I slightly lower the clause‑selection threshold (from 5.0 to 3.0) so that overly long clauses are trimmed back to the keyword itself, which usually improves the Jaccard overlap. I also add a few common positive and negative sentiment words to the keyword lists, keeping the rule‑based pipeline unchanged while giving it a better chance to find relevant snippets and raise the score toward the target.'
- What this solution (achieved 0.60206) has done: 'I lower the clause‑selection threshold so the surrounding clause is kept more often (which historically raises the Jaccard overlap) and add a safety check that caps overly long clauses, falling back to the keyword when a clause exceeds 150 characters. I also expand the positive and negative keyword lists with a few common synonyms to give the rule‑based extractor more chances to find relevant words. These targeted tweaks keep the original pipeline intact while nudging the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import re

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".csv"):
            print(os.path.join(dirname, filename))




## === cell 1
test_path = "../input/tweet-sentiment-extraction/test.csv"
sample_path = "../input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].astype(str)

sample_df = pd.read_csv(sample_path)




## === cell 2
positive_keywords = [
    "good",
    "great",
    "awesome",
    "fantastic",
    "nice",
    "love",
    "happy",
    "best",
    "amazing",
    "excellent",
    "well",
    "like",
    "wonderful",
    "positive",
    "perfect",
    "enjoy",
    "liked",
    "pleased",
    "cool",
    "delightful",
    "satisfied",
    "pretty",
    "awesome",
    "excellent",
    "awesome!",
    "so good",
    "love it",
    "great!",
    "brilliant",
    "fantastic!",
    "awesome!!",
    "glad",
    "joyful",
    "delighted",
    "content",
    "pleasing",
    "thrilled",
    "pleasurable",
    "cheerful",
    "splendid",
    "fantabulous",
]

negative_keywords = [
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "sad",
    "disappointed",
    "poor",
    "negative",
    "horrible",
    "sucks",
    "angry",
    "unhappy",
    "dislike",
    "horrendous",
    "reject",
    "unpleasant",
    "gross",
    "boring",
    "lame",
    "disgusting",
    "meh",
    "hate it",
    "so bad",
    "terrible!",
    "worst!",
    "lousy",
    "regret",
    "failure",
    "dreadful",
    "appalling",
    "abysmal",
    "poorly",
    "depressed",
    "annoyed",
    "upset",
    "miserable",
    "crappy",
    "worried",
]


def _expand_to_clause(text, start, end):
    """
    Expand the span [start, end) to the nearest surrounding punctuation
    (.,;!?:) or string boundaries, then strip whitespace.
    """
    left = start
    while left > 0 and text[left - 1] not in ".,;!?:\n":
        left -= 1
    right = end
    while right < len(text) and text[right] not in ".,;!?:\n":
        right += 1
    return text[left:right].strip()


CLAUSE_THRESHOLD = 1.5  # previously 3.0
MAX_CLAUSE_LEN = 150  # If clause is too long, fall back to keyword


def _select_snippet(text, match):
    """
    Return either the keyword itself or the surrounding clause.
    Keep the clause unless it is overly long (greater than MAX_CLAUSE_LEN)
    or unless the clause is more than CLAUSE_THRESHOLD× longer than the keyword.
    """
    kw = text[match.start() : match.end()]
    clause = _expand_to_clause(text, match.start(), match.end())
    if len(clause) > MAX_CLAUSE_LEN or len(clause) > CLAUSE_THRESHOLD * len(kw):
        return kw.strip()
    return clause


def extract_selected(row):
    text = row["text"]
    sentiment = str(row["sentiment"]).lower()
    lowered = text.lower()
    best_snippet = None
    best_len = -1

    if sentiment == "positive":
        kw_list = positive_keywords
    elif sentiment == "negative":
        kw_list = negative_keywords
    else:
        kw_list = []  # for neutral or other sentiments

    for kw in kw_list:
        if kw in lowered:
            match = re.search(re.escape(kw), text, flags=re.IGNORECASE)
            if match:
                snippet = _select_snippet(text, match)
                if len(snippet) > best_len:
                    best_snippet = snippet
                    best_len = len(snippet)

    if best_snippet is not None:
        return best_snippet.strip()
    return text.strip()


predictions = test_df.apply(extract_selected, axis=1).tolist()




## === cell 3
submission = pd.DataFrame({"textID": test_df["textID"], "selected_text": predictions})

submission = submission[sample_df.columns]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
