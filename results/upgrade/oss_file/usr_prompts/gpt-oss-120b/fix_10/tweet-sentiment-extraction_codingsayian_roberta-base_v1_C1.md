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

3.10

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

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

0.7191751003265381

# 6. Current score

0.58503

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61768) has done: 'I replace the failing transformer loading and downstream processing with a lightweight, rule‑based prediction that works entirely offline. The new code keeps the original data loading, defines a simple function that selects a sentiment‑related word (or the whole tweet if none is found), fills the `selected_text` column, and writes a valid `submission.csv`. This removes the HF validation errors and ensures the notebook runs end‑to‑end while producing a properly formatted submission file.'
- What this solution (achieved 0.58666) has done: 'I expand the sentiment word lists and replace the single‑word extraction with a small regex‑based phrase extraction: after finding a sentiment word, the code now returns the substring from that word up to the next punctuation (or the end of the tweet). This modest heuristic usually captures a larger, more relevant portion of the tweet and should raise the Jaccard score toward the target while keeping the original simple rule‑based structure unchanged.'
- What this solution (achieved 0.58694) has done: 'I expand the sentiment word lists slightly, add a small set of preceding modifiers (e.g., “not”, “very”), and revise `select_text` to consider **all** occurrences of sentiment words, optionally include a preceding modifier, and choose the longest resulting phrase up to the next punctuation. This keeps the rule‑based core unchanged while giving a richer extracted substring, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.5859) has done: 'I expand the sentiment word lists with more common positive and negative terms and broaden the characters that end a phrase (adding `;`, `:`, `-`, `—`). This keeps the rule‑based core unchanged but lets the extractor capture longer, more relevant substrings, which should raise the Jaccard score toward the target. The rest of the pipeline remains identical, and the script still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.58952) has done: 'I expand the sentiment word lists slightly and improve the phrase‑boundary logic: instead of stopping only at the first punctuation after the sentiment word, the extractor now looks back to the previous punctuation (or start) and forward to the next punctuation (or end) so the whole meaningful clause is captured. This modestly widens the extracted text, which should raise the Jaccard score toward the target while keeping the overall rule‑based approach unchanged.'
- What this solution (achieved 0.58434) has done: 'I expand the word and modifier lists used by the rule‑based extractor so it can capture more sentiment‑related phrases (e.g., common contractions, informal words, and emojis). This slightly broadens the candidate substrings while keeping the original logic unchanged, which should move the Jaccard score closer to the target without altering the core pipeline.'
- What this solution (achieved 0.58488) has done: 'I modify the `select_text` function so it picks the shortest matching phrase (instead of the longest) because the gold “selected_text” is usually the minimal relevant snippet, which should raise the Jaccard score toward the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved 0.58434) has done: 'I adjust the rule‑based extractor to choose the **longest** sentiment‑containing phrase (instead of the shortest) bounded by punctuation, which historically raises the Jaccard score toward the target region. The change is limited to the comparison logic inside `select_text`, keeping all other data handling and output steps unchanged, so the script still runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.58503) has done: 'I load the training data to compute the median length of the gold `selected_text` strings and use that statistic to pick, among all candidate phrases that contain a sentiment word, the one whose length is closest to the median (instead of always taking the longest). This small calibration keeps the rule‑based core unchanged while aligning the extracted snippets more closely with typical gold snippets, which should move the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_path = "../input/tweet-sentiment-extraction/train.csv"
df_train = pd.read_csv(train_path)
MEDIAN_SEL_LEN = int(df_train["selected_text"].str.len().median())



## === cell 1
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")



## === cell 2
import re
import string

positive_words = {
    "good",
    "great",
    "love",
    "awesome",
    "best",
    "fantastic",
    "nice",
    "perfect",
    "happy",
    "excellent",
    "wonderful",
    "amazing",
    "cool",
    "liked",
    "enjoy",
    "enjoyed",
    "pleasant",
    "delight",
    "sweet",
    "brilliant",
    "like",
    "better",
    "well",
    "fine",
    "positive",
    "superb",
    "glad",
    "joyful",
    "delighted",
    "satisfied",
    "thrilled",
    "fabulous",
    "marvelous",
    "okay",
    "ok",
    "pleased",
    "content",
    "charming",
    "lovely",
    "lol",
    "haha",
    "yay",
    "yes",
    "😊",
    "😍",
    "😁",
}
negative_words = {
    "bad",
    "hate",
    "worst",
    "terrible",
    "awful",
    "sad",
    "poor",
    "disappointed",
    "angry",
    "negative",
    "horrible",
    "dislike",
    "sucks",
    "unhappy",
    "lousy",
    "dreadful",
    "painful",
    "ugly",
    "annoyed",
    "disgusting",
    "regret",
    "bored",
    "annoying",
    "depressing",
    "miserable",
    "unpleasant",
    "cry",
    "upset",
    "horrid",
    "gross",
    "filthy",
    "nasty",
    "despair",
    "distressed",
    "ugh",
    "no",
    "😞",
    "😡",
    "😭",
    "💔",
}

preceding_modifiers = {
    "not",
    "no",
    "never",
    "n't",
    "very",
    "so",
    "really",
    "quite",
    "much",
    "still",
    "hardly",
    "barely",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "mustn't",
    "needn't",
    "mightn't",
    "mayn't",
}

punctuation_chars = set(string.punctuation) | {"—"}  # include em‑dash


def find_boundary(text, pos, direction):
    """
    Return the index of the nearest punctuation (or string end) in the given
    direction from position `pos`. `direction` should be -1 (search backwards)
    or +1 (search forwards). The returned index is the first character after
    the punctuation when searching backwards, and the position of the punctuation
    when searching forwards (or the end of the string).
    """
    if direction == -1:
        for i in range(pos - 1, -1, -1):
            if text[i] in punctuation_chars:
                return i + 1
        return 0
    else:
        for i in range(pos, len(text)):
            if text[i] in punctuation_chars:
                return i
        return len(text)


def select_text(row):
    """
    Return the phrase that contains a sentiment word (plus an optional preceding
    modifier) and is bounded by punctuation. Among all candidate phrases we
    choose the one whose length is closest to the median selected_text length
    observed in the training data (MEDIAN_SEL_LEN). This keeps the rule‑based
    extractor intact while calibrating phrase size to typical gold answers.
    """
    text = row["text"]
    sentiment = row["sentiment"]
    target_set = (
        positive_words
        if sentiment == "positive"
        else negative_words if sentiment == "negative" else None
    )
    if target_set is None:
        return text

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, target_set)) + r")\b", re.IGNORECASE
    )

    candidates = []  # list of (phrase, length)

    for match in pattern.finditer(text):
        start_idx = find_boundary(text, match.start(), -1)

        prefix = text[start_idx : match.start()].rstrip()
        if prefix:
            last_token = prefix.split()[-1]
            if last_token.lower() in preceding_modifiers:
                modifier_start = text.rfind(last_token, start_idx, match.start())
                if modifier_start != -1:
                    start_idx = modifier_start

        end_idx = find_boundary(text, match.end(), 1)

        phrase = text[start_idx:end_idx].strip()
        if phrase:
            candidates.append((phrase, len(phrase)))

    if not candidates:
        return text

    best_phrase = min(
        candidates,
        key=lambda x: (abs(x[1] - MEDIAN_SEL_LEN), -x[1]),  # tie‑breaker: longer
    )[0]

    return best_phrase


sub_df["selected_text"] = df_test.apply(select_text, axis=1)



## === cell 3
sub_df.to_csv("submission.csv", index=False)
print("submission.csv written successfully, shape:", sub_df.shape)
