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
sklearn-pandas==2.2.0

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

0.0024781166575849

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'Your notebook fails because it tries to read a non-existent input submission file (`/kaggle/input/subbbbbbbbbbbbb/...`). I remove that dependency and instead generate a valid baseline submission directly from the provided test data, keeping the overall “produce submission.csv” flow intact. To ensure the output is always valid and quoted correctly, the code fill `selected_text` with the full tweet text (a common safe baseline for this competition) and handle missing texts. This run end-to-end in the Kaggle environment and write `submission.csv` with the required columns.'
- What this solution (achieved 0.01264) has done: 'Your current score (0.59324) is far above the target (0.002478), so we should deliberately reduce performance toward the target rather than improve it. The smallest, safest way to do that while keeping the same “baseline submission from test.csv” core logic is to output an almost-always-wrong constant string for `selected_text` (still valid/quoted CSV), which should drive the Jaccard score down sharply. I keep the exact same I/O paths and submission-writing flow, but replace the prediction rule with a constant token and ensure there are no missing values. This preserves evaluation semantics (still predicting a string span) while moving the score closer to the low target.'
- What this solution (achieved 0.0) has done: 'Your current score (0.01264) is still above the very low target (0.002478), so we should further decrease performance toward the target while keeping the same “constant-string submission” core logic. The smallest reliable way to make predictions even less likely to overlap with the true selected spans is to output a long constant string made of rare tokens, which usually share zero words with the ground truth and drive Jaccard closer to 0. I also align the submission rows to `test.csv` by merging on `textID` (a correctness/stability fix that shouldn’t improve score, just prevents accidental misalignment). The output remains a valid quoted CSV with the required columns and row count.'
- What this solution (achieved 0.47378) has done: 'Your current score (0.0) is below the target (0.002478), so we need to *slightly* increase performance while keeping the same “simple rule-based submission from test.csv” core logic. The minimal change is to stop forcing an always-zero-overlap rare-token string and instead use a very small sentiment-aware heuristic that typically overlaps a little: output the full tweet for neutral, and just the first token for positive/negative. This should lift the Jaccard above 0 while staying far from strong models, likely moving closer to the low target band. I also simplify the row alignment to strictly follow `test.csv` order (no merge needed) to avoid any accidental mismatches.'
- What this solution (achieved 0.01303) has done: 'Your current score (0.47378) is far above the very low target (0.002478), so we should deliberately reduce performance toward the target, not improve it. The smallest safe change that keeps the same “simple rule-based submission from test.csv” core logic is to predict an almost-always-wrong constant token for all rows, which should push Jaccard close to 0 but not necessarily exactly 0. To avoid accidentally getting exact-zero overlap too often, we use a simple common token (`"the"`) rather than rare gibberish. We keep the same I/O paths and still write a valid `submission.csv` with the required columns and correct row order.'
- What this solution (achieved 0.0) has done: 'Your current score (0.01303) is above the target (0.002478), so we should deliberately reduce performance toward the target rather than improve it. Keeping the same core “constant prediction” logic, the smallest reliable way to drop Jaccard is to predict a token that is less likely to appear in true selected spans than `"the"`. I switch the constant output to a rare-looking alphabetic token (still a valid single-word phrase) and add a tiny safety cast to avoid NaNs, while keeping the same I/O paths and producing a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'Your current score is below the target, so we need a very small, controlled increase without changing the “constant prediction” core logic. The most reliable minimal lift is to return the constant token only when it actually appears as a whole word in the tweet; otherwise return the full tweet text, which guarantees some overlap and nudges the Jaccard upward slightly. This keeps the same rule-based, no-training approach and preserves submission semantics, while avoiding jumping back to a high-performing heuristic. The rest of the pipeline (paths, columns, row order, CSV writing) stays the same.'
- What this solution (achieved 0.0) has done: 'Your current score (0.59324) is far above the very low target (0.002478), so we should intentionally reduce performance toward the target rather than improve it. The smallest, most reliable way to do that while preserving your existing “rule-based submission from test.csv” core logic is to stop outputting the full tweet (which creates lots of overlap) and instead output a constant token for every row. To avoid accidentally matching real tweet words (which would increase Jaccard), we keep a rare alphabetic token and remove the conditional that sometimes returns the tweet text. The pipeline, paths, ordering, and submission format remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.01138) has done: 'Your current score (0.0) is below the target (0.002478), so we need a very small, controlled increase while keeping the same constant-prediction core logic. The minimal, stable way is to sometimes output a very common token that is likely to overlap with true selected spans (raising Jaccard slightly), while still usually being wrong so the score doesn’t jump too high. Concretely, we use a sentiment-gated constant: for a small fraction of rows (neutral only), output `"I"` (common in tweets); otherwise keep the rare token `"xqzjv"`. This preserves the no-training, rule-based flow and still writes a valid `submission.csv` with correct columns and row order.'
- What this solution (achieved 0.00828) has done: 'Your current score (0.01138) is above the target (0.002478), so we should slightly reduce performance while keeping the same constant-token, no-training submission logic. The smallest controllable knob here is to reduce how often we output the common token `"I"` (which likely overlaps with true spans) by only using it for a narrower subset of neutrals. Concretely, we output `"I"` only when the tweet text actually contains a standalone “I” (word-boundary match); otherwise we fall back to the rare token `"xqzjv"`, which should lower average Jaccard while remaining stable and valid. All paths, row order, output columns, and CSV writing remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current score (0.00828) is above the target (0.002478), so we should reduce performance slightly while keeping the same constant-token, no-training submission logic. The smallest controllable knob is to make the “more-overlapping” `COMMON_TOKEN` appear even less often by tightening the neutral-condition from “contains standalone `I`” to “contains standalone `I` followed by an apostrophe (as in `I'm`/`I'd`/`I'll`)”, which should reduce accidental overlaps and move the score downward toward the target. All file paths, row order, output columns, and CSV writing remain unchanged, and it still produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is below the target (0.002478), so we need a small, controlled increase while keeping the same constant-token, no-training submission approach. The least risky way is to make the “overlap-more” token trigger slightly more often by broadening the neutral condition from matching only `I'` to matching common `I` forms like `I'm/I've/I'll/I'd` (and still only for neutral tweets). This should lift the Jaccard above 0 but remain very low because the fallback stays the rare token for most rows. Paths, row order, column names, and CSV writing remain unchanged to ensure a valid submission.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import re

df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

CONST_TOKEN = "xqzjv"  # rare token (keeps score low)
COMMON_TOKEN = "I"  # common token (slightly increases overlap when used)

_I_CONTRACTION_RE = re.compile(r"(?<!\w)I'(?:m|ve|ll|d)(?!\w)", re.IGNORECASE)


def predict_selected_text(text: str, sentiment: str) -> str:
    s = (sentiment or "").strip().lower()
    txt = "" if text is None else str(text)
    if s == "neutral" and _I_CONTRACTION_RE.search(txt) is not None:
        return COMMON_TOKEN
    return CONST_TOKEN


out = df_test[["textID", "text", "sentiment"]].copy()

out["text"] = out["text"].fillna("").astype(str)
out["sentiment"] = out["sentiment"].fillna("").astype(str)

out["selected_text"] = [
    predict_selected_text(t, s) for t, s in zip(out["text"], out["sentiment"])
]

out["selected_text"] = out["selected_text"].fillna("").astype(str)
out.loc[out["selected_text"].str.len().eq(0), "selected_text"] = CONST_TOKEN

out = out[["textID", "selected_text"]]

out.to_csv("submission.csv", index=False)

df_sub = out



## === cell 2
df_sub.head()
