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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

fuzzywuzzy==0.18.0
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
ydata-profiling==4.17.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.4504

# 6. Current score

0.39153

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.39153) has done: 'I fix the runtime errors by removing the unavailable `en_core_web_lg` dependency and replacing the embedding-based similarity with a lightweight, installed-package-only approach that still follows your existing “compute a similarity score per row, then submit” core logic. Specifically, I keep your fuzzy matching feature and add a minimal, deterministic calibration step learned from the training set to map fuzzy similarity into the 0–1 score range, which should improve correlation toward your target. I also ensure `test_df['score']` is always created and that the final `submission.csv` has exactly the required `id,score` columns and correct row alignment. Finally, I skip the expensive `pandas_profiling` report (it isn’t needed for training/prediction and often breaks/runs too long in Kaggle).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
train_df["score"] = train_df["score"].astype(float)

test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
test_id = test_df["id"]



## === cell 3
def baseline_result(df):
    df = df.copy()

    def even_odd(row):
        if row.number % 2 == 0:
            return 0.50
        else:
            return 0.25

    df["number"] = pd.Series(range(0, df.shape[0]), index=df.index)
    df["pred"] = df.apply(lambda x: even_odd(x), axis=1)
    correlation = df["pred"].corr(df["score"], method="pearson")
    return correlation


train_corr = baseline_result(train_df)
print(f"Train pearson correlation {train_corr:.4f}")



## === cell 4
from fuzzywuzzy import fuzz


def round_values(x):
    if x >= 0 and x < 0.125:
        return 0.0
    if x >= 0.125 and x < 0.375:
        return 0.25
    if x >= 0.375 and x < 0.625:
        return 0.5
    if x >= 0.625 and x < 0.875:
        return 0.75
    if x >= 0.875 and x <= 1:
        return 1.0
    return float(np.clip(x, 0.0, 1.0))


train_df["fuzzy"] = train_df.apply(
    lambda x: fuzz.partial_ratio(x["anchor"], x["target"]) / 100.0, axis=1
)
train_df["fuzzy_rounded"] = train_df["fuzzy"].apply(round_values)

train_corr = train_df["fuzzy"].corr(train_df["score"], method="pearson")
print(f"Train pearson correlation {train_corr:.4f}")

train_corr = train_df["fuzzy_rounded"].corr(train_df["score"], method="pearson")
print(f"Train pearson correlation for rounded similarity {train_corr:.4f}")



## === cell 5
x = train_df["fuzzy"].to_numpy(dtype=float)
y = train_df["score"].to_numpy(dtype=float)

a, b = np.polyfit(x, y, deg=1)

train_pred = np.clip(a * x + b, 0.0, 1.0)
train_cal_corr = pd.Series(train_pred).corr(train_df["score"], method="pearson")
print(f"Train pearson correlation for calibrated fuzzy {train_cal_corr:.4f}")

test_df["fuzzy"] = test_df.apply(
    lambda x: fuzz.partial_ratio(x["anchor"], x["target"]) / 100.0, axis=1
)
test_df["score"] = np.clip(a * test_df["fuzzy"].to_numpy(dtype=float) + b, 0.0, 1.0)



## === cell 6
submission_df = pd.DataFrame({"id": test_id, "score": test_df["score"].astype(float)})

assert submission_df.shape[0] == test_df.shape[0]
assert list(submission_df.columns) == ["id", "score"]



## === cell 7
submission_df.to_csv("submission.csv", index=False, header=True)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
