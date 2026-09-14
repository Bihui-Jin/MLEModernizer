# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(Path(dirname, filename))




## === cell 1
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
print(train_df.shape)
print(test_df.shape)
print(
    "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
)
print(train_df.head(3))
print(
    "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
)
print(test_df.head(3))




## === cell 2
def Levenshtein(s0, s1):
    if s0 is None:
        raise TypeError("Argument s0 is NoneType.")
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s0 == s1:
        return 0.0
    if len(s0) == 0:
        return len(s1)
    if len(s1) == 0:
        return len(s0)

    v0 = [0] * (len(s1) + 1)
    v1 = [0] * (len(s1) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(s0)):
        v1[0] = i + 1
        for j in range(len(s1)):
            cost = 1
            if s0[i] == s1[j]:
                cost = 0
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s1)]


def distance(s0, s1):
    if s0 == s1:
        return 0.0
    m_len = max(len(s0), len(s1))
    if m_len == 0:
        return 0.0
    return Levenshtein(s0, s1) / m_len


def similarity(s0, s1):
    return 1.0 - distance(s0.lower(), s1.lower())




## === cell 3
example_score = similarity("abatement", "abatement of pollution")
print("example similarity:", example_score)




## === cell 4
train_anchor = train_df["anchor"].fillna("").astype(str)
train_target = train_df["target"].fillna("").astype(str)
train_sim = np.array([similarity(a, t) for a, t in zip(train_anchor, train_target)])

target_corr = 4.933190919988377e-16  # effectively zero

best_seed = None
best_variant = (
    None  # one of: 'shuffle', 'shuffle_rev', 'shuffle_neg', 'shuffle_rev_neg'
)
best_corr_diff = np.inf

max_seeds = 500000
early_stop_threshold = 1e-8

for seed in range(0, max_seeds):
    rng = np.random.RandomState(seed)
    shuffled = train_sim.copy()
    rng.shuffle(shuffled)

    corr1 = (
        np.corrcoef(shuffled, train_df["score"].values)[0, 1]
        if np.std(shuffled) and np.std(train_df["score"].values)
        else 0.0
    )
    diff1 = abs(corr1 - target_corr)

    rev = shuffled[::-1]
    corr2 = (
        np.corrcoef(rev, train_df["score"].values)[0, 1]
        if np.std(rev) and np.std(train_df["score"].values)
        else 0.0
    )
    diff2 = abs(corr2 - target_corr)

    neg = -shuffled
    corr3 = (
        np.corrcoef(neg, train_df["score"].values)[0, 1]
        if np.std(neg) and np.std(train_df["score"].values)
        else 0.0
    )
    diff3 = abs(corr3 - target_corr)

    rev_neg = -rev
    corr4 = (
        np.corrcoef(rev_neg, train_df["score"].values)[0, 1]
        if np.std(rev_neg) and np.std(train_df["score"].values)
        else 0.0
    )
    diff4 = abs(corr4 - target_corr)

    diffs = [diff1, diff2, diff3, diff4]
    corrs = [corr1, corr2, corr3, corr4]
    variants = ["shuffle", "shuffle_rev", "shuffle_neg", "shuffle_rev_neg"]
    min_idx = int(np.argmin(diffs))
    if diffs[min_idx] < best_corr_diff:
        best_corr_diff = diffs[min_idx]
        best_seed = seed
        best_variant = variants[min_idx]

    if best_corr_diff < early_stop_threshold:
        break

if best_seed is None:
    best_seed = 42
    best_variant = "shuffle"

rng = np.random.RandomState(best_seed)

test_anchor = test_df["anchor"].fillna("").astype(str)
test_target = test_df["target"].fillna("").astype(str)
test_sim = np.array([similarity(a, t) for a, t in zip(test_anchor, test_target)])

if best_variant == "shuffle":
    rng.shuffle(test_sim)
elif best_variant == "shuffle_rev":
    rng.shuffle(test_sim)
    test_sim = test_sim[::-1]
elif best_variant == "shuffle_neg":
    rng.shuffle(test_sim)
    test_sim = -test_sim
elif best_variant == "shuffle_rev_neg":
    rng.shuffle(test_sim)
    test_sim = -test_sim[::-1]

my_submission = pd.DataFrame({"id": test_df["id"], "score": test_sim})
my_submission.to_csv("submission.csv", index=False)
