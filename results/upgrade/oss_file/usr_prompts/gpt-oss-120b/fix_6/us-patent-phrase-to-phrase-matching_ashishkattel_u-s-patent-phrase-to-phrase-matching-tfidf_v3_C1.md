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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 5. Target score

0.4719

# 6. Current score

0.52306

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6189) has done: 'I fix the Sparse CG solver issue by switching Ridge to the ‘lsqr’ solver, and correctly build the common‑word feature as a column‑vector sparse matrix for both train and test sets. These changes let the model train and predict without errors, producing a proper `submission.csv` that can be scored (higher Pearson is better).'
- What this solution (achieved 0.58063) has done: 'I increase the Ridge regularisation strength (α) from 1.0 to 10.0. Stronger regularisation shrinks the fitted coefficients, typically lowering predictive power and therefore decreasing the Pearson correlation on the validation/train split. This modest change keeps the original model architecture and feature construction intact while moving the score closer to the target of 0.4719.'
- What this solution (achieved 0.53877) has done: 'I slightly increase the Ridge regularization strength (α) to reduce the model’s predictive power, which lower the Pearson correlation toward the target value of 0.4719 while keeping the overall pipeline unchanged. This is the only modification needed, and it still produce a valid `submission.csv`.'
- What this solution (achieved 0.52306) has done: 'I increase the Ridge regularisation strength from α=50.0 to α=100.0. A stronger α shrinks the fitted coefficients more, modestly lowering the model’s predictive power and therefore decreasing the Pearson correlation on the training data. This moves the score from 0.5388 down into the target tolerance band (~0.51) while keeping the original feature construction, model type, and overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy import sparse
from scipy.stats import pearsonr
from nltk.stem import PorterStemmer




## === cell 1
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
train_data = pd.read_csv(train_path)




## === cell 2
ps = PorterStemmer()


def find_common_word(row):
    word1 = str(row["anchor"]).lower()
    word2 = str(row["target"]).lower()
    w1 = [ps.stem(w) for w in word1.split()]
    w2 = [ps.stem(w) for w in word2.split()]
    set_all = set(w1 + w2)
    if len(set_all) == 0:
        return 0.0
    overlap = len(set(w1).intersection(set(w2)))
    return overlap / len(set_all)


train_data["common"] = train_data.apply(find_common_word, axis=1)




## === cell 3
vectorizer_anchor = TfidfVectorizer()
anchor_tfid = vectorizer_anchor.fit_transform(train_data["anchor"].fillna("").values)

vectorizer_target = TfidfVectorizer()
target_tfid = vectorizer_target.fit_transform(train_data["target"].fillna("").values)

common_sparse = sparse.csr_matrix(train_data["common"].values[:, None])
train_X = sparse.hstack([anchor_tfid, target_tfid, common_sparse])

y = train_data["score"].values




## === cell 4
model = Ridge(alpha=100.0, solver="lsqr", random_state=42)
model.fit(train_X, y)

train_pred = model.predict(train_X)
pearson_train, _ = pearsonr(y, train_pred)
print(f"Train Pearson correlation: {pearson_train:.4f}")




## === cell 5
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
test = pd.read_csv(test_path)

test["common"] = test.apply(find_common_word, axis=1)

test_anchor = vectorizer_anchor.transform(test["anchor"].fillna("").values)
test_target = vectorizer_target.transform(test["target"].fillna("").values)
test_common = sparse.csr_matrix(test["common"].values[:, None])

test_X = sparse.hstack([test_anchor, test_target, test_common])




## === cell 6
test_pred = model.predict(test_X)
test_pred = np.clip(test_pred, 0.0, 1.0)




## === cell 7
submission = pd.DataFrame({"id": test["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
