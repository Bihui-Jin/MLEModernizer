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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.1947

# 6. Current score

0.22676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42813) has done: 'Implemented a fix for the `Ridge` model by switching to the `"lsqr"` solver, which is compatible with the current SciPy version and prevents the `cg()` argument error. Adjusted the model initialization accordingly and ensured the training, validation, full‑data fitting, and prediction steps use this corrected model. No other logic was altered, preserving the original TF‑IDF feature pipeline and submission creation.'
- What this solution (achieved 0.22676) has done: 'I lower the model’s capacity and increase regularisation so the predictions become less correlated with the true scores, moving the Pearson‑r from the current 0.428 toward the target 0.1947. Specifically, I reduce TF‑IDF features from 50 k to 10 k and raise the Ridge α from 1.0 to 20.0; these are the smallest changes that reliably weaken the fit without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_sub = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)
df_train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
df_test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")



## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr



## === cell 3
train_text = (
    df_train["anchor"] + " " + df_train["context"] + " " + df_train["target"]
).str.lower()
test_text = (
    df_test["anchor"] + " " + df_test["context"] + " " + df_test["target"]
).str.lower()
y = df_train["score"].astype(float).values



## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    train_text, y, test_size=0.3, random_state=42
)



## === cell 5
vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), stop_words="english"
)
X_tr_vec = vectorizer.fit_transform(X_tr)
X_val_vec = vectorizer.transform(X_val)



## === cell 6
model = Ridge(alpha=20.0, random_state=42, solver="lsqr")
model.fit(X_tr_vec, y_tr)



## === cell 7
val_pred = model.predict(X_val_vec)
pearson_corr, _ = pearsonr(y_val, val_pred)
print(f"Validation Pearson correlation: {pearson_corr:.5f}")



## === cell 8
X_full_vec = vectorizer.fit_transform(train_text)
model.fit(X_full_vec, y)



## === cell 9
X_test_vec = vectorizer.transform(test_text)
test_pred = model.predict(X_test_vec)



## === cell 10
test_pred_clipped = np.clip(test_pred, 0.0, 1.0)
test_pred_rounded = np.round(test_pred_clipped / 0.25) * 0.25



## === cell 11
df_test["score"] = test_pred_rounded
submission = df_test[["id", "score"]]
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
