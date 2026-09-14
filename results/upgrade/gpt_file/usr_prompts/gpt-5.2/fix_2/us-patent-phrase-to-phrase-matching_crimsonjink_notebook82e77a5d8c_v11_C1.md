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

catboost==1.2.8
cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.2163

# 6. Current score

0.35787

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.35787) has done: 'Your current pipeline is scoring far above the target (0.39365 vs 0.2163), so the goal is to *reduce* performance slightly toward the target band with the smallest safe change. The least invasive way is to keep the same TF‑IDF + LogisticRegression core, but prevent leakage by adding a proper train/validation split and selecting a more conservative model snapshot (stronger regularization) based on validation Pearson rather than fitting on all data at once. This typically lowers generalization (and thus public LB) while still producing a valid submission and keeping the same modeling approach. I also switch from hard class predictions to expected-value regression using `predict_proba` mapped onto the original score levels, which is closer to the Pearson objective and stabilizes outputs while still being minimal.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

from cuml.linear_model import LogisticRegression



## === cell 1
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
sample = pd.read_csv(
    "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

le = LabelEncoder()



## === cell 2
y = train.score
X = train.drop(["id", "context", "score"], axis=1)

y_enc = le.fit_transform(y).astype("float32")

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)



## === cell 3
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
transformer = make_column_transformer(
    (vectorizer, "anchor"),
    (vectorizer, "target"),
)

X_tr_vec = transformer.fit_transform(X_tr)
X_va_vec = transformer.transform(X_va)




## === cell 4
def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


score_levels = le.inverse_transform(np.arange(len(le.classes_))).astype(np.float32)

best_corr = -1.0
best_C = None
best_model = None

for C in [0.05, 0.1, 0.2]:
    model = LogisticRegression(C=C)
    model.fit(X_tr_vec, y_tr)

    va_proba = model.predict_proba(X_va_vec)
    va_pred_score = (va_proba * score_levels[None, :]).sum(axis=1)

    va_true_score = le.inverse_transform(y_va.astype(np.int32)).astype(np.float32)
    corr = pearson_corr(va_true_score, va_pred_score)

    if corr > best_corr:
        best_corr = corr
        best_C = C
        best_model = model



## === cell 5
X_all_vec = transformer.fit_transform(train[["anchor", "target"]])
y_all_enc = y_enc

final_model = LogisticRegression(C=best_C)
final_model.fit(X_all_vec, y_all_enc)



## === cell 6
t = test[["anchor", "target"]]
t_vec = transformer.transform(t)

test_proba = final_model.predict_proba(t_vec)
pred_score = (test_proba * score_levels[None, :]).sum(axis=1)



## === cell 7
sample["score"] = pred_score.astype(np.float32)
sample.to_csv("submission.csv", index=False)
