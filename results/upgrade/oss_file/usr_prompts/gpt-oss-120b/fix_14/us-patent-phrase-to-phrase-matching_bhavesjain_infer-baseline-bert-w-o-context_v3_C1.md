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
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
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

0.4821241527019558

# 6. Current score

0.34954

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35175) has done: 'I keep the overall pipeline unchanged and only adjust the regularization strength of the Ridge regressor (lowering α from 0.5 to 0.1). This minor tweak often yields a modest boost in Pearson correlation without altering the core model or feature engineering, moving the validation score closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.35175) has done: 'I lower the Ridge regularization strength to α=0.01 (less shrinkage) and add three absolute‑difference features between the cosine similarities (|anchor‑target|, |anchor‑context|, |target‑context|). These tiny extensions keep the original linear‑model pipeline intact while giving the model a bit more signal, which should raise the Pearson correlation toward the target value.'
- What this solution (achieved 0.35175) has done: 'I slightly reduce the Ridge regularization (α = 0.001) and add three complementary features — `1‑cosine` for each of the anchor‑target, anchor‑context and target‑context similarities — which give the model a bit more signal while keeping the linear‑ridge pipeline unchanged. These minimal changes should raise the validation Pearson correlation toward the target score.'
- What this solution (achieved 0.35173) has done: 'I reduce the Ridge regularization to zero (α = 0.0), effectively using an unregularized linear model while keeping all other preprocessing, feature engineering, and scaling unchanged. This minimal tweak often improves Pearson correlation when the original regularization is already very low, moving the validation score closer to the target without altering the core pipeline.'
- What this solution (achieved 0.35365) has done: 'I add two lightweight features – the product of all three cosine similarities and a log‑transformed version of each cosine – to give the linear model a bit more signal, and I switch the Ridge regularisation from 0.0 to a very small α=0.001 to improve generalisation. These changes keep the original pipeline and model type intact while aiming to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.35365) has done: 'I slightly increase the Ridge regularisation to α=0.01 (a modest amount of shrinkage that often improves generalisation) and add three lightweight ratio features – each cosine similarity divided by the sum of the other two – to give the linear model a bit more discriminative signal while keeping the core pipeline unchanged. These minimal adjustments are expected to raise the validation Pearson correlation toward the target score.'
- What this solution (achieved 0.34954) has done: 'I add a few lightweight aggregate features (average, minimum, and maximum of the three cosine similarities) which give the linear model a bit more signal without changing its core architecture. I also remove the regularisation by setting the Ridge α to 0.0, as an un‑regularised linear model often captures more variance in this type of data. These minimal adjustments are expected to raise the validation Pearson correlation toward the target while keeping the original pipeline intact.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




## === cell 1
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
def clean_text(text):
    """Lower‑case, keep alphanumerics/spaces, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


for col in ["anchor", "target", "context"]:
    train_df[f"{col}_clean"] = train_df[col].apply(clean_text)
    test_df[f"{col}_clean"] = test_df[col].apply(clean_text)




## === cell 3
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    norm="l2",
    sublinear_tf=True,
)

all_texts = pd.concat(
    [train_df["anchor_clean"], train_df["target_clean"], train_df["context_clean"]]
)
vectorizer.fit(all_texts)


def cosine_sim(vec_a, vec_b):
    """Return cosine similarity array for two sparse CSR matrices."""
    dot = vec_a.multiply(vec_b).sum(axis=1).A1
    norm_a = np.sqrt(vec_a.multiply(vec_a).sum(axis=1)).A1
    norm_b = np.sqrt(vec_b.multiply(vec_b).sum(axis=1)).A1
    return dot / (norm_a * norm_b + 1e-10)




## === cell 4
anchor_train = vectorizer.transform(train_df["anchor_clean"])
target_train = vectorizer.transform(train_df["target_clean"])
context_train = vectorizer.transform(train_df["context_clean"])

cos_at_train = cosine_sim(anchor_train, target_train)
cos_ac_train = cosine_sim(anchor_train, context_train)
cos_tc_train = cosine_sim(target_train, context_train)

X_train_full = np.vstack([cos_at_train, cos_ac_train, cos_tc_train]).T

avg_cos_train = np.mean(X_train_full, axis=1)[:, None]
min_cos_train = np.min(X_train_full, axis=1)[:, None]
max_cos_train = np.max(X_train_full, axis=1)[:, None]

inter_at_ac = (X_train_full[:, 0] * X_train_full[:, 1])[:, None]
inter_at_tc = (X_train_full[:, 0] * X_train_full[:, 2])[:, None]
inter_ac_tc = (X_train_full[:, 1] * X_train_full[:, 2])[:, None]

cos_sq = X_train_full**2
sum_sim = (X_train_full[:, 0] + X_train_full[:, 1] + X_train_full[:, 2])[:, None]

diff_at_ac = np.abs(X_train_full[:, 0] - X_train_full[:, 1])[:, None]
diff_at_tc = np.abs(X_train_full[:, 0] - X_train_full[:, 2])[:, None]
diff_ac_tc = np.abs(X_train_full[:, 1] - X_train_full[:, 2])[:, None]

inv_cos = 1.0 - X_train_full  # shape (n_samples, 3)

triple_product = (X_train_full[:, 0] * X_train_full[:, 1] * X_train_full[:, 2])[:, None]
log_cos = np.log1p(X_train_full)  # shape (n_samples, 3)

ratio_at = (X_train_full[:, 0] / (X_train_full[:, 1] + X_train_full[:, 2] + 1e-6))[
    :, None
]
ratio_ac = (X_train_full[:, 1] / (X_train_full[:, 0] + X_train_full[:, 2] + 1e-6))[
    :, None
]
ratio_tc = (X_train_full[:, 2] / (X_train_full[:, 0] + X_train_full[:, 1] + 1e-6))[
    :, None
]

X_train_full = np.hstack(
    [
        X_train_full,  # original cosine similarities
        avg_cos_train,
        min_cos_train,
        max_cos_train,
        inter_at_ac,
        inter_at_tc,
        inter_ac_tc,
        cos_sq,
        sum_sim,
        diff_at_ac,
        diff_at_tc,
        diff_ac_tc,
        inv_cos,  # three (1‑cos) features
        triple_product,  # triple‑interaction feature
        log_cos,  # log‑transformed cosine features
        ratio_at,
        ratio_ac,
        ratio_tc,  # lightweight ratio features
    ]
)

y_train_full = train_df["score"].values




## === cell 5
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

reg = Ridge(alpha=0.0, random_state=42)
reg.fit(X_tr, y_tr)

val_pred = reg.predict(X_val)
val_corr = np.corrcoef(y_val, val_pred)[0, 1]
print(f"Validation Pearson correlation: {val_corr:.5f}")




## === cell 6
anchor_test = vectorizer.transform(test_df["anchor_clean"])
target_test = vectorizer.transform(test_df["target_clean"])
context_test = vectorizer.transform(test_df["context_clean"])

cos_at_test = cosine_sim(anchor_test, target_test)
cos_ac_test = cosine_sim(anchor_test, context_test)
cos_tc_test = cosine_sim(target_test, context_test)

X_test = np.vstack([cos_at_test, cos_ac_test, cos_tc_test]).T

avg_cos_test = np.mean(X_test, axis=1)[:, None]
min_cos_test = np.min(X_test, axis=1)[:, None]
max_cos_test = np.max(X_test, axis=1)[:, None]

inter_at_ac_test = (X_test[:, 0] * X_test[:, 1])[:, None]
inter_at_tc_test = (X_test[:, 0] * X_test[:, 2])[:, None]
inter_ac_tc_test = (X_test[:, 1] * X_test[:, 2])[:, None]

cos_sq_test = X_test**2
sum_sim_test = (X_test[:, 0] + X_test[:, 1] + X_test[:, 2])[:, None]

diff_at_ac_test = np.abs(X_test[:, 0] - X_test[:, 1])[:, None]
diff_at_tc_test = np.abs(X_test[:, 0] - X_test[:, 2])[:, None]
diff_ac_tc_test = np.abs(X_test[:, 1] - X_test[:, 2])[:, None]

inv_cos_test = 1.0 - X_test

triple_product_test = (X_test[:, 0] * X_test[:, 1] * X_test[:, 2])[:, None]
log_cos_test = np.log1p(X_test)

ratio_at_test = (X_test[:, 0] / (X_test[:, 1] + X_test[:, 2] + 1e-6))[:, None]
ratio_ac_test = (X_test[:, 1] / (X_test[:, 0] + X_test[:, 2] + 1e-6))[:, None]
ratio_tc_test = (X_test[:, 2] / (X_test[:, 0] + X_test[:, 1] + 1e-6))[:, None]

X_test = np.hstack(
    [
        X_test,
        avg_cos_test,
        min_cos_test,
        max_cos_test,
        inter_at_ac_test,
        inter_at_tc_test,
        inter_ac_tc_test,
        cos_sq_test,
        sum_sim_test,
        diff_at_ac_test,
        diff_at_tc_test,
        diff_ac_tc_test,
        inv_cos_test,
        triple_product_test,
        log_cos_test,
        ratio_at_test,
        ratio_ac_test,
        ratio_tc_test,
    ]
)

X_test = scaler.transform(X_test)

test_pred = reg.predict(X_test)
test_pred = np.clip(test_pred, 0.0, 1.0)




## === cell 7
submission_df = pd.DataFrame({"id": test_df["id"], "score": test_pred})
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} – shape: {submission_df.shape}")
