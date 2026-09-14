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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



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
    ngram_range=(1, 2),
    max_features=20000,
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
y_train_full = train_df["score"].values



## === cell 5
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

reg = LinearRegression()
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
test_pred = reg.predict(X_test)

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 7
submission_df = pd.DataFrame({"id": test_df["id"], "score": test_pred})
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} – shape: {submission_df.shape}")
