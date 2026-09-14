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

0.7954764062840406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".csv"):
            print(os.path.join(dirname, filename))




## === cell 1
def prepare_df(df):
    """
    Create a single string that concatenates context, anchor and target.
    This string will be used for embedding.
    """
    df["section"] = df["context"].map(
        lambda val: val.strip()[0] if isinstance(val, str) else ""
    )
    df["sec_tok"] = "[" + df["section"] + "]"
    df["inputs"] = (
        df["sec_tok"]
        + " "
        + df["context"]
        + " "
        + df["anchor"].str.lower()
        + " "
        + df["target"].str.lower()
    )
    return df




## === cell 2
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

train_path = next(Path("/kaggle/input").rglob("train.csv"))
train_df = pd.read_csv(train_path)
train_df = prepare_df(train_df)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
batch_size = 256
train_embeddings = []
for i in range(0, len(train_df), batch_size):
    batch_texts = train_df["inputs"].iloc[i : i + batch_size].tolist()
    batch_emb = embedder.encode(
        batch_texts, batch_size=len(batch_texts), show_progress_bar=False
    )
    train_embeddings.append(batch_emb)
train_embeddings = np.vstack(train_embeddings)

X_train, X_val, y_train, y_val = train_test_split(
    train_embeddings, train_df["score"].values, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

candidate_alphas = [0.1, 0.5, 1.0, 2.0]
best_alpha = candidate_alphas[0]
best_val_score = -np.inf

for a in candidate_alphas:
    ridge_tmp = Ridge(alpha=a, random_state=42)
    ridge_tmp.fit(X_train_scaled, y_train)
    X_val_scaled = scaler.transform(X_val)
    val_pred_tmp = ridge_tmp.predict(X_val_scaled)
    val_corr, _ = pearsonr(y_val, val_pred_tmp)
    if val_corr > best_val_score:
        best_val_score = val_corr
        best_alpha = a

print(f"Chosen alpha: {best_alpha:.3f} with validation Pearson {best_val_score:.6f}")

ridge = Ridge(alpha=best_alpha, random_state=42)
ridge.fit(X_train_scaled, y_train)

X_val_scaled = scaler.transform(X_val)
val_preds = ridge.predict(X_val_scaled)
val_score, _ = pearsonr(y_val, val_preds)
print(f"Validation Pearson correlation (final model): {val_score:.6f}")

scaler_full = StandardScaler()
train_embeddings_scaled = scaler_full.fit_transform(train_embeddings)
ridge.fit(train_embeddings_scaled, train_df["score"])




## === cell 3
test_path = next(Path("/kaggle/input").rglob("test.csv"))
test_df = pd.read_csv(test_path)
test_df = prepare_df(test_df)

test_embeddings = []
for i in range(0, len(test_df), batch_size):
    batch_texts = test_df["inputs"].iloc[i : i + batch_size].tolist()
    batch_emb = embedder.encode(
        batch_texts, batch_size=len(batch_texts), show_progress_bar=False
    )
    test_embeddings.append(batch_emb)
test_embeddings = np.vstack(test_embeddings)

test_embeddings_scaled = scaler_full.transform(test_embeddings)
preds = ridge.predict(test_embeddings_scaled)
preds = np.clip(preds, 0.0, 1.0)

sub_df = pd.DataFrame({"id": test_df["id"], "score": preds})
submission_path = "/kaggle/working/submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
