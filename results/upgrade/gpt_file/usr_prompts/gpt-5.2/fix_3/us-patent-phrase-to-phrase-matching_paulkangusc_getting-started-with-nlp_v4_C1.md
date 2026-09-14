# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
kaggle==1.7.4.5
kaggle-environments==1.18.0
kagglehub==0.3.13
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

0.7564637808094372

# 6. Current score

0.47523

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47523) has done: 'I fix the end-to-end runtime blockers by (1) making the data path resolution work with your provided `/kaggle/input/...` layout, (2) removing the hard dependency on an offline Hugging Face BERT checkpoint (which is not present) and switching to a locally-available sentence-transformers model for embeddings, and (3) replacing the broken Transformers `Trainer` section (also failing due to a protobuf-related issue) with a minimal regression head trained on those embeddings. This preserves the core “text → transformer-based representation → regression → clip to [0,1] → submission.csv” semantics while ensuring the notebook runs fully offline within Kaggle. Finally, it always write a valid `submission.csv` with the required `id,score` columns.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
print("iskaggle:", iskaggle)



## === cell 1
from pathlib import Path

candidate_paths = [
    Path("/kaggle/input/us-patent-phrase-to-phrase-matching"),
    Path("../input/us-patent-phrase-to-phrase-matching"),
    Path("/kaggle/data/us-patent-phrase-to-phrase-matching"),
    Path("us-patent-phrase-to-phrase-matching"),
]

path = next(
    (p for p in candidate_paths if (p.exists() and (p / "train.csv").exists())), None
)
if path is None:
    candidate_paths2 = [
        Path("/kaggle/input"),
        Path("/kaggle/data"),
    ]
    path2 = next(
        (p for p in candidate_paths2 if (p.exists() and (p / "train.csv").exists())),
        None,
    )
    if path2 is None:
        raise FileNotFoundError(
            f"Could not find train.csv under any of: {candidate_paths + candidate_paths2}"
        )
    path = path2

print("Using data path:", path)
print("Has train.csv:", (path / "train.csv").exists())
print("Has test.csv:", (path / "test.csv").exists())
print("Has sample_submission.csv:", (path / "sample_submission.csv").exists())



## === cell 2
import pandas as pd

df = pd.read_csv(path / "train.csv")
df.head()



## === cell 3
df.describe(include="object")



## === cell 4
eval_df = pd.read_csv(path / "test.csv")
eval_df.head()



## === cell 5
eval_df.describe(include="object")



## === cell 6
df["input"] = "TEXT1: " + df.context + "; TEXT2: " + df.target + "; ANC1: " + df.anchor
eval_df["input"] = (
    "TEXT1: "
    + eval_df.context
    + "; TEXT2: "
    + eval_df.target
    + "; ANC1: "
    + eval_df.anchor
)



## === cell 7
import numpy as np
from sentence_transformers import SentenceTransformer

model_name_candidates = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
]

st_model = None
last_err = None
for name in model_name_candidates:
    try:
        st_model = SentenceTransformer(name)
        print("Loaded SentenceTransformer:", name)
        break
    except Exception as e:
        last_err = e
        continue

if st_model is None:
    raise RuntimeError(
        "Could not load any SentenceTransformer model offline. "
        "Tried: " + ", ".join(model_name_candidates) + f". Last error: {last_err}"
    )

train_texts = df["input"].astype(str).tolist()
test_texts = eval_df["input"].astype(str).tolist()

X = st_model.encode(
    train_texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
X_test = st_model.encode(
    test_texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

y = df["score"].astype(float).values

print("X shape:", X.shape, "X_test shape:", X_test.shape, "y shape:", y.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42)

reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_tr, y_tr)

va_pred = reg.predict(X_va)
va_corr = np.corrcoef(va_pred.reshape(-1), y_va.reshape(-1))[0, 1]
va_rmse = mean_squared_error(y_va, va_pred, squared=False)

print("Validation pearson:", float(va_corr))
print("Validation RMSE:", float(va_rmse))



## === cell 9
reg_full = Ridge(alpha=1.0, random_state=42)
reg_full.fit(X, y)



## === cell 10
preds = reg_full.predict(X_test).astype(float).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)
print(preds[:10], len(preds))



## === cell 11
submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
assert submission_df.shape[0] == eval_df.shape[0]
assert submission_df.columns.tolist() == ["id", "score"]
