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

0.50704

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43997) has done: 'I extend the list of sentence‑transformer models and broaden the range of ridge regularisation values so the validation Pearson can improve and move closer to the target score, while keeping the overall pipeline unchanged. The rest of the notebook stays the same, and a valid `submission.csv` is still written to `/kaggle/working`.'
- What this solution (achieved 0.50721) has done: 'The fix adds robust handling for models that fail to load (skipping them instead of crashing) and builds an ensemble by averaging ridge‑regressed predictions from all successfully loaded sentence‑transformer embeddings. This keeps the original pipeline while improving the validation Pearson correlation, moving the score closer to the target.'
- What this solution (achieved 0.50704) has done: 'I set the protobuf implementation environment variable before importing `sentence_transformers` to avoid the AttributeError, move the import inside the model‑loading loop, and combine model predictions using validation‑Pearson weights rather than a simple mean. These changes fix the crash and should lift the validation correlation toward the target while preserving the original pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

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
train_path = next(Path("/kaggle/input").rglob("train.csv"))
train_df = pd.read_csv(train_path)
train_df = prepare_df(train_df)

embedder_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # may fail, handled gracefully
]

batch_size = 256
candidate_alphas = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

X_train_idx, X_val_idx, y_train, y_val = train_test_split(
    train_df.index, train_df["score"].values, test_size=0.2, random_state=42
)

model_infos = []

for name in embedder_names:
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(name)
    except Exception as e:
        print(f"Skipping embedder {name} due to load error: {e}")
        continue

    all_embeddings = []
    for i in range(0, len(train_df), batch_size):
        batch_texts = train_df["inputs"].iloc[i : i + batch_size].tolist()
        batch_emb = embedder.encode(
            batch_texts, batch_size=len(batch_texts), show_progress_bar=False
        )
        all_embeddings.append(batch_emb)
    all_embeddings = np.vstack(all_embeddings)

    X_train = all_embeddings[X_train_idx]
    X_val = all_embeddings[X_val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    best_alpha = candidate_alphas[0]
    best_val_corr = -np.inf
    for a in candidate_alphas:
        ridge_tmp = Ridge(alpha=a, random_state=42)
        ridge_tmp.fit(X_train_scaled, y_train)
        X_val_scaled = scaler.transform(X_val)
        val_pred = ridge_tmp.predict(X_val_scaled)
        val_corr, _ = pearsonr(y_val, val_pred)
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_alpha = a

    model_infos.append(
        {
            "name": name,
            "embedder": embedder,
            "embeddings": all_embeddings,
            "scaler": scaler,
            "alpha": best_alpha,
            "val_corr": best_val_corr,
        }
    )
    print(
        f"Model {name}: best alpha {best_alpha}, validation Pearson {best_val_corr:.6f}"
    )

if not model_infos:
    raise RuntimeError("No sentence‑transformer models could be loaded.")

model_weights = np.array([max(info["val_corr"], 0.0) for info in model_infos])
if model_weights.sum() == 0:
    model_weights = np.ones_like(model_weights)  # fallback to equal weighting
model_weights = model_weights / model_weights.sum()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
test_path = next(Path("/kaggle/input").rglob("test.csv"))
test_df = pd.read_csv(test_path)
test_df = prepare_df(test_df)

all_test_preds = []

for info in model_infos:
    X_full = info["embeddings"]
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)

    ridge = Ridge(alpha=info["alpha"], random_state=42)
    ridge.fit(X_full_scaled, train_df["score"])

    test_embeds = []
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df["inputs"].iloc[i : i + batch_size].tolist()
        batch_emb = info["embedder"].encode(
            batch_texts, batch_size=len(batch_texts), show_progress_bar=False
        )
        test_embeds.append(batch_emb)
    test_embeds = np.vstack(test_embeds)

    test_scaled = scaler_full.transform(test_embeds)
    preds = ridge.predict(test_scaled)
    preds = np.clip(preds, 0.0, 1.0)
    all_test_preds.append(preds)

pred_matrix = np.column_stack(all_test_preds)
final_preds = np.average(pred_matrix, axis=1, weights=model_weights)

sub_df = pd.DataFrame({"id": test_df["id"], "score": final_preds})
submission_path = "/kaggle/working/submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
