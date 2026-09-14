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

3.11

# 3. Installed packages

No external packages required in the script and installed.

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

0.8066012823938029

# 6. Current score

0.5091

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57055) has done: 'I fix the Ridge regression fitting error caused by an incompatibility with the SciPy version by switching the solver to `lsqr`, which works with sparse matrices and avoids the problematic `cg` call. I also lower the regularization strength (alpha = 0.1) to improve the Pearson correlation without altering the overall model architecture. The script is renumbered to start from cell 1 and now runs end‑to‑end, producing a valid `submission.csv` file.'
- What this solution (achieved 0.56495) has done: 'I replace the single character‑level TF‑IDF with a combination of character and word TF‑IDF features, stack them together, and lower the Ridge regularisation (alpha = 0.05) to give the linear model more capacity. This keeps the overall pipeline (split, Ridge regression, clipping, CSV output) unchanged while providing richer text representations that should raise the Pearson correlation toward the target score. The script is renumbered to start at cell 1 and now writes a valid `submission.csv`.'
- What this solution (achieved 0.55873) has done: 'I add a few lightweight numeric similarity features (lengths of each text field and token overlap between anchor and target) and concatenate them to the existing TF‑IDF representation. This enriches the feature set without altering the overall model architecture. I also lower the Ridge regularisation slightly (alpha = 0.02) to let the richer features improve the Pearson correlation, aiming to move the validation score closer to the target.'
- What this solution (achieved 0.5744) has done: 'I add a lightweight scaling step for the numeric length/overlap features (using StandardScaler) to bring them onto a comparable magnitude with the sparse TF‑IDF vectors, and I set the Ridge regularisation strength to a modest α=0.1 (slightly stronger than the very low 0.02) which often improves Pearson correlation on validation without altering the overall linear‑model pipeline. These minimal tweaks keep the core architecture unchanged while helping the model fit the data more effectively.'
- What this solution (achieved 0.51453) has done: 'I expand the TF‑IDF vocab (more n‑grams) and add two extra numeric similarity features – token‑overlap ratio and absolute length difference – then scale them together with the existing numeric columns. Finally I lower the Ridge regularisation to α=0.001 so the richer feature set can be used more fully. These tweaks keep the original pipeline (train/val split, Ridge regression, clipping, CSV output) while giving the model more expressive power to raise the Pearson correlation toward the target.'
- What this solution (achieved 0.60669) has done: 'I keep the same Ridge‑linear pipeline but strengthen the text representation (wider character n‑grams and larger vocabularies) and add a few extra token‑overlap numeric features, then set a modest regularisation (α=0.1) to avoid over‑fitting. These small, targeted changes should raise the validation Pearson correlation toward the target while preserving the core model logic and still producing a correct `submission.csv`.'
- What this solution (achieved 0.61989) has done: 'The fix updates the linear model initialization: `RidgeCV` does not accept a `solver` argument, causing the earlier crash. We replace it with `Ridge`, setting a modest regularisation strength (`alpha=0.05`) and the `"lsqr"` solver that works with sparse TF‑IDF features. The import is adjusted accordingly, and the notebook cells are renumbered to start at 1. No other logic changes are made, so the pipeline now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.61966) has done: 'I lower the Ridge regularization to α=0.001 (giving the linear model more capacity) and increase the TF‑IDF vocab size to 600 000 features for both character and word vectorizers. These minimal adjustments keep the original pipeline unchanged while allowing the model to fit the data better, which should raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.61702) has done: 'I raise the main Ridge regularisation slightly (α = 0.01) to improve generalisation and add a lightweight secondary Ridge model that trains only on the numeric similarity features. Their predictions be blended (70 % text‑model, 30 % numeric‑model) before clipping. This keeps the overall linear‑model pipeline intact while giving the numeric cues a modest influence, which should push the validation Pearson correlation closer to the target.'
- What this solution (achieved 0.50987) has done: 'The script had three critical issues preventing it from running and producing a submission: (1) the anchor/target TF‑IDF matrices were built on the full training set, causing mismatched row dimensions when stacking features; (2) `RidgeCV` was incorrectly instantiated with a `solver` argument; (3) the blending weight variable `best_w` was never defined due to the earlier failures. I fixed the feature‑matrix size mismatch by fitting the anchor and target vectorizers only on the training split, replaced `RidgeCV` with a regular `Ridge` model (alpha = 0.05, solver = “lsqr”), and recomputed the optimal blend weight. The pipeline now runs end‑to‑end and writes a correct `submission.csv`.'
- What this solution (achieved 0.50945) has done: 'I keep the same overall pipeline but tighten the regularisation for the text‑only Ridge model (α = 0.01) and add a tiny linear calibration step: after finding the best blend weight on the validation set, I fit a simple y = a·pred + b regression on the validation predictions and apply this (a,b) correction to both validation and test predictions. This post‑processing aligns the raw blended scores more closely with the true scores, which should raise the Pearson correlation toward the target while preserving the original model structure.'
- What this solution (achieved 0.5091) has done: 'I keep the overall linear‑model pipeline but make three small, score‑oriented tweaks: (1) lower the text‑model regularisation (α = 0.005) so the rich TF‑IDF features can be used more fully, (2) enrich the numeric feature set with two additional token‑overlap ratios (anchor–context and target–context) and a length‑ratio, and (3) replace the coarse grid search for the blend weight with a linear‑regression fit on the validation predictions (text + numeric + intercept). These minimal changes stay within the original architecture while giving the model extra expressive power and a more optimal blending, which should push the Pearson correlation toward the target.'

# 9. Code solution

## === cell 0
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.sparse import hstack, csr_matrix

path = Path("../input/us-patent-phrase-to-phrase-matching")
print("Data path:", path)
print("Files:", list(path.iterdir()))




## === cell 1
train_df = pd.read_csv(path / "train.csv")
train_df["input"] = (
    "TEXT1: "
    + train_df["context"]
    + "; TEXT2: "
    + train_df["target"]
    + "; ANC: "
    + train_df["anchor"]
)
print("Train shape:", train_df.shape)
print(train_df.head())




## === cell 2
test_df = pd.read_csv(path / "test.csv")
test_df["input"] = (
    "TEXT1: "
    + test_df["context"]
    + "; TEXT2: "
    + test_df["target"]
    + "; ANC: "
    + test_df["anchor"]
)
print("Test shape:", test_df.shape)
print(test_df.head())




## === cell 3
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 8),
    min_df=1,
    max_features=600_000,
    sublinear_tf=True,
)
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=1,
    max_features=600_000,
    sublinear_tf=True,
)
anchor_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=200_000,
    sublinear_tf=True,
)
target_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=200_000,
    sublinear_tf=True,
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["input"].values,
    train_df["score"].values,
    test_size=0.25,
    random_state=42,
    stratify=train_df["score"],
)

X_train_char = char_vectorizer.fit_transform(train_texts)
X_val_char = char_vectorizer.transform(val_texts)

X_train_word = word_vectorizer.fit_transform(train_texts)
X_val_word = word_vectorizer.transform(val_texts)

train_mask = train_df["input"].isin(train_texts)
val_mask = ~train_mask

X_train_anchor = anchor_vectorizer.fit_transform(
    train_df.loc[train_mask, "anchor"].astype(str).values
)
X_val_anchor = anchor_vectorizer.transform(
    train_df.loc[val_mask, "anchor"].astype(str).values
)

X_train_target = target_vectorizer.fit_transform(
    train_df.loc[train_mask, "target"].astype(str).values
)
X_val_target = target_vectorizer.transform(
    train_df.loc[val_mask, "target"].astype(str).values
)


def numeric_features(df):
    anchor_len = df["anchor"].astype(str).apply(len).values.astype(float)
    target_len = df["target"].astype(str).apply(len).values.astype(float)
    context_len = df["context"].astype(str).apply(len).values.astype(float)

    anchor_tok = df["anchor"].astype(str).str.split()
    target_tok = df["target"].astype(str).str.split()
    context_tok = df["context"].astype(str).str.split()

    overlap_at = anchor_tok.combine(
        target_tok, lambda a, b: len(set(a).intersection(set(b)))
    ).astype(float)
    union_at = anchor_tok.combine(
        target_tok, lambda a, b: len(set(a).union(set(b)))
    ).astype(float)
    overlap_ratio_at = np.divide(
        overlap_at, union_at, out=np.zeros_like(overlap_at), where=union_at != 0
    )

    overlap_ac = anchor_tok.combine(
        context_tok, lambda a, b: len(set(a).intersection(set(b)))
    ).astype(float)
    union_ac = anchor_tok.combine(
        context_tok, lambda a, b: len(set(a).union(set(b)))
    ).astype(float)
    overlap_ratio_ac = np.divide(
        overlap_ac, union_ac, out=np.zeros_like(overlap_ac), where=union_ac != 0
    )

    overlap_tc = target_tok.combine(
        context_tok, lambda a, b: len(set(a).intersection(set(b)))
    ).astype(float)
    union_tc = target_tok.combine(
        context_tok, lambda a, b: len(set(a).union(set(b)))
    ).astype(float)
    overlap_ratio_tc = np.divide(
        overlap_tc, union_tc, out=np.zeros_like(overlap_tc), where=union_tc != 0
    )

    len_diff = np.abs(anchor_len - target_len)
    len_ratio = np.divide(
        anchor_len, target_len, out=np.zeros_like(anchor_len), where=target_len != 0
    )

    return np.vstack(
        [
            anchor_len,
            target_len,
            context_len,
            overlap_at,
            overlap_ratio_at,
            len_diff,
            len_ratio,
            overlap_ac,
            overlap_ratio_ac,
            overlap_tc,
            overlap_ratio_tc,
        ]
    ).T.astype(float)


train_num_raw = numeric_features(train_df.loc[train_mask])
val_num_raw = numeric_features(train_df.loc[val_mask])

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num_raw)
val_num_scaled = scaler.transform(val_num_raw)

train_num = csr_matrix(train_num_scaled)
val_num = csr_matrix(val_num_scaled)

X_train = hstack(
    [X_train_char, X_train_word, X_train_anchor, X_train_target, train_num]
)
X_val = hstack([X_val_char, X_val_word, X_val_anchor, X_val_target, val_num])




## === cell 4
text_model = Ridge(alpha=0.005, solver="lsqr")
text_model.fit(X_train, train_labels)

num_model = Ridge(alpha=0.05, solver="lsqr")
num_model.fit(train_num, train_labels)

val_pred_text = text_model.predict(X_val)
val_pred_num = num_model.predict(val_num)

A = np.vstack([val_pred_text, val_pred_num, np.ones_like(val_pred_text)]).T
w_text, w_num, b_blend = np.linalg.lstsq(A, val_labels, rcond=None)[0]

val_pred_raw = w_text * val_pred_text + w_num * val_pred_num + b_blend
val_pred_raw = np.clip(val_pred_raw, 0, 1)
val_pearson = pearsonr(val_labels, val_pred_raw)[0]
print(f"Validation Pearson after linear blend: {val_pearson:.5f}")

A2 = np.vstack([val_pred_raw, np.ones_like(val_pred_raw)]).T
a_cal, b_cal = np.linalg.lstsq(A2, val_labels, rcond=None)[0]
val_pred = a_cal * val_pred_raw + b_cal
val_pred = np.clip(val_pred, 0, 1)
val_pearson_cal = pearsonr(val_labels, val_pred)[0]
print(f"Validation Pearson after calibration: {val_pearson_cal:.5f}")




## === cell 5
X_test_char = char_vectorizer.transform(test_df["input"].values)
X_test_word = word_vectorizer.transform(test_df["input"].values)
X_test_anchor = anchor_vectorizer.transform(test_df["anchor"].astype(str).values)
X_test_target = target_vectorizer.transform(test_df["target"].astype(str).values)

test_num_raw = numeric_features(test_df)
test_num_scaled = scaler.transform(test_num_raw)
test_num = csr_matrix(test_num_scaled)

X_test = hstack([X_test_char, X_test_word, X_test_anchor, X_test_target, test_num])

test_pred_text = text_model.predict(X_test)
test_pred_num = num_model.predict(test_num)

test_pred_raw = w_text * test_pred_text + w_num * test_pred_num + b_blend
test_pred_raw = np.clip(test_pred_raw, 0, 1)

test_pred = a_cal * test_pred_raw + b_cal
test_pred = np.clip(test_pred, 0, 1)

submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}, shape: {submission.shape}")
