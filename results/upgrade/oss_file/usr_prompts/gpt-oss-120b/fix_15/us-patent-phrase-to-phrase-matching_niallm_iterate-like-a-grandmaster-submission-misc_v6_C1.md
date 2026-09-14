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

0.8084755399348172

# 6. Current score

0.65139

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38176) has done: 'The fix updates the Ridge regression to use a solver compatible with the available SciPy version (`solver='lsqr'`), preventing the `cg`‑related TypeError and allowing the model to train. After training, predictions are clipped to the valid 0‑1 score range to keep the submission format correct and potentially improve the Pearson score. No other logic is altered, preserving the original workflow.'
- What this solution (achieved 0.5186) has done: 'I add a complementary word‑level TF‑IDF representation and combine it with the existing character‑level TF‑IDF using a sparse hstack. This richer feature set typically captures semantic similarity better and should raise the Pearson correlation toward the target. I also slightly lower the Ridge regularisation (alpha = 0.5) to let the model fit the data more closely while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.56861) has done: 'I add two simple similarity features (token‑overlap ratio and character‑length difference) to the existing TF‑IDF representation and lower the Ridge regularisation (α = 0.2). These extra numeric cues often boost Pearson correlation without altering the core pipeline, and the smaller α lets the model utilise the richer feature set more fully, moving the score toward the target.'
- What this solution (achieved 0.57149) has done: 'I add two inexpensive similarity features (log‑scaled length difference and length‑ratio) to the existing overlap feature, and lower the Ridge regularisation strength (α = 0.1). These small enrichments keep the original TF‑IDF pipeline intact while giving the linear model more useful signals, which should raise the Pearson correlation toward the target.'
- What this solution (achieved 0.57986) has done: 'I add a few inexpensive similarity cues to the feature set—character‑level Jaccard overlap, raw character‑length difference, and word‑count ratio—and switch the ridge regularisation to a slightly lower α (0.05) so the linear model can exploit the richer signals. I also enable sublinear TF for the character TF‑IDF vectoriser (a harmless tweak that often improves text similarity models). These minimal adjustments keep the overall pipeline unchanged while giving the model more useful information, which should raise the validation Pearson toward the target.'
- What this solution (achieved 0.57329) has done: 'I slightly broaden the character‑level TF‑IDF (using 1‑3‑grams and more features) and lower the Ridge regularisation from 0.05 to 0.01. These minimal tweaks give the linear model richer text signals while allowing it to fit them a bit more closely, which should raise the Pearson correlation toward the target without changing the overall pipeline.'
- What this solution (achieved 0.62659) has done: 'I keep the overall pipeline but make several small, targeted tweaks that are expected to raise the validation Pearson score: use a random train/validation split (often yields a higher correlation on the held‑out set), add two interaction‑style numeric features to the extra feature set, increase the TF‑IDF capacities slightly, and remove the ridge regularisation (α = 0) so the linear model can fully exploit the richer feature space. These changes preserve the core model and data handling while nudging the score toward the target.'
- What this solution (achieved 0.63174) has done: 'I keep the existing TF‑IDF and extra numeric features but add a tiny linear blending step that combines the Ridge prediction with the simple token‑overlap feature already computed in `extra_features`. By fitting a one‑step linear regression on the validation split, we can slightly boost the Pearson correlation toward the target without changing the core model or training process. The blended weights are then applied to the test predictions before creating the submission.'
- What this solution (achieved 0.64052) has done: 'I add feature scaling for the numeric similarity cues and let the blending step use **all** of those scaled features instead of only the token‑overlap. The extra features are standardized (variance scaling) before being concatenated with the TF‑IDF matrices, and the Ridge model receives a tiny L2 regularisation (α = 0.1) to improve generalisation. The linear blend now combines the Ridge prediction with the eight scaled similarity features, which should raise the Pearson correlation toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.63886) has done: 'I slightly enlarge the TF‑IDF vocabularies (more character‑ and word‑ngrams) and reduce the Ridge regularisation from 0.1 to 0.05. These adjustments keep the overall pipeline unchanged while giving the linear model a richer signal and a bit more flexibility, which should raise the validation Pearson closer to the target score.'
- What this solution (achieved 0.65139) has done: 'I slightly enrich the handcrafted similarity features by adding their squared terms (giving the linear model a bit more expressive power) and broaden the word‑level TF‑IDF to 1‑4‑grams with a modest increase in vocabulary size. These minimal changes keep the original pipeline intact while providing extra signals that should raise the validation Pearson correlation toward the target.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy import sparse




## === cell 1
def locate_file(rel_path):
    """
    Locate a file that may reside in one of several common Kaggle directories.
    """
    candidates = [
        Path("data") / rel_path,
        Path("input") / rel_path,
        Path("kaggle") / "input" / rel_path,
        Path("/kaggle") / "input" / rel_path,  # absolute fallback for Kaggle kernels
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not locate {rel_path} in any known location.")


base_rel = Path("us-patent-phrase-to-phrase-matching")
train_path = locate_file(base_rel / "train.csv")
test_path = locate_file(base_rel / "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
def make_input(df):
    return (
        df["context"].fillna("")
        + " "
        + df["anchor"].fillna("")
        + " "
        + df["target"].fillna("")
    )


train_df["inputs"] = make_input(train_df)
test_df["inputs"] = make_input(test_df)



## === cell 3
np.random.seed(42)

train_split, val_split = train_test_split(train_df, test_size=0.25, random_state=42)

X_train, y_train = train_split["inputs"].values, train_split["score"].values
X_val, y_val = val_split["inputs"].values, val_split["score"].values

char_vectorizer = TfidfVectorizer(
    max_features=400000,  # increased from 300k
    ngram_range=(1, 3),
    analyzer="char",
    sublinear_tf=True,
)

word_vectorizer = TfidfVectorizer(
    max_features=300000,  # modest increase
    ngram_range=(1, 4),  # broadened from (1,3)
    analyzer="word",
    stop_words=None,
    sublinear_tf=True,
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_train_word = word_vectorizer.fit_transform(X_train)

X_val_char = char_vectorizer.transform(X_val)
X_val_word = word_vectorizer.transform(X_val)




## === cell 4
def extra_features(df):
    anchor = df["anchor"].fillna("").astype(str)
    target = df["target"].fillna("").astype(str)
    overlap = []  # token‑level Jaccard
    log_char_diff = []  # log1p of character length difference
    length_ratio = []  # min/max character length ratio
    char_overlap = []  # character‑level Jaccard
    raw_char_diff = []  # absolute character length difference
    word_len_ratio = []  # token‑count ratio
    overlap_len_ratio = []  # overlap * length_ratio
    char_overlap_word_ratio = []  # char_overlap * word_len_ratio

    for a, t in zip(anchor, target):
        a_set = set(a.split())
        t_set = set(t.split())
        union = a_set | t_set
        inter = a_set & t_set
        ov = len(inter) / len(union) if union else 0.0
        overlap.append(ov)

        diff = abs(len(a) - len(t))
        log_char_diff.append(np.log1p(diff))
        raw_char_diff.append(diff)

        la, lt = len(a), len(t)
        max_len = max(la, lt)
        lr = min(la, lt) / max_len if max_len > 0 else 0.0
        length_ratio.append(lr)

        a_chars = set(a)
        t_chars = set(t)
        char_union = a_chars | t_chars
        char_inter = a_chars & t_chars
        co = len(char_inter) / len(char_union) if char_union else 0.0
        char_overlap.append(co)

        wa, wt = len(a_set), len(t_set)
        max_w = max(wa, wt)
        wl = min(wa, wt) / max_w if max_w > 0 else 0.0
        word_len_ratio.append(wl)

        overlap_len_ratio.append(ov * lr)
        char_overlap_word_ratio.append(co * wl)

    base_feats = np.column_stack(
        [
            overlap,
            log_char_diff,
            length_ratio,
            char_overlap,
            raw_char_diff,
            word_len_ratio,
            overlap_len_ratio,
            char_overlap_word_ratio,
        ]
    )
    squared_feats = base_feats**2
    return np.column_stack([base_feats, squared_feats])


extra_train = extra_features(train_split)
extra_val = extra_features(val_split)

extra_train_sparse = sparse.csr_matrix(extra_train)
extra_val_sparse = sparse.csr_matrix(extra_val)

scaler = StandardScaler(with_mean=False)
extra_train_scaled = scaler.fit_transform(extra_train_sparse)
extra_val_scaled = scaler.transform(extra_val_sparse)

X_train_vec = sparse.hstack([X_train_char, X_train_word, extra_train_scaled])
X_val_vec = sparse.hstack([X_val_char, X_val_word, extra_val_scaled])



## === cell 5
model = Ridge(alpha=0.05, solver="lsqr")
model.fit(X_train_vec, y_train)

val_preds_ridge = model.predict(X_val_vec)

extra_val_scaled_dense = extra_val_scaled.toarray()
X_blend = np.column_stack(
    [val_preds_ridge, extra_val_scaled_dense, np.ones_like(val_preds_ridge)]
)
blend_weights, _, _, _ = np.linalg.lstsq(X_blend, y_val, rcond=None)

val_preds_blend = X_blend @ blend_weights
pearson_blend, _ = pearsonr(y_val, val_preds_blend)
print(f"Validation Pearson (Ridge only): {pearsonr(y_val, val_preds_ridge)[0]:.6f}")
print(f"Validation Pearson (Blended): {pearson_blend:.6f}")



## === cell 6
X_test_char = char_vectorizer.transform(test_df["inputs"].values)
X_test_word = word_vectorizer.transform(test_df["inputs"].values)

extra_test = extra_features(test_df)
extra_test_sparse = sparse.csr_matrix(extra_test)
extra_test_scaled = scaler.transform(extra_test_sparse)

X_test_vec = sparse.hstack([X_test_char, X_test_word, extra_test_scaled])

test_preds_ridge = model.predict(X_test_vec)

extra_test_scaled_dense = extra_test_scaled.toarray()
X_test_blend = np.column_stack(
    [test_preds_ridge, extra_test_scaled_dense, np.ones_like(test_preds_ridge)]
)
test_preds_blend = X_test_blend @ blend_weights

test_preds = np.clip(test_preds_blend, 0.0, 1.0)



## === cell 7
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path.resolve()}")
