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

0.8046283730655472

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59036) has done: 'The fix changes the Ridge regression solver to one compatible with the current SciPy version (`solver='lsqr'`) so the model can be trained and predictions generated without errors. This also ensures the `test_pred` variable is defined for the submission step, allowing a proper CSV file to be written.'
- What this solution (achieved 0.61556) has done: 'I keep the overall pipeline unchanged but improve the model’s ability to capture the data by (1) removing the artificial feature‑size caps so the TF‑IDF vectors can use all available tokens, (2) lowering the Ridge regularisation strength (`alpha=0.01`) to let the model fit more closely, and (3) after the validation step, refitting the vectorizers and Ridge model on the **full** training set before predicting the test set. These minimal tweaks stay within the existing linear‑model framework and should raise the Pearson correlation toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.65022) has done: 'I add a few lightweight numeric features (lengths, common‑word count, character‑length difference) and concatenate them to the existing TF‑IDF matrices, then lower the Ridge regularisation slightly (α=0.001) so the model can use the richer feature set. These small augmentations keep the original linear‑model pipeline unchanged while giving the model more signal, which should raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.66564) has done: 'I add a normalized common‑word similarity feature (`common_ratio`) to the numeric‑feature set and slightly decrease the Ridge regularisation (`alpha=0.0005`). These lightweight tweaks keep the linear‑model pipeline unchanged while giving the model a bit more signal, which should raise the Pearson correlation toward the target.'
- What this solution (achieved 0.66413) has done: 'I add a few more lightweight numeric features (token‑count difference and sum) to the existing extra feature set and slightly lower the Ridge regularisation (α = 0.0001) so the linear model can fit the data a bit more closely. These changes keep the overall pipeline unchanged while giving the model a bit more signal, which should raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.66415) has done: 'I add a small hyper‑parameter search over a few Ridge regularisation strengths, pick the α that gives the highest validation Pearson correlation, and then use that α for the final model trained on the full data. This keeps the original linear‑model pipeline unchanged while likely raising the score toward the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy.sparse import hstack, csr_matrix

BASE_DIR = "/kaggle/input/us-patent-phrase-to-phrase-matching"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

print("Files available:")
for root, _, files in os.walk(BASE_DIR):
    for f in files:
        print(os.path.join(root, f))



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print(train_df.head())




## === cell 2
def combine_fields(row):
    return f"{row['anchor']} {row['target']} {row['context']}"


def extra_features(df):
    anchor = df["anchor"].fillna("").astype(str)
    target = df["target"].fillna("").astype(str)
    context = df["context"].fillna("").astype(str)

    len_anchor = anchor.str.split().apply(len)
    len_target = target.str.split().apply(len)
    len_context = context.str.split().apply(len)

    token_diff = (len_anchor - len_target).abs()
    token_sum = len_anchor + len_target

    common_words = (
        anchor.str.split()
        .apply(set)
        .combine(target.str.split().apply(set), lambda a, b: len(a & b))
    )

    total_unique = (
        anchor.str.split()
        .apply(set)
        .combine(target.str.split().apply(set), lambda a, b: len(a | b))
    )
    common_ratio = common_words / (total_unique + 1e-6)

    char_len_anchor = anchor.str.len()
    char_len_target = target.str.len()
    char_len_diff = (char_len_anchor - char_len_target).abs()

    return pd.DataFrame(
        {
            "len_anchor": len_anchor,
            "len_target": len_target,
            "len_context": len_context,
            "token_diff": token_diff,
            "token_sum": token_sum,
            "common_words": common_words,
            "common_ratio": common_ratio,
            "char_len_diff": char_len_diff,
        }
    )


def row_cosine_similarity(A, B):
    """Return an array with cosine similarity for each row of sparse matrices A and B."""
    dot = A.multiply(B).sum(axis=1)
    norm_a = np.sqrt(A.multiply(A).sum(axis=1))
    norm_b = np.sqrt(B.multiply(B).sum(axis=1))
    return np.array(dot / (norm_a * norm_b + 1e-6)).ravel()


train_idx, val_idx = train_test_split(train_df.index, test_size=0.2, random_state=42)

train_text = train_df.loc[train_idx].apply(combine_fields, axis=1)
val_text = train_df.loc[val_idx].apply(combine_fields, axis=1)

train_y = train_df.loc[train_idx, "score"]
val_y = train_df.loc[val_idx, "score"]

train_extra = csr_matrix(extra_features(train_df.loc[train_idx]).values)
val_extra = csr_matrix(extra_features(train_df.loc[val_idx]).values)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)
X_train_word = word_vectorizer.fit_transform(train_text)
X_val_word = word_vectorizer.transform(val_text)
X_test_word = word_vectorizer.transform(test_df.apply(combine_fields, axis=1))

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
)
X_train_char = char_vectorizer.fit_transform(train_text)
X_val_char = char_vectorizer.transform(val_text)
X_test_char = char_vectorizer.transform(test_df.apply(combine_fields, axis=1))

phrase_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)

anchor_train = phrase_vec.fit_transform(
    train_df.loc[train_idx, "anchor"].fillna("").astype(str)
)
target_train = phrase_vec.transform(
    train_df.loc[train_idx, "target"].fillna("").astype(str)
)
anchor_val = phrase_vec.transform(
    train_df.loc[val_idx, "anchor"].fillna("").astype(str)
)
target_val = phrase_vec.transform(
    train_df.loc[val_idx, "target"].fillna("").astype(str)
)

cos_train = row_cosine_similarity(anchor_train, target_train)
cos_val = row_cosine_similarity(anchor_val, target_val)

X_train = hstack(
    [X_train_word, X_train_char, train_extra, csr_matrix(cos_train).transpose()]
)
X_val = hstack([X_val_word, X_val_char, val_extra, csr_matrix(cos_val).transpose()])
X_test = hstack([X_test_word, X_test_char, csr_matrix(extra_features(test_df).values)])

print("Combined feature shape (train split):", X_train.shape)

candidate_alphas = [0.001, 0.0005, 0.0002, 0.00015, 0.0001, 0.00005, 0.00001]
best_alpha = candidate_alphas[0]
best_pearson = -np.inf

for a in candidate_alphas:
    tmp_model = Ridge(alpha=a, solver="lsqr", random_state=42)
    tmp_model.fit(X_train, train_y)
    val_pred_tmp = tmp_model.predict(X_val)
    pearson_tmp, _ = pearsonr(val_y, val_pred_tmp)
    if pearson_tmp > best_pearson:
        best_pearson = pearson_tmp
        best_alpha = a

print(f"Best alpha from validation search: {best_alpha} (Pearson {best_pearson:.6f})")

model = Ridge(alpha=best_alpha, solver="lsqr", random_state=42)
model.fit(X_train, train_y)

val_pred = model.predict(X_val)
pearson, _ = pearsonr(val_y, val_pred)
print(f"Validation Pearson correlation (80% train) with best alpha: {pearson:.6f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/541441291.py in <cell line: 0>()
    113 )
    114 
--> 115 cos_train = row_cosine_similarity(anchor_train, target_train)
    116 cos_val = row_cosine_similarity(anchor_val, target_val)
    117 

/tmp/ipykernel_11/541441291.py in row_cosine_similarity(A, B)
     51     norm_a = np.sqrt(A.multiply(A).sum(axis=1))
     52     norm_b = np.sqrt(B.multiply(B).sum(axis=1))
---> 53     return np.array(dot / (norm_a * norm_b + 1e-6)).ravel()
     54 
     55 

/usr/local/lib/python3.11/dist-packages/numpy/matrixlib/defmatrix.py in __mul__(self, other)
    217         if isinstance(other, (N.ndarray, list, tuple)) :
    218             # This promotes 1-D vectors to row vectors
--> 219             return N.dot(self, asmatrix(other))
    220         if isscalar(other) or not hasattr(other, '__rmul__') :
    221             return N.dot(self, other)

ValueError: shapes (26260,1) and (26260,1) not aligned: 1 (dim 1) != 26260 (dim 0)

## === cell 3
full_word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)

full_char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
)

train_full_text = train_df.apply(combine_fields, axis=1)

X_full_word = full_word_vec.fit_transform(train_full_text)
X_full_char = full_char_vec.fit_transform(train_full_text)
full_extra = csr_matrix(extra_features(train_df).values)

phrase_vec_full = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)

anchor_full = phrase_vec_full.fit_transform(train_df["anchor"].fillna("").astype(str))
target_full = phrase_vec_full.transform(train_df["target"].fillna("").astype(str))

cos_full = row_cosine_similarity(anchor_full, target_full)

X_full = hstack(
    [X_full_word, X_full_char, full_extra, csr_matrix(cos_full).transpose()]
)

full_model = Ridge(alpha=best_alpha, solver="lsqr", random_state=42)
full_model.fit(X_full, train_df["score"])

X_test_full_word = full_word_vec.transform(test_df.apply(combine_fields, axis=1))
X_test_full_char = full_char_vec.transform(test_df.apply(combine_fields, axis=1))
test_extra = csr_matrix(extra_features(test_df).values)

anchor_test = phrase_vec_full.transform(test_df["anchor"].fillna("").astype(str))
target_test = phrase_vec_full.transform(test_df["target"].fillna("").astype(str))
cos_test = row_cosine_similarity(anchor_test, target_test)

X_test_full = hstack(
    [
        X_test_full_word,
        X_test_full_char,
        test_extra,
        csr_matrix(cos_test).transpose(),
    ]
)

test_pred = full_model.predict(X_test_full)
test_pred = np.clip(test_pred, 0.0, 1.0)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3006500275.py in <cell line: 0>()
     33 target_full = phrase_vec_full.transform(train_df["target"].fillna("").astype(str))
     34 
---> 35 cos_full = row_cosine_similarity(anchor_full, target_full)
     36 
     37 X_full = hstack(

/tmp/ipykernel_11/541441291.py in row_cosine_similarity(A, B)
     51     norm_a = np.sqrt(A.multiply(A).sum(axis=1))
     52     norm_b = np.sqrt(B.multiply(B).sum(axis=1))
---> 53     return np.array(dot / (norm_a * norm_b + 1e-6)).ravel()
     54 
     55 

/usr/local/lib/python3.11/dist-packages/numpy/matrixlib/defmatrix.py in __mul__(self, other)
    217         if isinstance(other, (N.ndarray, list, tuple)) :
    218             # This promotes 1-D vectors to row vectors
--> 219             return N.dot(self, asmatrix(other))
    220         if isscalar(other) or not hasattr(other, '__rmul__') :
    221             return N.dot(self, other)

ValueError: shapes (32825,1) and (32825,1) not aligned: 1 (dim 1) != 32825 (dim 0)

## === cell 4
submission = pd.read_csv(sample_sub_path)
submission["score"] = test_pred
submission = submission[["id", "score"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(submission.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1304022703.py in <cell line: 0>()
      1 submission = pd.read_csv(sample_sub_path)
----> 2 submission["score"] = test_pred
      3 submission = submission[["id", "score"]]
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv")

NameError: name 'test_pred' is not defined
