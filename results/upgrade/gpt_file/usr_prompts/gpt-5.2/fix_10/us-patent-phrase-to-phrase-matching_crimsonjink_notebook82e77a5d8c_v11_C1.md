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

0.15691

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35787) has done: 'Your current pipeline is scoring far above the target (0.39365 vs 0.2163), so the goal is to *reduce* performance slightly toward the target band with the smallest safe change. The least invasive way is to keep the same TF‑IDF + LogisticRegression core, but prevent leakage by adding a proper train/validation split and selecting a more conservative model snapshot (stronger regularization) based on validation Pearson rather than fitting on all data at once. This typically lowers generalization (and thus public LB) while still producing a valid submission and keeping the same modeling approach. I also switch from hard class predictions to expected-value regression using `predict_proba` mapped onto the original score levels, which is closer to the Pearson objective and stabilizes outputs while still being minimal.'
- What this solution (achieved 0.35787) has done: 'Your current score (0.35787) is well above the target (0.2163), so we should *reduce* performance slightly with the smallest safe change while keeping the same TF‑IDF + cuML LogisticRegression approach. The least invasive knob is to increase regularization by searching smaller `C` values and then using the most conservative `C` among those that are still “good enough” on validation (this typically lowers LB without changing the modeling core). I keep your Pearson-based validation selection, but add a small tolerance rule so we pick the smallest `C` within a close range of the best validation correlation. This preserves semantics, still trains on all data for the final model, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.35787) has done: 'Your current score is well above the target (0.35787 vs 0.2163), so to move *toward* the target we should slightly and safely reduce generalization while keeping the same TF‑IDF + cuML LogisticRegression core. The smallest reliable knob is stronger regularization, so I extend the `C_grid` to include smaller values and then choose a more conservative `C` by widening the “within tolerance of best validation Pearson” rule. This keeps the same training loop and Pearson-based model selection semantics, but intentionally picks a simpler model likely to score closer to the target. The submission writing and row alignment stay unchanged.'
- What this solution (achieved 0.25753) has done: 'Your current score (0.35787) is well above the target (0.2163), so we should deliberately reduce performance slightly with the smallest safe knob while keeping the same TF‑IDF + cuML LogisticRegression core and the same Pearson-based validation selection logic. The minimal way to do that is to widen the “within tolerance of best validation Pearson” rule so we more aggressively choose a smaller (more regularized) `C`, which typically underfits a bit and lowers LB correlation. I keep the same train/valid split, same vectorization, same expected-value-from-probabilities prediction, and the same submission writing. This should move the score downward toward the target band without changing the overall approach.'
- What this solution (achieved 0.16563) has done: 'Your current score (0.25753) is above the target (0.2163), so we should *slightly reduce* performance with the smallest safe knob while keeping the same TF‑IDF + cuML LogisticRegression core and the same expected-value-from-probabilities prediction. The most direct minimal change is to pick a more conservative (more regularized) `C` by widening the “within tolerance of best validation Pearson” rule so smaller `C` values become eligible more often. This preserves the same training/validation procedure and model family, but intentionally underfits a bit to move the LB score downward toward the target band. The rest of the pipeline (data loading, vectorization, fitting on all data, and submission writing) remains unchanged.'
- What this solution (achieved 0.16563) has done: 'Your current score (0.16563) is below the target (0.2163), so we should cautiously improve generalization without changing the core TF‑IDF + cuML LogisticRegression approach. The smallest high-impact fix is to remove an unintended feature-collision bug: you are reusing the *same* `TfidfVectorizer` instance for both `anchor` and `target`, which forces both columns to share one vocabulary and can hurt performance. I switch to two separate vectorizers (same settings) inside the same `ColumnTransformer`, keeping everything else (split, C-grid search, expected-value prediction, submission writing) identical. This typically improves correlation while staying within the same modeling semantics and runtime limits.'
- What this solution (achieved 0.14639) has done: 'Your current score (0.16563) is below the target (0.2163), so we should make a small, safe improvement without changing the TF‑IDF + cuML LogisticRegression core. The most direct minimal gain for Pearson here is to add `context` back as a simple categorical feature (one-hot), because similarity is conditioned on CPC class and this typically boosts correlation. This keeps the same train/valid split, the same C-grid loop, the same expected-value-from-probabilities prediction, and the same submission writing—just a slightly richer feature set. I also keep the two separate TF‑IDF vectorizers to avoid the earlier vocabulary-collision issue.'
- What this solution (achieved 0.15691) has done: 'You’re currently below the target (0.14639 vs 0.2163), so we should make a small, legitimate improvement without changing the TF‑IDF + (cuML) LogisticRegression core. The most direct low-risk boost for Pearson here is to use stronger text normalization in TF‑IDF (sublinear TF scaling + stopword removal + slightly higher `min_df`) which often improves semantic matching signal and reduces noise. I keep your same train/valid split, same C-grid search and tolerance selection, same expected-value-from-probabilities mapping, and the same submission writing. I also set a deterministic `random_state` in cuML LogisticRegression for stability (should not change core logic).'
- What this solution (achieved 0.15691) has done: 'You’re currently below the target (0.15691 vs 0.2163), so we should make a small, safe improvement without changing the TF‑IDF + cuML LogisticRegression core. The most direct low-risk boost for Pearson here is to correct a small train/test feature mismatch: `score_levels` is derived from `LabelEncoder.inverse_transform(...)` which can be brittle; instead we should map class indices to the true numeric score values via `le.classes_` (guaranteed aligned with `predict_proba` columns). I also set `max_iter` a bit higher to reduce under-convergence risk (same model, just more iterations), which commonly improves correlation slightly without changing the approach. Everything else (split, vectorization, C-grid search + tolerance selection, expected-value prediction, and submission writing) stays the same.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
X = train.drop(["id", "score"], axis=1)

y_enc = le.fit_transform(y).astype("float32")

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)



## === cell 3
vectorizer_anchor = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=2,
)
vectorizer_target = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=2,
)

context_ohe = OneHotEncoder(handle_unknown="ignore")

transformer = make_column_transformer(
    (vectorizer_anchor, "anchor"),
    (vectorizer_target, "target"),
    (context_ohe, ["context"]),
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


score_levels = le.classes_.astype(np.float32)

C_grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

results = []
for C in C_grid:
    model = LogisticRegression(C=C, random_state=42, max_iter=2000)
    model.fit(X_tr_vec, y_tr)

    va_proba = model.predict_proba(X_va_vec)
    va_pred_score = (va_proba * score_levels[None, :]).sum(axis=1)

    va_true_score = le.inverse_transform(y_va.astype(np.int32)).astype(np.float32)
    corr = pearson_corr(va_true_score, va_pred_score)
    results.append((C, corr))

best_corr = max(c for _, c in results)

tol = 0.18
eligible = [(C, c) for (C, c) in results if c >= best_corr - tol]
best_C, chosen_corr = min(eligible, key=lambda x: x[0])



## === cell 5
X_all_vec = transformer.fit_transform(train[["anchor", "target", "context"]])
y_all_enc = y_enc

final_model = LogisticRegression(C=best_C, random_state=42, max_iter=2000)
final_model.fit(X_all_vec, y_all_enc)



## === cell 6
t = test[["anchor", "target", "context"]]
t_vec = transformer.transform(t)

test_proba = final_model.predict_proba(t_vec)
pred_score = (test_proba * score_levels[None, :]).sum(axis=1)

sample["score"] = pred_score.astype(np.float32)
sample.to_csv("submission.csv", index=False)
