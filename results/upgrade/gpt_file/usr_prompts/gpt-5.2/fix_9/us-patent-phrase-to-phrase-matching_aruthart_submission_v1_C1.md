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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.7716525329500983

# 6. Current score

0.5945

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52246) has done: 'I fix the runtime error by forcing Ridge to use a solver that doesn’t call SciPy’s `cg()` with the incompatible `tol` argument (this is what currently crashes your fit). I keep the exact same TF‑IDF + Ridge core approach, only changing the Ridge solver selection to a stable alternative for sparse TF‑IDF matrices. Then I ensure `preds` is always defined before building the submission and write `submission.csv` with the required `id,score` columns. These changes are execution/stability fixes and should produce a valid submission without altering the overall modeling semantics beyond the necessary solver compatibility.'
- What this solution (achieved 0.65586) has done: 'Your current TF‑IDF + Ridge setup is underperforming likely because it’s trying to learn absolute scores from raw concatenated text rather than modeling similarity between anchor/target. To move the Pearson score toward your target while keeping the same core approach (TF‑IDF features + Ridge regression in a sklearn Pipeline), I switch the text representation to a minimal “pairwise” feature: TF‑IDF on the anchor and target separately, then combine them as (a) anchor features, (b) target features, and (c) their elementwise absolute difference—this stays in the same linear model family and keeps the same loss/fit semantics. I also add a tiny amount of extra signal by including character n‑grams (still TF‑IDF) in a FeatureUnion, which often helps with abbreviations/typos in this specific competition. These are small, legitimate feature-extraction changes intended to improve correlation without changing the model class or training loop.'
- What this solution (achieved 0.64038) has done: 'Your current gap to target is 0.77165 − 0.65586 ≈ 0.1158 (about 15%), so we should improve score but keep the same TF‑IDF + Ridge core. The smallest likely win in this competition is better text normalization and using context as a separate field rather than duplicating it inside both sides, because duplicates can drown out anchor/target differences and hurt correlation. I (1) normalize text (lowercasing + light whitespace cleanup) and (2) build left/right as just anchor/target, while providing context as its own TF‑IDF block concatenated to the pairwise features; this preserves the same modeling family (sparse TF‑IDF features + Ridge) and training semantics. I also set `norm="l2"` and `sublinear_tf=True` on vectorizers (still TF‑IDF) which usually improves linear regression correlation on short phrases without changing the approach.'
- What this solution (achieved 0.63779) has done: 'To move your Pearson score upward toward the 0.77165 target without changing the core “TF‑IDF features + Ridge regression” approach, I make two minimal, high-impact adjustments: (1) add an explicit interaction feature between anchor and target via elementwise product (in addition to the existing absolute-difference), and (2) mildly retune Ridge regularization (alpha) to better fit this richer sparse feature space. Both changes keep the same model family, training loop, and loss semantics, and they’re legitimate feature-engineering tweaks that typically improve correlation for phrase-pair similarity. I also keep the existing text normalization and context block unchanged, and ensure the pipeline still writes a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.63144) has done: 'Your current score (0.63779) is well below the target (0.77165), so we should cautiously improve correlation while keeping the same TF‑IDF + Ridge core. The smallest high-impact change for this competition is to train the linear model to better match Pearson correlation by standardizing the target during fitting (and inverting the transform on predictions), which preserves the same model family and loss but improves calibration/conditioning. I also very lightly retune Ridge regularization (alpha) to better suit the current richer feature space (word/char + diff + product + context) without changing the approach. Everything else (feature construction, training flow, output schema) is kept intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.5945) has done: 'Your current gap to the target is large (0.77165 − 0.63144 ≈ 0.1402, ~18%), so we should improve correlation while keeping the same TF‑IDF + Ridge core. The smallest high-impact change in this competition is to better align the regression output with the label distribution by snapping predictions to the allowed 0.25 grid (0, 0.25, 0.5, 0.75, 1.0), which often improves Pearson because labels are quantized. I also add a very small amount of extra context signal by including a word-based TF‑IDF for `context` in addition to the existing char-based one, without changing the model class or training approach. Everything else (pairwise TF‑IDF blocks, Ridge with transformed target, train/test flow, submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "upb"

import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/us-patent-phrase-to-phrase-matching"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train_df.head()



## === cell 2
import re

_ws_re = re.compile(r"\s+")


def _norm_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("")
    s = s.str.lower()
    s = s.apply(lambda x: _ws_re.sub(" ", x).strip())
    return s


def make_text_cols(df: pd.DataFrame):
    context = _norm_text(df["context"])
    anchor = _norm_text(df["anchor"])
    target = _norm_text(df["target"])
    left = anchor
    right = target
    return left, right, context


X_train_left, X_train_right, X_train_ctx = make_text_cols(train_df)
y_train = train_df["score"].astype(float).values
X_test_left, X_test_right, X_test_ctx = make_text_cols(test_df)

(X_train_left.iloc[0], X_train_right.iloc[0], X_train_ctx.iloc[0], y_train[0])



## === cell 3
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


class PairwiseTfidfFeatures(BaseEstimator, TransformerMixin):
    """
    TF-IDF + linear model core preserved:
    - Pairwise features for (left, right, |left-right|, left*right)
    - word + char_wb TF-IDF
    - Adds context TF-IDF as a separate block (no duplication).

    Minimal improvement: add an extra *word* TF-IDF for context (in addition to char context),
    which often helps because context is a short structured code that also benefits from token-level features.
    """

    def __init__(
        self,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        min_df=2,
        max_features_word=200000,
        max_features_char=100000,
        max_features_ctx=20000,
        max_features_ctx_word=5000,
    ):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self.max_features_ctx = max_features_ctx
        self.max_features_ctx_word = max_features_ctx_word

        self.word_vectorizer_ = None
        self.char_vectorizer_ = None
        self.ctx_vectorizer_ = None
        self.ctx_word_vectorizer_ = None

    def fit(self, X, y=None):
        left, right, ctx = X
        all_text = pd.concat([pd.Series(left), pd.Series(right)], ignore_index=True)

        self.word_vectorizer_ = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            max_features=self.max_features_word,
            sublinear_tf=True,
            norm="l2",
        )
        self.char_vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_features=self.max_features_char,
            sublinear_tf=True,
            norm="l2",
        )
        self.ctx_vectorizer_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=self.max_features_ctx,
            sublinear_tf=True,
            norm="l2",
        )
        self.ctx_word_vectorizer_ = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=self.max_features_ctx_word,
            sublinear_tf=True,
            norm="l2",
        )

        self.word_vectorizer_.fit(all_text)
        self.char_vectorizer_.fit(all_text)
        self.ctx_vectorizer_.fit(ctx)
        self.ctx_word_vectorizer_.fit(ctx)
        return self

    def transform(self, X):
        left, right, ctx = X

        Lw = self.word_vectorizer_.transform(left)
        Rw = self.word_vectorizer_.transform(right)

        Lc = self.char_vectorizer_.transform(left)
        Rc = self.char_vectorizer_.transform(right)

        X_word = sparse.hstack([Lw, Rw, abs(Lw - Rw), Lw.multiply(Rw)], format="csr")
        X_char = sparse.hstack([Lc, Rc, abs(Lc - Rc), Lc.multiply(Rc)], format="csr")

        X_ctx_char = self.ctx_vectorizer_.transform(ctx)
        X_ctx_word = self.ctx_word_vectorizer_.transform(ctx)

        return sparse.hstack([X_word, X_char, X_ctx_char, X_ctx_word], format="csr")


ridge = Ridge(alpha=0.4, random_state=42, solver="lsqr")

model = Pipeline(
    steps=[
        ("pair_tfidf", PairwiseTfidfFeatures()),
        (
            "ridge_ttr",
            TransformedTargetRegressor(
                regressor=ridge,
                transformer=StandardScaler(with_mean=True, with_std=True),
            ),
        ),
    ]
)

model



## === cell 4
model.fit((X_train_left, X_train_right, X_train_ctx), y_train)
preds = model.predict((X_test_left, X_test_right, X_test_ctx)).astype(np.float64)

preds = np.clip(preds, 0.0, 1.0)

preds = np.round(preds * 4.0) / 4.0
preds = np.clip(preds, 0.0, 1.0)

preds[:10], preds.min(), preds.max()



## === cell 5
submission = pd.DataFrame({"id": test_df["id"].values, "score": preds})

assert (
    submission.shape[0] == sample_sub.shape[0] == test_df.shape[0]
), "Row count mismatch."
assert list(submission.columns) == [
    "id",
    "score",
], "Submission columns must be ['id','score']"

submission.to_csv("submission.csv", index=False)
submission.head()
