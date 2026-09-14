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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
datasets==4.4.1
geopandas==0.14.4
imbalanced-learn==0.13.0
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8155382193257055

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.02203) has done: 'The timeout is almost certainly dominated by the two vectorizers being fit on 139k essays plus the 5-fold LightGBM training fallback; the engineered-feature Polars pipeline is also heavy but already cacheable. I make the runtime deterministic and fast by (1) avoiding the extremely slow identity-tokenizer path in scikit-learn by letting vectorizers use their native fast tokenization (equivalent semantics for word n-grams on raw text), (2) caching *all* expensive artifacts (engineered features, sparse matrices, and models) keyed by row counts + a stable config signature, and (3) forcing the solution to use the provided pretrained model file when present (and otherwise training once and caching). These changes preserve the algorithm’s core logic (same features: engineered + TFIDF + Count, same LightGBM objective/metric, same inference aggregation), but remove redundant work that causes the 10-minute timeout.'
- What this solution (achieved -0.02203) has done: 'I first remove the import-time crash that prevents the notebook from running (it comes from `datasets/transformers` pulling in an incompatible protobuf stack in this environment, and those imports aren’t used by the solution). Next, I fix the LightGBM training configuration bug where `class_weight` is incorrectly passed to `LGBMRegressor` (this can break training and/or lead to very poor models), which is likely a major cause of the bad score. Finally, I keep the feature/model logic identical but make the submission alignment more robust by ensuring we predict for every `essay_id` in `sample_submission.csv` and preserve its order.'
- What this solution (achieved 0.0) has done: 'Your very low score is most consistent with a label/prediction scale mismatch inside the custom QWK objective/metric: `y_true` is provided to LightGBM exactly as you pass it, but your code adds `a` to `y_true` (and also shifts `y` by `-a`), effectively misaligning training and evaluation. I keep your exact feature pipeline, model type, and training loop, but fix the objective/metric to operate directly on the true 1–6 labels (removing the `a`-shift) and adjust the prediction post-processing accordingly. This is a minimal semantic correction (not a new approach) and should move QWK sharply upward toward your target. I also make the model-loading path safe (keep it if present) but verify it’s compatible; otherwise train and cache as you already do.'
- What this solution (achieved -0.00246) has done: 'Your 0.0 score is most consistent with the custom LightGBM objective being mathematically mis-specified for this task (it uses a constant-anchor term but never stabilizes the denominator), which can lead to degenerate training and effectively random predictions after rounding. I keep your exact feature pipeline (engineered + TFIDF + Count), model type, folds, and inference averaging, but replace the objective with a stable differentiable surrogate that matches your intent: optimize squared error on the continuous 1–6 target (a minimal semantic correction that aligns with your own QWK eval/rounding). I also remove early stopping (it conflicts with the “no relaxed convergence” requirement) while keeping the same `n_estimators`, and I add a small safety check to ensure predictions are finite before rounding (prevents silent NaNs that can tank QWK). These are minimal changes intended to move the score sharply upward toward your target band without changing the overall approach.'
- What this solution (achieved -0.00246) has done: 'Your score is far below the target, so the safest way to move QWK upward (without changing your feature set, model type, folds, or training loop) is to fix a schema mismatch that silently breaks inference: your engineered feature columns for train/test are not guaranteed to be identical or in the same order, which can produce essentially random predictions and tank QWK. I enforce a single canonical engineered-feature column list from train, align/reindex test features to that list (filling missing with 0), and use that exact order for both train and test dense blocks before the sparse hstack. I also make the essay_id ordering consistent by sorting engineered feature frames by `essay_id` before building matrices (prediction output remains aligned to `sample_submission.csv` as you already do). These are minimal changes that preserve core modeling semantics but remove a common cause of near-zero/negative kappa.'

# 9. Code solution

## === cell 0
import os
import gc
import re
import copy
import random
import pickle
import warnings
import hashlib

import numpy as np
import pandas as pd
import polars as pl

import torch  # kept to preserve original imports/compat
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)

import nltk  # noqa: F401

from scipy import sparse

warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

gc.set_threshold(700, 10, 10)




## === cell 1
def dataPreprocessing_expr(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
    )


columns = [
    (pl.col("full_text").str.split(by="\n\n").alias("paragraph")),
]
PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

train.head(1)



## === cell 2
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I would have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}

c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x




## === cell 3
paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
sentence_fea = ["sentence_len", "sentence_word_cnt"]


def build_engineered_features(df: pl.DataFrame) -> pd.DataFrame:
    base = df.select(["essay_id", "full_text", "paragraph"])

    base_lf = base.lazy().with_columns(
        dataPreprocessing_expr(pl.col("full_text")).alias("full_text_clean")
    )

    p = (
        base_lf.explode("paragraph")
        .with_columns(dataPreprocessing_expr(pl.col("paragraph")).alias("paragraph"))
        .with_columns(
            pl.col("paragraph").str.len_chars().alias("paragraph_len"),
            (pl.col("paragraph").str.count_matches(r"\.") + 1).alias(
                "paragraph_sentence_cnt"
            ),
            (pl.col("paragraph").str.count_matches(r" ") + 1).alias(
                "paragraph_word_cnt"
            ),
        )
    )
    p_aggs = [
        *[
            (pl.col("paragraph_len") >= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [
                50,
                75,
                100,
                125,
                150,
                175,
                200,
                250,
                300,
                350,
                400,
                500,
                600,
                700,
            ]
        ],
        *[
            (pl.col("paragraph_len") <= i).sum().alias(f"paragraph_{i}_cnt")
            for i in [25, 49]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
    ]
    p_df = p.group_by(["essay_id"], maintain_order=True).agg(p_aggs)

    s = (
        base_lf.with_columns(
            pl.col("full_text_clean").str.split(by=".").alias("sentence")
        )
        .explode("sentence")
        .with_columns(pl.col("sentence").str.len_chars().alias("sentence_len"))
        .filter(pl.col("sentence_len") >= 15)
        .with_columns(
            (pl.col("sentence").str.count_matches(r" ") + 1).alias("sentence_word_cnt")
        )
    )
    s_aggs = [
        *[
            (pl.col("sentence_len") >= i).sum().alias(f"sentence_{i}_cnt")
            for i in [15, 50, 100, 150, 200, 250, 300]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
    ]
    s_df = s.group_by(["essay_id"], maintain_order=True).agg(s_aggs)

    w = (
        base_lf.with_columns(pl.col("full_text_clean").str.split(by=" ").alias("word"))
        .explode("word")
        .with_columns(pl.col("word").str.len_chars().alias("word_len"))
        .filter(pl.col("word_len") != 0)
    )
    w_aggs = [
        *[
            (pl.col("word_len") >= (i + 1)).sum().alias(f"word_{i+1}_cnt")
            for i in range(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.50).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    w_df = w.group_by(["essay_id"], maintain_order=True).agg(w_aggs)

    feats = (
        p_df.join(s_df, on="essay_id", how="left")
        .join(w_df, on="essay_id", how="left")
        .sort("essay_id")
        .collect(streaming=True)
        .to_pandas()
    )
    return feats


def _feature_cache_path(name: str, df: pl.DataFrame) -> str:
    return f"/kaggle/working/{name}_feats_{df.height}.pkl"


train_cache = _feature_cache_path("train", train)
if os.path.exists(train_cache):
    train_feats = pd.read_pickle(train_cache)
else:
    train_feats = build_engineered_features(train)
    train_feats["score"] = train["score"].to_numpy()
    train_feats.to_pickle(train_cache)

train_feats = train_feats.sort_values("essay_id").reset_index(drop=True)

engineered_feature_names = [
    c for c in train_feats.columns if c not in ["essay_id", "score"]
]

print("Features Number (engineered): ", len(engineered_feature_names))
train_feats.head(3)



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass




## === cell 7
def _vec_config_sig(vec) -> str:
    cfg = vec.get_params(deep=False)
    for k in list(cfg.keys()):
        if callable(cfg[k]):
            cfg[k] = str(cfg[k])
    blob = repr(sorted(cfg.items())).encode("utf-8")
    return hashlib.md5(blob).hexdigest()  # stable, short


def _vec_cache_paths(name: str, n_rows: int, sig: str) -> dict:
    base = f"/kaggle/working/{name}_{n_rows}_{sig}"
    return {
        "tfidf_vec": base + "_tfidf_vectorizer.pkl",
        "cnt_vec": base + "_cnt_vectorizer.pkl",
        "tfidf_mat": base + "_tfidf.npz",
        "cnt_mat": base + "_cnt.npz",
    }


train_text = train.get_column("full_text").to_numpy()

vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(4, 8),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

tfidf_sig = _vec_config_sig(vectorizer)
train_paths = _vec_cache_paths("train", train.height, tfidf_sig)

if os.path.exists(train_paths["tfidf_vec"]) and os.path.exists(
    train_paths["tfidf_mat"]
):
    with open(train_paths["tfidf_vec"], "rb") as f:
        vectorizer = pickle.load(f)
    train_tfid = sparse.load_npz(train_paths["tfidf_mat"])
    if not sparse.isspmatrix_csr(train_tfid):
        train_tfid = train_tfid.tocsr()
else:
    train_tfid = vectorizer.fit_transform(train_text)
    if not sparse.isspmatrix_csr(train_tfid):
        train_tfid = train_tfid.tocsr()
    with open(train_paths["tfidf_vec"], "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    sparse.save_npz(train_paths["tfidf_mat"], train_tfid)

n_tfid = train_tfid.shape[1]
print("TFIDF features:", n_tfid)

del train_text
gc.collect()



## === cell 8
vectorizer_cnt = CountVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(3, 5),
    min_df=0.10,
    max_df=0.85,
)

cnt_sig = _vec_config_sig(vectorizer_cnt)
train_paths_cnt = _vec_cache_paths("train", train.height, cnt_sig)

train_text = train.get_column("full_text").to_numpy()

if os.path.exists(train_paths_cnt["cnt_vec"]) and os.path.exists(
    train_paths_cnt["cnt_mat"]
):
    with open(train_paths_cnt["cnt_vec"], "rb") as f:
        vectorizer_cnt = pickle.load(f)
    train_cnt = sparse.load_npz(train_paths_cnt["cnt_mat"])
    if not sparse.isspmatrix_csr(train_cnt):
        train_cnt = train_cnt.tocsr()
else:
    train_cnt = vectorizer_cnt.fit_transform(train_text)
    if not sparse.isspmatrix_csr(train_cnt):
        train_cnt = train_cnt.tocsr()
    with open(train_paths_cnt["cnt_vec"], "wb") as f:
        pickle.dump(vectorizer_cnt, f, protocol=pickle.HIGHEST_PROTOCOL)
    sparse.save_npz(train_paths_cnt["cnt_mat"], train_cnt)

n_cnt = train_cnt.shape[1]
print("Count features:", n_cnt)

del train_text
gc.collect()




## === cell 9
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(preds, train_data):
    y_true = train_data.get_label().astype(np.float64, copy=False)
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.clip(preds, 1.0, 6.0)
    grad = preds - y_true
    hess = np.ones_like(grad)
    return grad, hess





## === cell 10
X_dense = train_feats[engineered_feature_names].to_numpy(dtype=np.float32, copy=False)

X = sparse.hstack(
    [sparse.csr_matrix(X_dense), train_tfid, train_cnt],
    format="csr",
    dtype=np.float32,
)

y_split = train_feats["score"].astype(int).to_numpy()
y = train_feats["score"].to_numpy(dtype=np.float32, copy=False)
oof = train_feats["score"].to_numpy(dtype=np.float32, copy=True)

print(
    "Final X shape:",
    X.shape,
    " (engineered:",
    X_dense.shape[1],
    "tfidf:",
    n_tfid,
    "cnt:",
    n_cnt,
    ")",
)

del X_dense
gc.collect()



## === cell 11
pretrained_model_path = "/kaggle/input/llgbm-models/lgbm_models.pkl"
trained_cache_path = "/kaggle/working/lgbm_models.pkl"

models = None
if os.path.exists(pretrained_model_path):
    with open(pretrained_model_path, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} models from: {pretrained_model_path}")
elif os.path.exists(trained_cache_path):
    with open(trained_cache_path, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} models from: {trained_cache_path}")
else:
    print(
        f"[WARN] Pretrained model file not found at {pretrained_model_path} and no cache at {trained_cache_path}. "
        "Falling back to training in-notebook."
    )

if models is not None:
    ok = isinstance(models, (list, tuple)) and len(models) > 0
    ok = ok and all(hasattr(m, "predict") for m in models)
    if not ok:
        models = None

if models is None:
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    f1_scores = []
    kappa_scores = []
    models = []

    callbacks = [
        log_evaluation(period=50),
    ]

    split_dummy = np.zeros(len(y_split), dtype=np.uint8)

    i = 1
    for train_index, test_index in skf.split(split_dummy, y_split):
        print("fold", i)
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = (
            y[train_index],
            y[test_index],
            y_split[test_index],
        )

        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=0.05,
            max_depth=5,
            num_leaves=10,
            colsample_bytree=0.3,
            reg_alpha=0.7,
            reg_lambda=0.1,
            n_estimators=700,
            random_state=42,
            extra_trees=True,
            verbosity=-1,
            n_jobs=-1,
        )

        predictor = model.fit(
            X_train_fold,
            y_train_fold,
            eval_names=["valid"],
            eval_set=[(X_test_fold, y_test_fold)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks,
        )
        models.append(predictor)

        predictions_fold = predictor.predict(X_test_fold)
        oof[test_index] = predictions_fold

        predictions_fold_int = predictions_fold.clip(1, 6).round()
        f1_fold = f1_score(y_test_fold_int, predictions_fold_int, average="weighted")
        kappa_fold = cohen_kappa_score(
            y_test_fold_int, predictions_fold_int, weights="quadratic"
        )
        f1_scores.append(f1_fold)
        kappa_scores.append(kappa_fold)

        print(f"F1 score across fold: {f1_fold}")
        print(f"Cohen kappa score across fold: {kappa_fold}")
        i += 1

    print("=" * 50)
    print(f"Mean F1 score across {n_splits} folds: {np.mean(f1_scores)}")
    print(f"Mean Cohen kappa score across {n_splits} folds: {np.mean(kappa_scores)}")
    print("=" * 50)

    with open(trained_cache_path, "wb") as fp:
        pickle.dump(models, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved trained models to {trained_cache_path}")

if not isinstance(models, (list, tuple)) or len(models) == 0:
    raise RuntimeError("No models available for inference after load/train fallback.")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/479690495.py in <cell line: 0>()
     63         )
     64 
---> 65         predictor = model.fit(
     66             X_train_fold,
     67             y_train_fold,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    320             )
    321 
--> 322         booster.update(fobj=fobj)
    323 
    324         evaluation_result_list: List[_LGBM_BoosterEvalMethodResultType] = []

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in update(self, train_set, fobj)
   4163             if not self.__set_objective_to_none:
   4164                 self.reset_parameter({"objective": "none"}).__set_objective_to_none = True
-> 4165             grad, hess = fobj(self.__inner_predict(0), self.train_set)
   4166             return self.__boost(grad, hess)
   4167 

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in __call__(self, preds, dataset)
    221         argc = len(signature(self.func).parameters)
    222         if argc == 2:
--> 223             grad, hess = self.func(labels, preds)  # type: ignore[call-arg]
    224             return grad, hess
    225 

/tmp/ipykernel_11/3348781954.py in qwk_obj(preds, train_data)
     12 # producing near-random predictions and very low/negative QWK.
     13 def qwk_obj(preds, train_data):
---> 14     y_true = train_data.get_label().astype(np.float64, copy=False)
     15     preds = np.asarray(preds, dtype=np.float64)
     16     preds = np.clip(preds, 1.0, 6.0)

AttributeError: 'numpy.ndarray' object has no attribute 'get_label'

## === cell 12
test_cache = _feature_cache_path("test", test)
if os.path.exists(test_cache):
    test_feats = pd.read_pickle(test_cache)
else:
    test_feats = build_engineered_features(test)
    test_feats.to_pickle(test_cache)

test_feats = test_feats.sort_values("essay_id").reset_index(drop=True)

for c in engineered_feature_names:
    if c not in test_feats.columns:
        test_feats[c] = 0.0
test_feats_aligned = test_feats[["essay_id"] + engineered_feature_names].copy()

test_text = test.get_column("full_text").to_numpy()

test_paths_tfid = _vec_cache_paths("test", test.height, tfidf_sig)
if os.path.exists(test_paths_tfid["tfidf_mat"]):
    test_tfid = sparse.load_npz(test_paths_tfid["tfidf_mat"])
    if not sparse.isspmatrix_csr(test_tfid):
        test_tfid = test_tfid.tocsr()
else:
    test_tfid = vectorizer.transform(test_text)
    if not sparse.isspmatrix_csr(test_tfid):
        test_tfid = test_tfid.tocsr()
    sparse.save_npz(test_paths_tfid["tfidf_mat"], test_tfid)

test_paths_cnt = _vec_cache_paths("test", test.height, cnt_sig)
if os.path.exists(test_paths_cnt["cnt_mat"]):
    test_cnt = sparse.load_npz(test_paths_cnt["cnt_mat"])
    if not sparse.isspmatrix_csr(test_cnt):
        test_cnt = test_cnt.tocsr()
else:
    test_cnt = vectorizer_cnt.transform(test_text)
    if not sparse.isspmatrix_csr(test_cnt):
        test_cnt = test_cnt.tocsr()
    sparse.save_npz(test_paths_cnt["cnt_mat"], test_cnt)

del test_text
gc.collect()

X_test_dense = test_feats_aligned[engineered_feature_names].to_numpy(
    dtype=np.float32, copy=False
)
X_test = sparse.hstack(
    [sparse.csr_matrix(X_test_dense), test_tfid, test_cnt],
    format="csr",
    dtype=np.float32,
)

print("Features number (engineered): ", len(engineered_feature_names))
print("Final X_test shape:", X_test.shape)

del X_test_dense
gc.collect()

pred_sum = None
for model in models:
    pred = model.predict(X_test)
    if pred_sum is None:
        pred_sum = pred.astype(np.float64, copy=False)
    else:
        pred_sum += pred

predictions = pred_sum / len(models)

if not np.isfinite(predictions).all():
    fill_val = float(np.clip(train_feats["score"].mean(), 1, 6))
    predictions = np.where(np.isfinite(predictions), predictions, fill_val)

predictions = np.round(np.clip(predictions, 1, 6)).astype(int)

print(predictions[:20], " ... total:", len(predictions))

submission = pd.read_csv(PATH + "sample_submission.csv")
pred_df = pd.DataFrame(
    {"essay_id": test_feats_aligned["essay_id"].values, "score": predictions}
)

submission = submission.drop(columns=["score"]).merge(
    pred_df, on="essay_id", how="left", sort=False
)

if submission["score"].isna().any():
    fill_val = int(np.round(np.clip(train_feats["score"].mean(), 1, 6)))
    submission["score"] = submission["score"].fillna(fill_val).astype(int)
else:
    submission["score"] = submission["score"].astype(int)

submission.to_csv("submission.csv", index=None)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4040601614.py in <cell line: 0>()
     63         pred_sum += pred
     64 
---> 65 predictions = pred_sum / len(models)
     66 
     67 if not np.isfinite(predictions).all():

TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
