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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9862344894988628

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your notebook is an ensemble that expects many precomputed submission files from `../input/...`, but in this environment those directories don’t exist, causing `FileNotFoundError` and cascading `NameError`s. I keep the same “average multiple model submissions” core logic, but make it robust by (1) auto-discovering available submission-like CSVs under the provided `/kaggle/input` and `/kaggle/data` trees, (2) validating/aligning them to the sample submission IDs/columns, and (3) falling back to a safe baseline (sample submission probabilities) if none are found so a valid `submission.csv` is always produced. This is score-neutral when no external predictions exist (baseline), and automatically use any present prediction files without changing the averaging semantics. The final output be written as `submission.csv` with the exact required columns and order.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

for probe in ["../input", "/kaggle/input", "/kaggle/data", "/kaggle/working"]:
    try:
        if os.path.isdir(probe):
            print(probe, "->", len(os.listdir(probe)), "entries")
            print(os.listdir(probe)[:20])
    except Exception as e:
        print("Could not list", probe, "due to", repr(e))



## === cell 1
import numpy as np
import pandas as pd

f_caps_gru = "../input/capsule-net-with-gru/submission.csv"
f_dual_embed_pl = "../input/submission-dual-embed-pl/submission_dual_embed (2).csv"
f_dual_embed_mish = "../input/bi-gru-lstm-dual-embedding-with-mish/submission.csv"
f_lstm_glove_tta = (
    "../input/improved-lstm-baseline-glove-dropout-trainta/submission.csv"
)
f_dual_embed_dehyp = (
    "../input/improved-lstm-baseline-bi-lstm-dual-embed-dehyp/submission.csv"
)
f_lstm_fast = "../input/improved-lstm-baseline-fasttext-dropout/submission.csv"
f_nbsvm = "../input/nb-svm-strong-linear-baseline/submission.csv"
f_bi_post = "../input/bi-post/10fold_lstmpp_am.csv"
f_dpcnn = "../input/dpcnn-wordcloud/10fold_dpcnn_test.csv"
f_dmcnn = "../input/dmcnn-demoji/10fold_dmcnn_am.csv"
f_rcn = "../input/rcn-capsule/10fold_capsule_am.csv"
f_attn_300d = "../input/attn-300d/10fold_attn_post_am.csv"
f_slgbm = "../input/simple-lightgbm-classifier/submission_001.csv"
f_catboost = "../input/catboost/results_preds_cat.csv"

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DATA_CANDIDATES = [
    Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/data/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]
DATA_ROOT = next((p for p in DATA_CANDIDATES if p.exists()), Path("."))

sample_path = None
for p in [
    DATA_ROOT / "sample_submission.csv",
    Path("/kaggle/input/sample_submission.csv"),
    Path("/kaggle/data/sample_submission.csv"),
]:
    if p.exists():
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(sample_path)
sample_sub = sample_sub[["id"] + label_cols]


def _read_pred_csv(path: str):
    """Read a prediction CSV if it exists; return None if missing/unusable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    if "id" not in df.columns:
        return None

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        return None

    df = df[["id"] + label_cols].copy()
    return df


pred_named = {
    "caps_gru": _read_pred_csv(f_caps_gru),
    "dual_embed_pl": _read_pred_csv(f_dual_embed_pl),
    "dual_embed_mish": _read_pred_csv(f_dual_embed_mish),
    "lstm_glove_tta": _read_pred_csv(f_lstm_glove_tta),
    "dual_embed_dehyp": _read_pred_csv(f_dual_embed_dehyp),
    "lstm_fast": _read_pred_csv(f_lstm_fast),
    "nbsvm": _read_pred_csv(f_nbsvm),
    "bi_post": _read_pred_csv(f_bi_post),
    "dpcnn": _read_pred_csv(f_dpcnn),
    "dmcnn": _read_pred_csv(f_dmcnn),
    "rcn": _read_pred_csv(f_rcn),
    "attn_300d": _read_pred_csv(f_attn_300d),
    "slgbm": _read_pred_csv(f_slgbm),
    "catboost": _read_pred_csv(f_catboost),
}


def _discover_prediction_csvs(search_roots, max_files=200):
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for csv_path in root.rglob("*.csv"):
            name = csv_path.name.lower()
            if name in {"train.csv", "test.csv", "sample_submission.csv"}:
                continue
            found.append(csv_path)
            if len(found) >= max_files:
                return found
    return found


auto_csvs = _discover_prediction_csvs(
    search_roots=[
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        Path("."),
        Path("/kaggle/working"),
    ],
    max_files=150,
)

auto_preds = []
for p in auto_csvs:
    df = _read_pred_csv(str(p))
    if df is not None:
        auto_preds.append((str(p), df))

pred_dfs = []
pred_sources = []

for k, df in pred_named.items():
    if df is not None:
        pred_dfs.append(df)
        pred_sources.append(k)

for src, df in auto_preds:
    if src in set(pred_sources):
        continue
    pred_dfs.append(df)
    pred_sources.append(src)

print(f"Loaded {len(pred_dfs)} prediction file(s).")
if len(pred_sources) > 0:
    print("Sources (first 20):")
    for s in pred_sources[:20]:
        print(" -", s)




## === cell 2
def _align_to_sample(df: pd.DataFrame, sample: pd.DataFrame):
    merged = sample[["id"]].merge(df, on="id", how="left", validate="one_to_one")
    miss_rate = merged[label_cols].isna().mean().mean()
    if miss_rate > 0.01:
        return None
    baseline = sample.set_index("id").loc[merged["id"], label_cols].to_numpy()
    arr = merged[label_cols].to_numpy(dtype=np.float64)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        arr[nan_mask] = baseline[nan_mask]
    arr = np.clip(arr, 0.0, 1.0)
    out = pd.DataFrame(arr, columns=label_cols)
    out.insert(0, "id", merged["id"].values)
    return out


aligned_preds = []
aligned_sources = []
for src, df in zip(pred_sources, pred_dfs):
    a = _align_to_sample(df, sample_sub)
    if a is not None:
        aligned_preds.append(a)
        aligned_sources.append(src)

print(f"Aligned {len(aligned_preds)} prediction file(s) to sample submission ids.")




## === cell 3
def _get_by_name(name):
    for src, df in zip(aligned_sources, aligned_preds):
        if src == name:
            return df
    return None


p_caps_gru = _get_by_name("caps_gru")
p_dual_embed_pl = _get_by_name("dual_embed_pl")
p_dual_embed_mish = _get_by_name("dual_embed_mish")
p_lstm_glove_tta = _get_by_name("lstm_glove_tta")
p_dual_embed_dehyp = _get_by_name("dual_embed_dehyp")
p_lstm_fast = _get_by_name("lstm_fast")
p_nbsvm = _get_by_name("nbsvm")
p_bi_post = _get_by_name("bi_post")
p_dpcnn = _get_by_name("dpcnn")
p_dmcnn = _get_by_name("dmcnn")
p_rcn = _get_by_name("rcn")
p_attn_300d = _get_by_name("attn_300d")
p_slgbm = _get_by_name("slgbm")
p_catboost = _get_by_name("catboost")

have_all_original = all(
    x is not None
    for x in [
        p_slgbm,
        p_catboost,
        p_caps_gru,
        p_dual_embed_pl,
        p_dual_embed_mish,
        p_lstm_glove_tta,
        p_dual_embed_dehyp,
        p_lstm_fast,
        p_nbsvm,
        p_bi_post,
        p_dpcnn,
        p_dmcnn,
        p_rcn,
        p_attn_300d,
    ]
)

if have_all_original:
    p_res = p_caps_gru.copy()
    p_res[label_cols] = (
        p_slgbm[label_cols]
        + p_catboost[label_cols]
        + p_caps_gru[label_cols]
        + p_dual_embed_pl[label_cols]
        + p_dual_embed_mish[label_cols]
        + p_lstm_glove_tta[label_cols]
        + p_dual_embed_dehyp[label_cols]
        + p_lstm_fast[label_cols]
        + p_nbsvm[label_cols]
        + p_bi_post[label_cols] * 5
        + p_dpcnn[label_cols] * 5
        + p_dmcnn[label_cols] * 5
        + p_rcn[label_cols] * 5
        + p_attn_300d[label_cols] * 5
    ) / 34.0
    ensemble_mode = "original_weighted_34"
else:
    if len(aligned_preds) == 0:
        p_res = sample_sub.copy()
        ensemble_mode = "baseline_sample_submission"
    else:
        p_res = aligned_preds[0][["id"] + label_cols].copy()
        acc = np.zeros((len(p_res), len(label_cols)), dtype=np.float64)
        for df in aligned_preds:
            acc += df[label_cols].to_numpy(dtype=np.float64)
        acc /= float(len(aligned_preds))
        p_res[label_cols] = np.clip(acc, 0.0, 1.0)
        ensemble_mode = f"mean_of_{len(aligned_preds)}"

print("Ensemble mode:", ensemble_mode)
print(p_res.head())



## === cell 4
p_res = p_res[["id"] + label_cols].copy()
p_res[label_cols] = p_res[label_cols].apply(pd.to_numeric, errors="coerce")
p_res[label_cols] = p_res[label_cols].fillna(0.5).clip(0.0, 1.0)

p_res = sample_sub[["id"]].merge(p_res, on="id", how="left", validate="one_to_one")
p_res[label_cols] = (
    p_res[label_cols].fillna(sample_sub[label_cols].values).clip(0.0, 1.0)
)

assert list(p_res.columns) == ["id"] + label_cols
assert len(p_res) == len(sample_sub)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3757259405.py in <cell line: 0>()
      7 p_res = sample_sub[["id"]].merge(p_res, on="id", how="left", validate="one_to_one")
      8 p_res[label_cols] = (
----> 9     p_res[label_cols].fillna(sample_sub[label_cols].values).clip(0.0, 1.0)
     10 )
     11 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7438                 new_data = self.where(self.notna(), value)._mgr
   7439             else:
-> 7440                 raise ValueError(f"invalid fill value with a {type(value)}")
   7441 
   7442         result = self._constructor_from_mgr(new_data, axes=new_data.axes)

ValueError: invalid fill value with a <class 'numpy.ndarray'>

## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", p_res.shape)
print(p_res.head())
