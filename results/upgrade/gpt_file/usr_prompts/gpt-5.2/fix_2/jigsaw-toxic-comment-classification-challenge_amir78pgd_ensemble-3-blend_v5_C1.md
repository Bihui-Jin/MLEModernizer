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

0.9863068319863196

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The failure is because the notebook tries to ensemble submissions from other Kaggle datasets that are not present in your `/kaggle/input` tree, so the first `read_csv` raises `FileNotFoundError` and all later variables are undefined. I keep the ensemble core logic (weighted averaging) but make it robust: dynamically discover which candidate submission files actually exist, load only those, and compute a properly normalized weighted mean. To ensure a valid end-to-end run and a correct `.csv` output, I also align/merge predictions by `id` against `sample_submission.csv`, enforce the required column order, clip probabilities to `[0,1]`, and always write `submission.csv`. If none of the external submissions exist, the code fall back to using `sample_submission.csv` (0.5s) so it still produces a valid submission file.'

# 9. Code solution

## === cell 0
import os

print("Listing /kaggle/input (top-level):")
try:
    print(os.listdir("/kaggle/input")[:50])
except FileNotFoundError:
    print("No /kaggle/input found; will rely on /kaggle/data paths.")



## === cell 1
import numpy as np
import pandas as pd
from pathlib import Path

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

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DATA_ROOTS = [
    Path("/kaggle/data/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/data"),
    Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/input"),
]


def first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


sample_path = first_existing(*[r / "sample_submission.csv" for r in DATA_ROOTS])
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known Kaggle roots."
    )

sample_sub = pd.read_csv(sample_path)
missing = [c for c in (["id"] + label_cols) if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

print("Using sample_submission:", str(sample_path))
print("sample_submission shape:", sample_sub.shape)



## === cell 2


def resolve_path(p: str) -> Path:
    """
    Resolve a Kaggle-style '../input/...' path to a real path if possible.
    Also try mapping '../input' -> '/kaggle/input' and similar.
    """
    p0 = Path(p)

    if p0.exists():
        return p0

    s = str(p0)
    if s.startswith("../input/"):
        p1 = Path("/kaggle/input") / s[len("../input/") :]
        if p1.exists():
            return p1

    if s.startswith("../input/"):
        p2 = Path("/kaggle/data") / s[len("../input/") :]
        if p2.exists():
            return p2

    return p0  # return original (will be checked by exists())


def load_submission(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{name}: missing 'id' column in {path}")
    missing_cols = [c for c in label_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing label columns {missing_cols} in {path}")
    df = df[["id"] + label_cols].copy()
    return df


candidates = [
    ("caps_gru", f_caps_gru, 1.0),
    ("dual_embed_pl", f_dual_embed_pl, 1.0),
    ("dual_embed_mish", f_dual_embed_mish, 1.0),
    ("lstm_glove_tta", f_lstm_glove_tta, 1.0),
    ("dual_embed_dehyp", f_dual_embed_dehyp, 1.0),
    ("lstm_fast", f_lstm_fast, 1.0),
    ("nbsvm", f_nbsvm, 1.0),
    ("bi_post", f_bi_post, 4.0),
    ("dpcnn", f_dpcnn, 4.0),
    ("dmcnn", f_dmcnn, 4.0),
    ("rcn", f_rcn, 4.0),
]

loaded = []
for name, path_str, w in candidates:
    p = resolve_path(path_str)
    if p.exists():
        try:
            df = load_submission(p, name)
            loaded.append((name, df, float(w)))
            print(f"Loaded {name} from {p} (weight={w}) shape={df.shape}")
        except Exception as e:
            print(f"Skipping {name} from {p}: {type(e).__name__}: {e}")
    else:
        print(f"Missing {name}: {p}")

print(f"Total loaded external submissions: {len(loaded)}")



## === cell 3

p_res = sample_sub[["id"] + label_cols].copy()

if len(loaded) == 0:
    print(
        "No external submissions found; using sample_submission probabilities as fallback."
    )
else:
    acc = np.zeros((len(p_res), len(label_cols)), dtype=np.float64)
    wsum = 0.0

    base_ids = p_res["id"].astype(str)
    for name, df, w in loaded:
        df_ids = df["id"].astype(str)
        aligned = pd.DataFrame({"id": base_ids}).merge(
            df.assign(id=df_ids), on="id", how="left", validate="one_to_one"
        )

        vals = aligned[label_cols].astype(np.float64)
        if vals.isna().any().any():
            n_miss = int(vals.isna().any(axis=1).sum())
            print(
                f"Warning: {name} missing {n_miss} ids after alignment; filling with 0.5"
            )
            vals = vals.fillna(0.5)

        acc += w * vals.to_numpy()
        wsum += w

    p_res[label_cols] = acc / wsum

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)
p_res = p_res[["id"] + label_cols]

print("Result head:")
print(p_res.head())
print("Result shape:", p_res.shape)



## === cell 4
expected_cols = list(sample_sub.columns)
if list(p_res.columns) != expected_cols:
    for c in expected_cols:
        if c not in p_res.columns:
            p_res[c] = 0.5
    p_res = p_res[expected_cols]

assert p_res["id"].notna().all()
assert p_res.shape[0] == sample_sub.shape[0]
for c in label_cols:
    assert ((p_res[c] >= 0.0) & (p_res[c] <= 1.0)).all()



## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {p_res.shape} and columns {list(p_res.columns)}")
print(p_res.head())
