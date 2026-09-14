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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.9112965725853808

# 6. Current score

0.57494

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58571) has done: 'The notebook fails because it tries to read many out-of-environment “../input/model-*” blended submission files that don’t exist here, so nothing downstream is defined and no CSV is written. I keep the same blending-style “produce a submission from existing predictions” core idea, but add a safe fallback: if those external blend files are missing, build a valid submission directly from the provided `sample_submission.csv` (neutral baseline) so a `.csv` is always produced end-to-end. I also make the code robust to different Kaggle directory layouts by probing the known dataset locations you listed. This is score-neutral/low (since current score is not yielded) but guarantees a valid submission file is created.'
- What this solution (achieved 0.58571) has done: 'Your current code falls back to an all-zero submission whenever any blended file is missing, which likely explains the low 0.58571 score; the smallest improvement is to blend whatever submission files are actually present instead of requiring all of them. I keep the same “average-blend existing submissions” core logic, but change the gating to use all available files (with equal weights) and only fall back to zeros if none exist. I also make the merge robust to partial coverage by filling missing predictions with the mean label from the available submissions (instead of 0.0), which typically improves AUC without changing the overall approach. The output path and required `Id,Label` format remain unchanged, and the script still run end-to-end and always write a valid `.csv`.'
- What this solution (achieved 0.58571) has done: 'Your score is low mainly because cell 3 accidentally overwrites `final_sub` and then merges from `merged` (which is not the blended output) and fills missing with 0.0—this effectively collapses predictions toward zeros and harms AUC. I make a minimal fix: keep the blended `final_sub` produced in cell 2 and only align it to `sample_sub`’s Id order (no re-merge from the wide `merged` table). I also change the “no blend files available” fallback from all-zeros to a constant equal to the sample’s mean label (usually 0.5), which is a safe, small AUC lift over zeros without changing the core “blend submissions” approach. The script still run end-to-end and always write a valid `Id,Label` CSV.'
- What this solution (achieved 0.57494) has done: 'Your current blend code is correct structurally, but in this environment none of the referenced `../input/model-*` blend files exist, so you always fall back to a constant prediction (essentially random ranking), which explains the low AUC. To move the score toward the 0.911 target without changing the “blend existing submissions” core idea, I add a minimal, legitimate fallback that generates non-constant, image-specific scores directly from the provided JPEG bytes (a simple steganalysis prior) when no blend files are found. This keeps the rest of your pipeline intact: it still uses `sample_submission.csv` as the Id source, preserves the final alignment/merge, and writes the same submission filename. The heuristic uses per-image compressed-size and a couple of byte-entropy/compressibility proxies to create a ranking signal, which should substantially improve AUC versus a constant baseline while staying lightweight and within the time limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zlib
from collections import Counter




## === cell 1
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _read_submission_or_none(path):
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if "Id" not in df.columns:
        for c in ["image_id", "ImageId", "id"]:
            if c in df.columns:
                df = df.rename(columns={c: "Id"})
                break
    if "Label" not in df.columns:
        for c in ["label", "pred", "prediction", "prob"]:
            if c in df.columns:
                df = df.rename(columns={c: "Label"})
                break
    if not {"Id", "Label"}.issubset(df.columns):
        raise ValueError(
            f"Submission file {path} does not contain required columns Id, Label. Found: {df.columns.tolist()}"
        )
    return df[["Id", "Label"]].copy()


blend_paths = {
    "sub_10": "../input/model-1-fold0/submission_fold0_epoch35.csv",
    "sub_20": "../input/model-2-fold0/submission_fold0_epoch38.csv",
    "sub_30": "../input/model-3-fold0/submission_fold0_epoch39.csv",
    "sub_11": "../input/model-1-fold1/submission_fold1_epoch_30.csv",
    "sub_21": "../input/model-2-fold1/submission_fold1_epoch_34.csv",
    "sub_31": "../input/model-3-fold1/submission_fold1_epoch_38.csv",
    "sub_12": "../input/model-1-fold2/submission_fold2_epoch_34.csv",
    "sub_22": "../input/model-2-fold2/submission_fold2_epoch_37.csv",
    "sub_32": "../input/model-3-fold2/submission_fold2_epoch_39.csv",
    "sub_13": "../input/model-1-fold3/submission_fold3_epoch_35.csv",
    "sub_23": "../input/model-2-fold3/submission_fold3_epoch_36.csv",
    "sub_33": "../input/model-3-fold3/submission_fold3_epoch_39.csv",
    "sub_14": "../input/model-1-fold4/submission_fold4_epoch_33.csv",
    "sub_24": "../input/model-2-fold4/submission_fold4_epoch_35.csv",
    "sub_34": "../input/model-3-fold4/submission_fold4_epoch_36.csv",
    "sub_b3_13": "../input/b3-fold3-m1/submission_fold3_b3_m1.csv",
    "sub_b3_23": "../input/b3-fold3-m2/submission_fold3_b3_m2.csv",
    "sub_b3_33": "../input/b3-fold3-m3/submission_fold3_b3_m3.csv",
    "sub_b3_10": "../input/b3-fold0-m1/submission_fold0_b3_m1.csv",
    "sub_b3_20": "../input/b3-fold0-m2/submission_fold0_b3_m2.csv",
    "sub_b3_30": "../input/b3-fold0-m3/submission_fold0_b3_m3.csv",
    "sub_oc_1": "../input/oc-m1/submission_openclose_m1.csv",
    "sub_oc_2": "../input/oc-m2/submission_openclose_m2.csv",
    "sub_oc_3": "../input/oc-m3/submission_openclose_m3.csv",
}

subs = {}
missing = []
for k, p in blend_paths.items():
    df = _read_submission_or_none(p)
    if df is None:
        missing.append(p)
    subs[k] = df

sample_path = _first_existing(
    [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/alaska2-image-steganalysis/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/alaska2-image-steganalysis/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/alaska2-image-steganalysis/sample_submission.csv",
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
if not {"Id", "Label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )

print(
    f"Found {len(subs) - len(missing)}/{len(subs)} blend files. Missing: {len(missing)}"
)




## === cell 2
def _locate_test_dir():
    return _first_existing(
        [
            "/kaggle/input/alaska2-image-steganalysis/Test",
            "/kaggle/input/Test",
            "/kaggle/data/alaska2-image-steganalysis/Test",
            "/kaggle/data/Test",
            "../input/alaska2-image-steganalysis/Test",
            "../input/Test",
        ]
    )


def _jpeg_byte_features(file_path, max_bytes=120_000):
    with open(file_path, "rb") as f:
        b = f.read(max_bytes)
    n = len(b)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)

    cnt = Counter(b)
    p = np.fromiter((v / n for v in cnt.values()), dtype=np.float64)
    ent = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    z = zlib.compress(b, level=6)
    zratio = float(len(z) / n)

    ff_ratio = float(cnt.get(0xFF, 0) / n)
    zero_ratio = float(cnt.get(0x00, 0) / n)

    return (float(n), ent, zratio, ff_ratio + zero_ratio)


def _make_heuristic_submission(sample_sub_df):
    test_dir = _locate_test_dir()
    if test_dir is None:
        ids = sample_sub_df["Id"].astype(str).values
        h = np.array([hash(x) % 10_000 for x in ids], dtype=np.float64)
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        out = sample_sub_df[["Id"]].copy()
        out["Label"] = h
        return out

    ids = sample_sub_df["Id"].astype(str).values
    feats = np.zeros((len(ids), 4), dtype=np.float64)

    for i, img_id in enumerate(ids):
        fp = os.path.join(test_dir, img_id)
        if not os.path.exists(fp):
            if not img_id.lower().endswith(".jpg") and os.path.exists(fp + ".jpg"):
                fp = fp + ".jpg"
            else:
                feats[i, :] = 0.0
                continue
        feats[i, :] = np.array(_jpeg_byte_features(fp), dtype=np.float64)

    def _norm01(x):
        lo = np.nanpercentile(x, 1)
        hi = np.nanpercentile(x, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    size_n = _norm01(feats[:, 0])
    ent_n = _norm01(feats[:, 1])
    zratio_n = _norm01(feats[:, 2])
    marker_n = _norm01(feats[:, 3])

    score = 0.15 * size_n + 0.55 * ent_n + 0.25 * zratio_n + 0.05 * marker_n

    out = sample_sub_df[["Id"]].copy()
    out["Label"] = score.astype(float)
    return out


available_items = [(k, df) for k, df in subs.items() if df is not None]

if len(available_items) > 0:
    cleaned = []
    for k, df in available_items:
        d = df.copy()
        d["Id"] = d["Id"].astype(str)
        d["Label"] = pd.to_numeric(d["Label"], errors="coerce")
        cleaned.append((k, d.sort_values("Id").reset_index(drop=True)))

    merged = sample_sub[["Id"]].copy()
    merged["Id"] = merged["Id"].astype(str)

    for k, d in cleaned:
        merged = merged.merge(
            d.rename(columns={"Label": k}),
            on="Id",
            how="left",
            validate="one_to_one",
        )

    pred_cols = [k for k, _ in cleaned]

    row_mean = merged[pred_cols].mean(axis=1, skipna=True)
    global_mean = (
        float(row_mean.mean(skipna=True))
        if np.isfinite(row_mean.mean(skipna=True))
        else 0.0
    )

    for c in pred_cols:
        merged[c] = merged[c].fillna(row_mean).fillna(global_mean)

    merged["Label"] = merged[pred_cols].mean(axis=1).astype(float)

    final_sub = merged[["Id", "Label"]].copy()
else:
    final_sub = _make_heuristic_submission(sample_sub)

final_sub["Id"] = final_sub["Id"].astype(str)
final_sub["Label"] = pd.to_numeric(final_sub["Label"], errors="coerce").astype(float)

final_sub = (
    sample_sub[["Id"]]
    .copy()
    .assign(Id=sample_sub["Id"].astype(str))
    .merge(
        final_sub[["Id", "Label"]],
        on="Id",
        how="left",
        validate="one_to_one",
    )
)

fill_value = float(pd.to_numeric(final_sub["Label"], errors="coerce").mean())
if not np.isfinite(fill_value):
    fill_value = 0.5
final_sub["Label"] = (
    pd.to_numeric(final_sub["Label"], errors="coerce").fillna(fill_value).astype(float)
)

out_path = "submission_Jul20_blend_b3_b2_oc.csv"
final_sub.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(final_sub.head())
print(f"Rows: {len(final_sub)}, Columns: {final_sub.columns.tolist()}")
print(
    f"Label stats: min={final_sub['Label'].min():.6f}, max={final_sub['Label'].max():.6f}, mean={final_sub['Label'].mean():.6f}"
)
