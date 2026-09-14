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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.4517647058823529

# 6. Current score

0.0412

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02342) has done: 'The main blocker is that the notebook depends on external Kaggle input modules (`dicom_reader.py`, `preprocess.py`, `nextvit.py`, and several pip-installed wheels) that are not available in your provided environment, so imports fail and `test_df` is never created. To make this run end-to-end without changing the overall “read test.csv → predict per image → group by prediction_id → write submission.csv” semantics, I replace the missing DICOM/PNG/model pipeline with a lightweight metadata-only probabilistic baseline and keep the submission formatting identical to the competition requirement. I also fix the incorrect cell numbering (your script starts at cell 0) and ensure all required imports are present so `pd`, `Dataset`, etc. are defined. Finally, the code always writes `submission.csv` with columns `prediction_id,cancer` and the correct number of rows matching `sample_submission.csv`.'
- What this solution (achieved 0.0412) has done: 'Your current metadata-only baseline is likely under-scoring because it mixes per-image “proxy risks” in a way that doesn’t match the metric’s need for sharp separation between positives and negatives. To move your pF1 upward toward the 0.45 target without changing the overall pipeline (read CSVs → compute per-image probabilities → aggregate by prediction_id → write submission), I (1) compute smoothed target rates at the *prediction_id* level for train and apply them to test via site/machine/view/laterality, (2) calibrate the final probabilities with a single temperature-like power transform chosen by GroupKFold cross-validation to directly maximize pF1 on out-of-fold predictions, and (3) keep the same submission formatting/alignment with `sample_submission.csv`. These are minimal, metric-aligned changes that keep the “metadata baseline” core logic intact while making probabilities better calibrated for pF1.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_CSV = f"{DATA_DIR}/sample_submission.csv"

np.random.seed(42)

print("Using data dir:", DATA_DIR)
print(
    "Files exist:",
    os.path.exists(TRAIN_CSV),
    os.path.exists(TEST_CSV),
    os.path.exists(SAMPLE_SUB_CSV),
)



## === cell 1
mode = ["submit"]  # keep same default behavior
test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print("train_df", train_df.shape)
print("test_df", test_df.shape)
print("sample_sub", sample_sub.shape)
print("test_df columns:", list(test_df.columns))
print("")




## === cell 2
def make_debug_submission(df, out_path="submission.csv", value=0.0):
    submit_df = pd.DataFrame(
        {"prediction_id": df["prediction_id"].values, "cancer": float(value)}
    )
    submit_df = (
        submit_df.groupby("prediction_id", sort=True)["cancer"].mean().reset_index()
    )
    submit_df.to_csv(out_path, index=False)
    print("Wrote", out_path, "shape:", submit_df.shape)
    return submit_df




## === cell 3
def _pf1_score(y_true, y_prob, eps=1e-12):
    """
    Probabilistic F1 from the competition description:
      pTP = sum(p_i * y_i)
      pFP = sum(p_i * (1-y_i))
      TP  = sum(y_i)
      FN  = sum((1-p_i)*y_i)  (equivalently TP - pTP)

    pPrecision = pTP / (pTP + pFP)
    pRecall    = pTP / (TP + FN) = pTP / TP
    pF1        = 2 * pPrec * pRec / (pPrec + pRec)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    pTP = float(np.sum(y_prob * y_true))
    pFP = float(np.sum(y_prob * (1.0 - y_true)))
    TP = float(np.sum(y_true))
    if TP <= 0:
        return 0.0

    pPrec = pTP / max(pTP + pFP, eps)
    pRec = pTP / max(TP, eps)
    return 2.0 * pPrec * pRec / max(pPrec + pRec, eps)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _apply_power_calibration(p, power):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.clip(_sigmoid(_logit(p) / float(power)), 1e-6, 1.0 - 1e-6)


def build_metadata_baseline(train_df, test_df, n_splits=5):
    tr = train_df.copy()
    te = test_df.copy()

    required = [
        "age",
        "implant",
        "site_id",
        "machine_id",
        "laterality",
        "view",
        "patient_id",
    ]
    for col in required:
        if col not in tr.columns:
            raise ValueError(f"Missing required column in train: {col}")
        if col not in te.columns:
            raise ValueError(f"Missing required column in test: {col}")

    tr["prediction_id"] = (
        tr["patient_id"].astype(str) + "-" + tr["laterality"].astype(str)
    )

    tr_breast = tr.groupby("prediction_id", as_index=False).agg(
        cancer=("cancer", "max"),
        age=("age", "median"),
        implant=("implant", "max"),
        site_id=("site_id", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
        machine_id=(
            "machine_id",
            lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
        ),
        laterality=(
            "laterality",
            lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
        ),
        view=("view", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
    )

    age_median = float(tr_breast["age"].median())
    tr_breast["age"] = tr_breast["age"].fillna(age_median).astype(float)
    te["age"] = te["age"].fillna(age_median).astype(float)

    a_min, a_max = float(tr_breast["age"].min()), float(tr_breast["age"].max())
    if a_max <= a_min:
        a_max = a_min + 1.0
    tr_breast["age_norm"] = (tr_breast["age"] - a_min) / (a_max - a_min)
    te["age_norm"] = (te["age"] - a_min) / (a_max - a_min)

    tr_breast["implant"] = tr_breast["implant"].fillna(0).astype(int)
    te["implant"] = te["implant"].fillna(0).astype(int)

    global_rate = float(tr_breast["cancer"].mean())

    k = 200.0

    def smooth_stats(col):
        g = tr_breast.groupby(col)["cancer"].agg(["mean", "count"])
        return g

    site_stats = smooth_stats("site_id")
    mach_stats = smooth_stats("machine_id")
    lat_stats = smooth_stats("laterality")
    view_stats = smooth_stats("view")

    def apply_smooth(te_col, stats):
        mean = te_col.map(stats["mean"]).astype(float)
        cnt = te_col.map(stats["count"]).astype(float)
        post = (mean * cnt + global_rate * k) / (cnt + k)
        return post.fillna(global_rate)

    te["p_site"] = apply_smooth(te["site_id"], site_stats)
    te["p_machine"] = apply_smooth(te["machine_id"], mach_stats)
    te["p_lat"] = apply_smooth(te["laterality"], lat_stats)
    te["p_view"] = apply_smooth(te["view"], view_stats)

    p_raw = (
        0.40 * te["p_site"].values
        + 0.35 * te["p_machine"].values
        + 0.15 * te["p_lat"].values
        + 0.10 * te["p_view"].values
    )
    p_raw = (
        p_raw
        * (0.90 + 0.20 * te["age_norm"].values)
        * (1.00 + 0.05 * te["implant"].values)
    )
    p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)

    tr_breast["patient_group"] = (
        tr_breast["prediction_id"].str.split("-").str[0].astype(np.int64)
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_raw = np.zeros(len(tr_breast), dtype=np.float64)

    cols_for_enc = ["site_id", "machine_id", "laterality", "view"]

    for fold, (idx_tr, idx_va) in enumerate(
        gkf.split(tr_breast, tr_breast["cancer"], groups=tr_breast["patient_group"])
    ):
        tr_f = tr_breast.iloc[idx_tr].copy()
        va_f = tr_breast.iloc[idx_va].copy()

        gr = float(tr_f["cancer"].mean())
        kf = k

        def fold_stats(col):
            return tr_f.groupby(col)["cancer"].agg(["mean", "count"])

        stats = {c: fold_stats(c) for c in cols_for_enc}

        def fold_apply(col_series, stat):
            mean = col_series.map(stat["mean"]).astype(float)
            cnt = col_series.map(stat["count"]).astype(float)
            post = (mean * cnt + gr * kf) / (cnt + kf)
            return post.fillna(gr)

        va_f["p_site"] = fold_apply(va_f["site_id"], stats["site_id"])
        va_f["p_machine"] = fold_apply(va_f["machine_id"], stats["machine_id"])
        va_f["p_lat"] = fold_apply(va_f["laterality"], stats["laterality"])
        va_f["p_view"] = fold_apply(va_f["view"], stats["view"])

        p_fold = (
            0.40 * va_f["p_site"].values
            + 0.35 * va_f["p_machine"].values
            + 0.15 * va_f["p_lat"].values
            + 0.10 * va_f["p_view"].values
        )
        p_fold = (
            p_fold
            * (0.90 + 0.20 * va_f["age_norm"].values)
            * (1.00 + 0.05 * va_f["implant"].values)
        )
        p_fold = np.clip(p_fold, 1e-6, 1.0 - 1e-6)

        oof_raw[idx_va] = p_fold

    y_true = tr_breast["cancer"].values.astype(np.float64)

    power_grid = [0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
    best_power = 1.0
    best_pf1 = -1.0
    for pw in power_grid:
        pf1 = _pf1_score(y_true, _apply_power_calibration(oof_raw, pw))
        if pf1 > best_pf1:
            best_pf1 = pf1
            best_power = pw

    print(f"CV-selected calibration power={best_power} (OOF pF1={best_pf1:.6f})")

    p_cal = _apply_power_calibration(p_raw, best_power).astype(np.float32)
    te["cancer_pred"] = p_cal

    return te[["prediction_id", "cancer_pred"]]


pred_image_df = build_metadata_baseline(train_df, test_df)
print(pred_image_df.head())
print("Per-image preds:", pred_image_df.shape)

gc.collect()



## === cell 4
submit_df = (
    pred_image_df.groupby("prediction_id", sort=True)["cancer_pred"]
    .mean()
    .reset_index()
    .rename(columns={"cancer_pred": "cancer"})
)

submit_df = sample_sub[["prediction_id"]].merge(
    submit_df, on="prediction_id", how="left"
)
submit_df["cancer"] = (
    submit_df["cancer"].fillna(float(train_df["cancer"].mean())).astype(np.float32)
)

print("submit_df", submit_df.shape)
print(submit_df.head())



## === cell 5
out_path = "submission.csv"
submit_df.to_csv(out_path, index=False)

assert list(submit_df.columns) == ["prediction_id", "cancer"]
assert submit_df.shape[0] == sample_sub.shape[0]
assert out_path.endswith(".csv")

print("Wrote submission to:", out_path)
print(
    "cancer min/max/mean:",
    float(submit_df["cancer"].min()),
    float(submit_df["cancer"].max()),
    float(submit_df["cancer"].mean()),
)
print("Done.")
