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

0.04395

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02342) has done: 'The main blocker is that the notebook depends on external Kaggle input modules (`dicom_reader.py`, `preprocess.py`, `nextvit.py`, and several pip-installed wheels) that are not available in your provided environment, so imports fail and `test_df` is never created. To make this run end-to-end without changing the overall “read test.csv → predict per image → group by prediction_id → write submission.csv” semantics, I replace the missing DICOM/PNG/model pipeline with a lightweight metadata-only probabilistic baseline and keep the submission formatting identical to the competition requirement. I also fix the incorrect cell numbering (your script starts at cell 0) and ensure all required imports are present so `pd`, `Dataset`, etc. are defined. Finally, the code always writes `submission.csv` with columns `prediction_id,cancer` and the correct number of rows matching `sample_submission.csv`.'
- What this solution (achieved 0.0412) has done: 'Your current metadata-only baseline is likely under-scoring because it mixes per-image “proxy risks” in a way that doesn’t match the metric’s need for sharp separation between positives and negatives. To move your pF1 upward toward the 0.45 target without changing the overall pipeline (read CSVs → compute per-image probabilities → aggregate by prediction_id → write submission), I (1) compute smoothed target rates at the *prediction_id* level for train and apply them to test via site/machine/view/laterality, (2) calibrate the final probabilities with a single temperature-like power transform chosen by GroupKFold cross-validation to directly maximize pF1 on out-of-fold predictions, and (3) keep the same submission formatting/alignment with `sample_submission.csv`. These are minimal, metric-aligned changes that keep the “metadata baseline” core logic intact while making probabilities better calibrated for pF1.'
- What this solution (achieved 0.04159) has done: 'Your current pipeline is already valid and score-limited by weak signal; the simplest metric-aligned improvement that preserves your “metadata-only → per-image probs → groupby prediction_id → submission.csv” core logic is to (1) add a *pairwise* smoothed target encoding for `(site_id, machine_id)` (often more informative than either alone) and `(site_id, view)` while keeping the existing single-column encodings, and (2) tune the linear blend weights and the smoothing strength `k` using the same GroupKFold OOF pF1 selection you already do for the power calibration. This keeps architecture/training semantics the same (still a calibrated smoothed-rate baseline), but should materially increase separation and move your public score upward toward the 0.45 target. I also keep submission alignment via `sample_submission.csv` unchanged and ensure deterministic behavior.'
- What this solution (achieved 0.04396) has done: 'I keep your metadata-only smoothed target-encoding baseline intact, but add two high-signal, low-risk encodings that fit the same framework: smoothed target rates for `site_id + laterality` and `site_id + machine_id + view`. Then I minimally extend your existing OOF pF1 grid-search to include these new encodings and re-tune the blend weights (still a simple linear blend of smoothed rates) so the probabilities better separate positives/negatives for pF1. Finally, I keep your current power-calibration step, but slightly widen the power grid (still deterministic, same calibration mechanism) to help pF1 without changing evaluation semantics or requiring any extra packages.'
- What this solution (achieved 0.04395) has done: 'Your current score (0.04396) is far below the target (0.45176), so we should push performance upward with minimal, metric-aligned changes while keeping your “smoothed target-encoding blend + power calibration + mean-by-prediction_id” core logic intact. The biggest likely issue is a train/test mismatch: you train encodings at the *breast (prediction_id)* level but apply them to *image-level* test rows (multiple rows share the same prediction_id), which can dilute signal and hurt pF1. I (1) aggregate test metadata to the same breast level before encoding/scoring, then (2) map the calibrated breast-level probability back onto all test rows so the final groupby mean is consistent (and effectively identity). This preserves your approach but fixes alignment, which is a high-impact correctness/performance improvement toward the target.'

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
    Probabilistic F1 from the competition description.
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


def _mode_or_first(x):
    m = x.mode()
    if len(m):
        return m.iloc[0]
    return x.iloc[0]


def _aggregate_to_breast_level(df, is_train: bool):
    """
    Change motivated by score-matching: training encodings are built at breast (prediction_id) level,
    but previous code applied them to image-level test rows, diluting signal. We aggregate test to the
    same breast level to improve separation (higher pF1) without changing the core encoding logic.
    """
    d = df.copy()
    if is_train:
        d["prediction_id"] = (
            d["patient_id"].astype(str) + "-" + d["laterality"].astype(str)
        )

    agg_dict = {
        "age": ("age", "median"),
        "implant": ("implant", "max"),
        "site_id": ("site_id", _mode_or_first),
        "machine_id": ("machine_id", _mode_or_first),
        "laterality": ("laterality", _mode_or_first),
        "view": ("view", _mode_or_first),
        "patient_id": ("patient_id", _mode_or_first),
    }
    if is_train:
        agg_dict["cancer"] = ("cancer", "max")

    out = d.groupby("prediction_id", as_index=False).agg(**agg_dict)
    return out


def build_metadata_baseline(train_df, test_df, n_splits=5):
    tr_breast = _aggregate_to_breast_level(train_df, is_train=True)
    te_breast = _aggregate_to_breast_level(test_df, is_train=False)

    required = [
        "age",
        "implant",
        "site_id",
        "machine_id",
        "laterality",
        "view",
        "patient_id",
        "prediction_id",
    ]
    for col in required:
        if col not in tr_breast.columns:
            raise ValueError(f"Missing required column in train breast table: {col}")
        if col not in te_breast.columns:
            raise ValueError(f"Missing required column in test breast table: {col}")

    age_median = float(tr_breast["age"].median())
    tr_breast["age"] = tr_breast["age"].fillna(age_median).astype(float)
    te_breast["age"] = te_breast["age"].fillna(age_median).astype(float)

    a_min, a_max = float(tr_breast["age"].min()), float(tr_breast["age"].max())
    if a_max <= a_min:
        a_max = a_min + 1.0
    tr_breast["age_norm"] = (tr_breast["age"] - a_min) / (a_max - a_min)
    te_breast["age_norm"] = (te_breast["age"] - a_min) / (a_max - a_min)

    tr_breast["implant"] = tr_breast["implant"].fillna(0).astype(int)
    te_breast["implant"] = te_breast["implant"].fillna(0).astype(int)

    global_rate = float(tr_breast["cancer"].mean())

    def add_pair_keys(df):
        df = df.copy()
        df["site_machine"] = (
            df["site_id"].astype(str) + "_" + df["machine_id"].astype(str)
        )
        df["site_view"] = df["site_id"].astype(str) + "_" + df["view"].astype(str)
        df["site_lat"] = df["site_id"].astype(str) + "_" + df["laterality"].astype(str)
        df["site_machine_view"] = (
            df["site_id"].astype(str)
            + "_"
            + df["machine_id"].astype(str)
            + "_"
            + df["view"].astype(str)
        )
        return df

    tr_breast = add_pair_keys(tr_breast)
    te_breast = add_pair_keys(te_breast)

    def _smooth_series(col_series, stats_df, gr, k):
        mean = col_series.map(stats_df["mean"]).astype(float)
        cnt = col_series.map(stats_df["count"]).astype(float)
        post = (mean * cnt + gr * k) / (cnt + k)
        return post.fillna(gr)

    tr_breast["patient_group"] = tr_breast["patient_id"].astype(np.int64)
    gkf = GroupKFold(n_splits=n_splits)

    y_true = tr_breast["cancer"].values.astype(np.float64)

    k_grid = [50.0, 120.0, 200.0, 350.0]

    weight_grid = [
        (0.18, 0.14, 0.08, 0.08, 0.22, 0.08, 0.10, 0.12),
        (0.16, 0.12, 0.08, 0.08, 0.24, 0.08, 0.10, 0.14),
        (0.14, 0.12, 0.08, 0.08, 0.26, 0.08, 0.10, 0.14),
        (0.14, 0.10, 0.08, 0.08, 0.24, 0.08, 0.12, 0.16),
        (0.12, 0.10, 0.08, 0.08, 0.24, 0.08, 0.12, 0.18),
    ]

    best_cfg = None
    best_pf1_raw = -1.0
    best_oof_raw = None

    enc_cols = [
        "site_id",
        "machine_id",
        "laterality",
        "view",
        "site_machine",
        "site_view",
        "site_lat",
        "site_machine_view",
    ]

    for k in k_grid:
        for w in weight_grid:
            (
                w_site,
                w_mach,
                w_lat,
                w_view,
                w_sm,
                w_sv,
                w_sl,
                w_smv,
            ) = w
            if abs((sum(w)) - 1.0) > 1e-9:
                continue

            oof_raw = np.zeros(len(tr_breast), dtype=np.float64)

            for idx_tr, idx_va in gkf.split(
                tr_breast, tr_breast["cancer"], groups=tr_breast["patient_group"]
            ):
                tr_f = tr_breast.iloc[idx_tr]
                va_f = tr_breast.iloc[idx_va]

                gr = float(tr_f["cancer"].mean())

                stats = {}
                for c in enc_cols:
                    stats[c] = tr_f.groupby(c)["cancer"].agg(["mean", "count"])

                p_site = _smooth_series(va_f["site_id"], stats["site_id"], gr, k)
                p_mach = _smooth_series(va_f["machine_id"], stats["machine_id"], gr, k)
                p_lat = _smooth_series(va_f["laterality"], stats["laterality"], gr, k)
                p_view = _smooth_series(va_f["view"], stats["view"], gr, k)
                p_sm = _smooth_series(
                    va_f["site_machine"], stats["site_machine"], gr, k
                )
                p_sv = _smooth_series(va_f["site_view"], stats["site_view"], gr, k)
                p_sl = _smooth_series(va_f["site_lat"], stats["site_lat"], gr, k)
                p_smv = _smooth_series(
                    va_f["site_machine_view"], stats["site_machine_view"], gr, k
                )

                p_fold = (
                    w_site * p_site.values
                    + w_mach * p_mach.values
                    + w_lat * p_lat.values
                    + w_view * p_view.values
                    + w_sm * p_sm.values
                    + w_sv * p_sv.values
                    + w_sl * p_sl.values
                    + w_smv * p_smv.values
                )
                p_fold = (
                    p_fold
                    * (0.90 + 0.20 * va_f["age_norm"].values)
                    * (1.00 + 0.05 * va_f["implant"].values)
                )
                p_fold = np.clip(p_fold, 1e-6, 1.0 - 1e-6)
                oof_raw[idx_va] = p_fold

            pf1_raw = _pf1_score(y_true, oof_raw)
            if pf1_raw > best_pf1_raw:
                best_pf1_raw = pf1_raw
                best_cfg = (k, w)
                best_oof_raw = oof_raw.copy()

    k_best, w_best = best_cfg
    print(
        f"Selected encoding blend via OOF pF1: k={k_best}, weights={w_best}, OOF raw pF1={best_pf1_raw:.6f}"
    )

    def smooth_stats_full(col):
        return tr_breast.groupby(col)["cancer"].agg(["mean", "count"])

    stats_full = {c: smooth_stats_full(c) for c in enc_cols}
    (
        w_site,
        w_mach,
        w_lat,
        w_view,
        w_sm,
        w_sv,
        w_sl,
        w_smv,
    ) = w_best

    te_breast["p_site"] = _smooth_series(
        te_breast["site_id"], stats_full["site_id"], global_rate, k_best
    )
    te_breast["p_machine"] = _smooth_series(
        te_breast["machine_id"], stats_full["machine_id"], global_rate, k_best
    )
    te_breast["p_lat"] = _smooth_series(
        te_breast["laterality"], stats_full["laterality"], global_rate, k_best
    )
    te_breast["p_view"] = _smooth_series(
        te_breast["view"], stats_full["view"], global_rate, k_best
    )
    te_breast["p_site_machine"] = _smooth_series(
        te_breast["site_machine"], stats_full["site_machine"], global_rate, k_best
    )
    te_breast["p_site_view"] = _smooth_series(
        te_breast["site_view"], stats_full["site_view"], global_rate, k_best
    )
    te_breast["p_site_lat"] = _smooth_series(
        te_breast["site_lat"], stats_full["site_lat"], global_rate, k_best
    )
    te_breast["p_site_machine_view"] = _smooth_series(
        te_breast["site_machine_view"],
        stats_full["site_machine_view"],
        global_rate,
        k_best,
    )

    p_raw = (
        w_site * te_breast["p_site"].values
        + w_mach * te_breast["p_machine"].values
        + w_lat * te_breast["p_lat"].values
        + w_view * te_breast["p_view"].values
        + w_sm * te_breast["p_site_machine"].values
        + w_sv * te_breast["p_site_view"].values
        + w_sl * te_breast["p_site_lat"].values
        + w_smv * te_breast["p_site_machine_view"].values
    )
    p_raw = (
        p_raw
        * (0.90 + 0.20 * te_breast["age_norm"].values)
        * (1.00 + 0.05 * te_breast["implant"].values)
    )
    p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)

    power_grid = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    best_power = 1.0
    best_pf1 = -1.0
    for pw in power_grid:
        pf1 = _pf1_score(y_true, _apply_power_calibration(best_oof_raw, pw))
        if pf1 > best_pf1:
            best_pf1 = pf1
            best_power = pw

    print(f"CV-selected calibration power={best_power} (OOF pF1={best_pf1:.6f})")

    te_breast["cancer_pred"] = _apply_power_calibration(p_raw, best_power).astype(
        np.float32
    )

    pred_map = te_breast.set_index("prediction_id")["cancer_pred"]
    te_img = test_df[["prediction_id"]].copy()
    te_img["cancer_pred"] = te_img["prediction_id"].map(pred_map).astype(np.float32)

    return te_img


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
