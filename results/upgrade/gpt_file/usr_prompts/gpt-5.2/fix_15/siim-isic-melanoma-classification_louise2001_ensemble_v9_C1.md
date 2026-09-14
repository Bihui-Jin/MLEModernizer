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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9160974650779262

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on the non-existent `/kaggle/input/melanoma` directory that currently crashes the notebook, and make the blending logic robust when no external prediction files are available. The core intent (average-blend of multiple submission files) is preserved: we search for candidate prediction CSVs inside the provided competition input folder and only merge those that match the required `image_name/target` schema. If none are found, we fall back to writing a valid submission using the official `sample_submission.csv` (all zeros), ensuring a `.csv` submission is always produced end-to-end. I also fix a pandas bug (`cols.values()` needs to be converted to a list) and ensure column naming stays exactly `image_name,target`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC indicates the submission is effectively constant (all zeros), so we need to produce non-constant, properly-aligned probabilities without changing the overall “blend CSV predictions” core idea. The smallest reliable improvement is to (1) broaden candidate discovery to also include `.csv.zip` files in the competition folder, (2) fix the merge so we don’t overwrite the `image_name`-only base frame with a renamed `target` column (a bug that currently makes merging behave incorrectly), and (3) if still nothing is found, fall back to a simple metadata-only model (logistic regression) trained on `train.csv` and applied to `test.csv` to generate legitimate probabilities (still lightweight, within constraints, and should move AUC well above 0.5). This keeps the original blending approach when external prediction files exist, and only uses the metadata fallback when needed to avoid constant predictions. The script still writes a valid `submission.csv` with `image_name,target`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC strongly suggests the submission is either constant or nearly constant; the lowest-risk way to move toward the 0.916 target without changing your blending core is to strengthen the metadata fallback so it produces a meaningfully-ranked probability signal. I keep the same lightweight logistic-regression approach, but fix two common AUC-killers: (1) ensure numeric coercion for `age_approx` (it can be parsed as object/strings) and (2) apply per-fold probability calibration via out-of-fold (OOF) training while respecting `patient_id` grouping, then average the fold-trained test predictions. This keeps the same model family/loss (logistic regression), avoids any heavy image modeling, and typically yields a big jump from 0.5 while staying minimal and deterministic. The blending path remains unchanged when external prediction CSVs exist; the improved fallback only triggers when they don’t.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is consistent with producing an almost-constant prediction (e.g., all zeros or near-constant from failed merges), so the smallest safe move toward the 0.916 target is to ensure the metadata fallback reliably outputs well-ranked probabilities. I keep the exact same model family (LogisticRegression) and same overall flow (blend external CSVs if present; otherwise fallback), but (1) make the fallback use `class_weight=None` (matching the natural prevalence for better AUC ranking here), (2) add `C=0.2` mild regularization to improve generalization, and (3) add a tiny amount of deterministic jitter to break any prediction ties (helps AUC when many identical probabilities occur). These are minimal, metric-aligned tweaks that should raise AUC materially above 0.5 without changing the core approach or adding heavy dependencies. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is consistent with tie-heavy or weakly-ranked probabilities from the metadata fallback; we keep the same LogisticRegression+GroupKFold core, but remove the added jitter (it can slightly hurt ranking stability) and strengthen the feature signal with two minimal, metric-aligned additions: one-hot encoding of `patient_id` and a coarse `age_bin` categorical feature. This still uses only metadata, preserves the overall training loop and model family, and should move AUC upward toward the 0.916 target without touching any image modeling. The external CSV blending path remains unchanged and still be used if valid prediction files are found. The script continues to run end-to-end and always writes a valid `submission.csv` with `image_name,target`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC suggests the fallback is still producing weak or near-tied rankings; to move toward the 0.916 target with minimal changes, I keep the exact same fallback model family (LogisticRegression) and GroupKFold-by-patient training loop, but strengthen the metadata signal in a metric-aligned way. Specifically, I add two very cheap numeric features (`age_missing` and `age_approx` squared via a polynomial expansion) while keeping the same preprocessing pipeline and semantics. I also compute and print an out-of-fold AUC as a sanity check (doesn’t affect training/prediction), and I leave the external CSV blending path unchanged. This should improve probability ranking (AUC) without altering the overall approach or adding heavy dependencies, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC indicates the submission is effectively uninformative; since no reliable external prediction CSVs are found, we should focus on making the metadata fallback produce a stronger *ranking* signal without changing the overall approach (GroupKFold-by-patient + LogisticRegression pipeline). I keep the same model family and CV loop, but make two minimal, score-relevant fixes: (1) set `solver="saga"` and a higher `max_iter` so the model reliably converges with the high-dimensional one-hot features, and (2) add `class_weight="balanced"` to counter the heavy class imbalance, which typically improves AUC ranking for this dataset. I also clip predictions to a safe probability range and ensure alignment to `sample_submission` stays exact. The blending logic remains unchanged and still be used if valid prediction files are present.'
- What this solution (achieved 0.5) has done: 'We focus on the metadata fallback since your blend path likely finds no usable external prediction files and that leaves you near-constant (AUC ~0.5). To move AUC upward toward 0.916 with minimal core-logic change, we keep the same GroupKFold-by-patient + LogisticRegression pipeline but (1) switch to a stronger, still-linear classifier (SGDClassifier with log-loss) that handles high-dimensional sparse one-hot features more reliably than LR under tight iteration limits, and (2) add a monotonic probability calibration step (CalibratedClassifierCV with sigmoid) within each fold to improve probability ranking for ROC-AUC. We also ensure `patient_id` is consistently string-typed in both train/test and keep submission alignment strictly by `image_name`. The blend logic and submission-writing remain unchanged.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests your submission is still essentially uninformative (likely the metadata fallback isn’t being used because the script crashes before it runs, since scikit-learn isn’t available in your listed environment). The minimal, score-relevant fix is to remove the scikit-learn dependency in the fallback and replace it with a lightweight, deterministic, metadata-only “risk score” built from target rates in `train.csv` (smoothed mean encoding) that produces non-constant probabilities and usually lifts AUC meaningfully above 0.5. The external CSV blending path is kept intact and unchanged: if valid prediction CSVs are found, we still average them exactly as before. We also keep strict `image_name` alignment to `sample_submission.csv` and always write a valid `submission.csv`. This should move your score upward toward the target without introducing any new packages or changing the blend core logic.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is consistent with writing a nearly constant prediction; since your environment lacks scikit-learn, the metadata fallback is the only place we can legitimately add ranking signal without changing the overall approach. I keep the same fallback idea (smoothed target-rate/mean-encoding from `train.csv` applied to `test.csv`) but make one minimal, score-relevant fix: compute encodings in an out-of-fold (GroupKFold by `patient_id`) way so the feature weights aren’t inflated by target leakage, which typically improves real (test) AUC even if in-sample AUC drops. I also make the blend path more robust by ensuring external prediction columns are numeric and by averaging with `skipna=True` so partial merges don’t collapse to NaNs. The script still runs end-to-end and always writes a valid `submission.csv` with `image_name,target`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC indicates the submission is still effectively uninformative; since external prediction CSVs are unlikely to exist, the metadata fallback must carry the score. To move toward 0.916 with minimal semantic change, I keep the same OOF smoothed mean-encoding approach but add two very small, score-relevant fixes: (1) add mean encodings for `diagnosis` and `benign_malignant` (train-only columns that are legitimately available at training time and strongly predictive) and apply them to test as “unknown” (prior), and (2) replace the current fixed weights with a simple, deterministic ridge regression fit on OOF-encoded features (still linear, still using the same features/encodings, just learning the combination instead of hand-tuning). This should substantially increase ranking signal (AUC) while preserving your overall pipeline (blend if available, otherwise metadata fallback) and still producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC suggests the metadata fallback isn’t adding useful rank signal (or is getting collapsed during merge), so the smallest score-relevant change is to strengthen the fallback in a leakage-safe way while keeping the same mean-encoding + ridge-combination core. I (1) add two well-known, cheap numeric metadata features (`log1p` age and a bounded “young/old” transform) and (2) add fold-wise mean-encoding for the joint category `sex × site` (often more informative than either alone) using the same OOF smoothing function you already use. I also make one robustness fix: ensure the sigmoid computation is numerically stable to avoid accidental saturation/near-constant probabilities (which can tank AUC). The blending path stays identical; these changes only affect the fallback branch.'
- What this solution (achieved 0.5) has done: 'We keep your current “blend external prediction CSVs if present, otherwise metadata fallback” structure unchanged, but fix the likely reason you’re stuck at 0.5 AUC: the fallback can silently become almost-constant due to weak/incorrect encodings and missing-signal columns. The smallest score-relevant improvement (without introducing new dependencies) is to add two strong, leakage-safe OOF mean-encoded features (`sex`×`age_bin` and `patient_id`×`site`) and to ensure the encoded columns are always treated as stable strings (so categories don’t collapse to `"nan"`/`"None"` inconsistently). Finally, we make the ridge output scaling slightly less regularized (alpha) to increase ranking separation, while keeping the same ridge+sigmoid semantics and GroupKFold-by-patient approach. This should move you materially upward from 0.5 toward the 0.916 target, while preserving your core logic and producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC indicates the predictions are still effectively uninformative, so we focus on strengthening the metadata fallback (since external prediction CSVs are usually absent) while preserving the same core approach: OOF smoothed mean-encodings by patient-group folds + ridge + sigmoid. The minimal score-relevant change is to reduce underfitting/tie-heavy outputs by (1) slightly lowering ridge regularization again and (2) adding one more cheap, leakage-safe OOF mean-encoded interaction (`sex × site × age_bin`) plus a bounded “age distance from 50” numeric feature to improve ranking separation. I also add a small robustness fix to ensure categorical normalization happens before interaction construction everywhere (to avoid accidental `"nan"` category splits) and keep submission alignment unchanged. This should move AUC upward toward the 0.916 target without changing the pipeline structure or requiring extra packages.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input/siim-isic-melanoma-classification"
SAMPLE_SUB_PATH = f"{BASE_INPUT}/sample_submission.csv"

base = pd.read_csv(SAMPLE_SUB_PATH)[["image_name"]].copy()
print("Base sample_submission shape:", base.shape)




## === cell 1
def find_candidate_prediction_csvs(root_dir: str):
    """
    Also consider .csv.zip (Kaggle datasets often ship zipped CSVs),
    which increases the chance we actually find prediction files and avoid
    the constant-0 fallback that yields ~0.5 AUC.
    """
    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            fnl = fn.lower()
            if not (fnl.endswith(".csv") or fnl.endswith(".csv.zip")):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.abspath(full) in {
                os.path.abspath(SAMPLE_SUB_PATH),
                os.path.abspath(f"{BASE_INPUT}/train.csv"),
                os.path.abspath(f"{BASE_INPUT}/test.csv"),
                os.path.abspath(f"{BASE_INPUT}/sample_submission.csv.zip"),
                os.path.abspath(f"{BASE_INPUT}/train.csv.zip"),
                os.path.abspath(f"{BASE_INPUT}/test.csv.zip"),
            }:
                continue
            candidates.append(full)
    return candidates


candidate_csvs = find_candidate_prediction_csvs("/kaggle/input")
print("Found candidate CSV/CSV.ZIP files under /kaggle/input:", len(candidate_csvs))



## === cell 2
"""
Fix: merge/rename bug.
We explicitly rename the right column before merge and merge it in cleanly,
so blended predictions are actually used when present.
"""
cols = {}  # maps filename (basename) -> merged column name in base
merged_any = False
f = base.copy()

for path in candidate_csvs:
    try:
        ff = pd.read_csv(path, compression="infer")
    except Exception:
        continue

    if not {"image_name", "target"}.issubset(ff.columns):
        continue

    ff = ff[["image_name", "target"]].copy()
    ff = ff.drop_duplicates(subset=["image_name"], keep="first")

    ff["target"] = pd.to_numeric(ff["target"], errors="coerce")
    ff = ff.dropna(subset=["target"])

    colname = f"target_{len(cols)}"
    ff = ff.rename(columns={"target": colname})

    before_n = len(f)
    tmp = f.merge(ff, on="image_name", how="left")
    if len(tmp) != before_n:
        continue

    non_null = tmp[colname].notna().mean()
    if non_null < 0.95:
        continue

    f = tmp
    cols[os.path.basename(path)] = colname
    merged_any = True

print("Merged prediction files:", len(cols))
if len(cols) > 0:
    print("Example merged columns:", list(cols.items())[:5])



## === cell 3
"""
Change (score-relevant, minimal):
- Keep the same OOF smoothed mean-encoding fallback + ridge combination,
  and add one additional interaction mean-encoding (sex|site|age_bin) and
  one bounded numeric "age distance" feature to increase ranking separation.
- Reduce ridge alpha slightly (15 -> 10) to reduce underfitting / tie-heavy outputs.
All changes preserve the same pipeline structure and evaluation semantics.
"""


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    ys = y_score[order]
    i = 0
    while i < len(ys):
        j = i + 1
        while j < len(ys) and ys[j] == ys[i]:
            j += 1
        if j - i > 1:
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _make_group_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 42):
    """
    Lightweight GroupKFold equivalent (no sklearn dependency).
    Assign groups to folds by shuffled unique groups, approximately balanced by size.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.RandomState(seed)
    order = np.arange(len(uniq))
    rng.shuffle(order)

    gsize = np.bincount(inv, minlength=len(uniq)).astype(np.int64)

    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    fold_for_group = np.empty(len(uniq), dtype=np.int64)

    sorted_groups = sorted(order.tolist(), key=lambda g: (-gsize[g], g))
    for g in sorted_groups:
        k = int(np.argmin(fold_sizes))
        fold_for_group[g] = k
        fold_sizes[k] += gsize[g]

    folds = fold_for_group[inv]
    return folds


def _ridge_fit_predict_oof(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, alpha: float = 20.0
):
    """
    Deterministic ridge regression (closed-form) trained per fold to avoid leakage.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = np.asarray(folds)

    n, d = X.shape
    oof = np.empty(n, dtype=np.float64)
    W = []

    I = np.eye(d + 1, dtype=np.float64)  # include intercept
    I[0, 0] = 0.0  # don't penalize intercept

    for k in np.unique(folds):
        idx_val = np.where(folds == k)[0]
        idx_trn = np.where(folds != k)[0]
        Xt = X[idx_trn]
        yt = y[idx_trn]

        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0) + 1e-8
        Xt_s = (Xt - mu) / sd
        Xv_s = (X[idx_val] - mu) / sd

        Xt_i = np.concatenate([np.ones((Xt_s.shape[0], 1)), Xt_s], axis=1)
        Xv_i = np.concatenate([np.ones((Xv_s.shape[0], 1)), Xv_s], axis=1)

        A = Xt_i.T @ Xt_i + alpha * I
        b = Xt_i.T @ yt
        w = np.linalg.solve(A, b)
        W.append((w, mu, sd))

        oof[idx_val] = Xv_i @ w

    return oof, W


def _ridge_predict_from_fits(X: np.ndarray, fits):
    X = np.asarray(X, dtype=np.float64)
    preds = np.zeros(X.shape[0], dtype=np.float64)
    for w, mu, sd in fits:
        Xs = (X - mu) / sd
        Xi = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        preds += Xi @ w
    preds /= max(1, len(fits))
    return preds


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _norm_cat(s: pd.Series, unknown: str = "NA") -> pd.Series:
    s = s.astype("string")
    s = s.fillna(unknown)
    s = s.str.strip()
    s = s.replace(
        {
            "<NA>": unknown,
            "": unknown,
            "nan": unknown,
            "NaN": unknown,
            "None": unknown,
            "NONE": unknown,
        }
    )
    return s.astype(str)


def metadata_fallback_submission(base_input: str, sample_sub_path: str) -> pd.DataFrame:
    train_path = f"{base_input}/train.csv"
    test_path = f"{base_input}/test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feat_cols = ["patient_id", "sex", "age_approx", "anatom_site_general_challenge"]

    tr = train[
        ["image_name", "target"] + feat_cols + ["diagnosis", "benign_malignant"]
    ].copy()
    te = test[["image_name"] + feat_cols].copy()

    te["diagnosis"] = "Unknown"
    te["benign_malignant"] = "Unknown"

    for df in (tr, te):
        df["patient_id"] = _norm_cat(df["patient_id"], unknown="NA")
        df["sex"] = _norm_cat(df["sex"], unknown="NA")
        df["anatom_site_general_challenge"] = _norm_cat(
            df["anatom_site_general_challenge"], unknown="NA"
        )
        df["diagnosis"] = _norm_cat(df["diagnosis"], unknown="Unknown")
        df["benign_malignant"] = _norm_cat(df["benign_malignant"], unknown="Unknown")

        df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
        df["age_missing"] = df["age_approx"].isna().astype(np.int8)

    tr["sex_site"] = tr["sex"] + "|" + tr["anatom_site_general_challenge"]
    te["sex_site"] = te["sex"] + "|" + te["anatom_site_general_challenge"]

    tr["pid_site"] = tr["patient_id"] + "|" + tr["anatom_site_general_challenge"]
    te["pid_site"] = te["patient_id"] + "|" + te["anatom_site_general_challenge"]

    age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]
    tr["age_bin"] = pd.cut(tr["age_approx"], bins=age_bins, labels=age_labels)
    te["age_bin"] = pd.cut(te["age_approx"], bins=age_bins, labels=age_labels)
    tr["age_bin"] = _norm_cat(tr["age_bin"], unknown="NA")
    te["age_bin"] = _norm_cat(te["age_bin"], unknown="NA")

    tr["sex_agebin"] = tr["sex"] + "|" + tr["age_bin"]
    te["sex_agebin"] = te["sex"] + "|" + te["age_bin"]

    tr["sex_site_agebin"] = (
        tr["sex"] + "|" + tr["anatom_site_general_challenge"] + "|" + tr["age_bin"]
    )
    te["sex_site_agebin"] = (
        te["sex"] + "|" + te["anatom_site_general_challenge"] + "|" + te["age_bin"]
    )

    y = tr["target"].astype(int).values
    prior = float(np.mean(y))

    folds = _make_group_folds(tr["patient_id"].values, n_splits=5, seed=42)

    def smoothed_mean_encode_oof(
        train_df, test_df, col, target_col="target", m=50.0, folds_arr=None
    ):
        full_prior = float(train_df[target_col].mean())
        grp_full = train_df.groupby(col, dropna=False)[target_col].agg(
            ["mean", "count"]
        )
        enc_full = (grp_full["count"] * grp_full["mean"] + m * full_prior) / (
            grp_full["count"] + m
        )
        test_enc = (
            test_df[col].map(enc_full).astype(np.float64).fillna(full_prior).values
        )

        oof = np.empty(len(train_df), dtype=np.float64)
        for k in np.unique(folds_arr):
            idx_val = np.where(folds_arr == k)[0]
            idx_trn = np.where(folds_arr != k)[0]
            df_trn = train_df.iloc[idx_trn]
            pr = float(df_trn[target_col].mean())
            grp = df_trn.groupby(col, dropna=False)[target_col].agg(["mean", "count"])
            enc = (grp["count"] * grp["mean"] + m * pr) / (grp["count"] + m)
            oof[idx_val] = (
                train_df.iloc[idx_val][col]
                .map(enc)
                .astype(np.float64)
                .fillna(pr)
                .values
            )
        return oof, test_enc

    tr_pid, te_pid = smoothed_mean_encode_oof(
        tr, te, "patient_id", m=200.0, folds_arr=folds
    )
    tr_sex, te_sex = smoothed_mean_encode_oof(tr, te, "sex", m=50.0, folds_arr=folds)
    tr_site, te_site = smoothed_mean_encode_oof(
        tr, te, "anatom_site_general_challenge", m=50.0, folds_arr=folds
    )
    tr_sexsite, te_sexsite = smoothed_mean_encode_oof(
        tr, te, "sex_site", m=50.0, folds_arr=folds
    )
    tr_pidsite, te_pidsite = smoothed_mean_encode_oof(
        tr, te, "pid_site", m=200.0, folds_arr=folds
    )
    tr_sexage, te_sexage = smoothed_mean_encode_oof(
        tr, te, "sex_agebin", m=50.0, folds_arr=folds
    )

    tr_ssa, te_ssa = smoothed_mean_encode_oof(
        tr, te, "sex_site_agebin", m=80.0, folds_arr=folds
    )

    tr_agebin, te_agebin = smoothed_mean_encode_oof(
        tr, te, "age_bin", m=50.0, folds_arr=folds
    )
    tr_missing, te_missing = smoothed_mean_encode_oof(
        tr, te, "age_missing", m=50.0, folds_arr=folds
    )
    tr_diag, te_diag = smoothed_mean_encode_oof(
        tr, te, "diagnosis", m=50.0, folds_arr=folds
    )
    tr_bm, te_bm = smoothed_mean_encode_oof(
        tr, te, "benign_malignant", m=50.0, folds_arr=folds
    )

    age_med = float(np.nanmedian(tr["age_approx"].values))
    tr_age = np.where(
        np.isnan(tr["age_approx"].values), age_med, tr["age_approx"].values
    ).astype(np.float64)
    te_age = np.where(
        np.isnan(te["age_approx"].values), age_med, te["age_approx"].values
    ).astype(np.float64)

    age_mean = float(tr_age.mean())
    age_std = float(tr_age.std() + 1e-6)
    tr_age_z = (tr_age - age_mean) / age_std
    te_age_z = (te_age - age_mean) / age_std

    tr_age_log = np.log1p(np.clip(tr_age, 0.0, None))
    te_age_log = np.log1p(np.clip(te_age, 0.0, None))
    tr_age_tanh = np.tanh((tr_age - 50.0) / 20.0)
    te_age_tanh = np.tanh((te_age - 50.0) / 20.0)

    tr_age_dist = np.clip(np.abs(tr_age - 50.0) / 50.0, 0.0, 1.0)
    te_age_dist = np.clip(np.abs(te_age - 50.0) / 50.0, 0.0, 1.0)

    X_tr = np.vstack(
        [
            (tr_pid - prior),
            (tr_site - prior),
            (tr_sexsite - prior),
            (tr_pidsite - prior),
            (tr_sexage - prior),
            (tr_ssa - prior),  # added
            (tr_agebin - prior),
            (tr_sex - prior),
            (tr_missing - prior),
            (tr_diag - prior),
            (tr_bm - prior),
            tr_age_z,
            tr_age_log,
            tr_age_tanh,
            tr_age_dist,  # added
        ]
    ).T
    X_te = np.vstack(
        [
            (te_pid - prior),
            (te_site - prior),
            (te_sexsite - prior),
            (te_pidsite - prior),
            (te_sexage - prior),
            (te_ssa - prior),  # added
            (te_agebin - prior),
            (te_sex - prior),
            (te_missing - prior),
            (te_diag - prior),
            (te_bm - prior),
            te_age_z,
            te_age_log,
            te_age_tanh,
            te_age_dist,  # added
        ]
    ).T

    oof_lin, fits = _ridge_fit_predict_oof(
        X_tr, y.astype(np.float64), folds, alpha=10.0
    )
    te_lin = _ridge_predict_from_fits(X_te, fits)

    prior_c = np.clip(prior, 1e-6, 1 - 1e-6)
    prior_logit = np.log(prior_c / (1.0 - prior_c))

    tr_prob = _sigmoid_stable(prior_logit + oof_lin)
    te_prob = _sigmoid_stable(prior_logit + te_lin)

    tr_prob = np.clip(tr_prob, 1e-6, 1.0 - 1e-6)
    te_prob = np.clip(te_prob, 1e-6, 1.0 - 1e-6)

    try:
        auc_oof = _safe_roc_auc(y, tr_prob)
        print("Train ROC-AUC (sanity check, OOF-encoded + ridge):", auc_oof)
    except Exception as e:
        print("AUC sanity check failed:", repr(e))

    sub = pd.read_csv(sample_sub_path)[["image_name"]].copy()
    pred_df = pd.DataFrame(
        {"image_name": te["image_name"].values, "target": te_prob.astype(np.float64)}
    )
    sub = sub.merge(pred_df, on="image_name", how="left")
    sub["target"] = sub["target"].fillna(float(np.mean(te_prob)))
    sub["target"] = np.clip(sub["target"].astype(np.float64).values, 1e-6, 1.0 - 1e-6)
    sub["target"] = sub["target"].astype(np.float64)
    return sub[["image_name", "target"]]


if merged_any:
    pred_cols = list(cols.values())
    f["target"] = f[pred_cols].mean(axis=1, skipna=True)
    if f["target"].isna().any():
        f["target"] = f["target"].fillna(float(f["target"].mean()))
    f_ = f[["image_name", "target"]].copy()
else:
    f_ = metadata_fallback_submission(BASE_INPUT, SAMPLE_SUB_PATH)

print("Final submission shape:", f_.shape)
print(f_.head())
print("Target stats:", f_["target"].describe())



## === cell 4
out_path = "submission.csv"
f_.to_csv(out_path, index=False)
print("Wrote:", out_path)



## === cell 5
pass
