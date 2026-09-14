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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9699

# 6. Current score

0.66326

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the immediate crash by removing the hard dependency on a non-existent `/kaggle/input/submissions/` folder and instead default to the competition’s provided `sample_submission.csv`/`test.csv` when no external submissions are found. I keep the ensemble logic intact, but add validation so it only uses available submission indices and checks required columns, preventing index/column errors. Finally, I ensure a valid `submission.csv` is always written with the correct columns and row order aligned to `test.csv`, so you get a Kaggle-uploadable file end-to-end.'
- What this solution (achieved 0.58918) has done: 'Your current 0.5 score comes from outputting the sample-submission priors (all 0.25) or an ensemble of unknown external CSVs; without an actual model, ROC AUC stay around chance. To move toward the 0.9699 target while keeping changes minimal, I add a lightweight, self-contained classical ML image pipeline (HOG features + One-vs-Rest Logistic Regression) trained on the provided train images and labels, then predict probabilities for test images. This preserves the existing submission-writing logic and schema, but changes the source of `submission_avg` from a constant/unknown ensemble to learned predictions. If `/kaggle/input/submissions/` exists, the code still be able to ensemble those; otherwise it train and generate a proper model-based submission.'
- What this solution (achieved 0.68575) has done: 'Your current score is far below the 0.9699 target, so we need a small, legitimate lift without changing the overall approach (still HOG features + One-vs-Rest Logistic Regression + same submission writer). The biggest low-risk gains here come from (1) extracting HOG from color channels (more signal than grayscale) while keeping the same feature-extraction paradigm, and (2) making LogisticRegression closer to a “proper” probabilistic multi-label setup by using `class_weight="balanced"` and a slightly higher `max_iter` for convergence stability. I also add deterministic settings and a fallback to the competition’s sample_submission template even when ensembling external CSVs, ensuring the output format and row alignment are always correct. These changes should improve ROC AUC materially while remaining within the same core logic and running comfortably within the time limit.'
- What this solution (achieved 0.58196) has done: 'The timeout is dominated by slow per-image PIL resizing and repeated per-channel HOG extraction in Python loops, plus single-threaded HOG and LogisticRegression. I keep the exact same feature definition (RGB HOG with identical parameters) and the same OneVsRest(LogisticRegression saga) training, but make it faster by (1) using skimage’s loader+resize (faster than PIL here), (2) computing HOG for all 3 channels in one call via `channel_axis`, (3) parallelizing feature extraction across CPU cores with joblib (purely independent per-image work), and (4) enabling safe multi-core use in the OVR classifier. I also avoid extra intermediate lists/copies and preallocate arrays while preserving identical semantics and stable randomness.'
- What this solution (achieved 0.66288) has done: 'Your current gap to the 0.9699 target is large, and the simplest legitimate way to move up without changing the core approach (RGB HOG features + One-vs-Rest Logistic Regression) is to fix two high-impact “quiet” issues: (1) HOG should be computed on a normalized float image (0–1) because `transform_sqrt=True` assumes intensity scale, and your current `preserve_range=True` makes this inconsistent across files; and (2) your model is likely under-regularized/misalibrated for mean ROC AUC, so switching LogisticRegression to `penalty="l2"` with `solver="lbfgs"` (still LogisticRegression, same OVR pipeline) and using a slightly more informative HOG setup (more orientations, slightly larger input) typically lifts AUC materially while keeping the same logic. I also ensure the test image order is exactly respected and keep the external-submission ensemble path unchanged. These are minimal, metric-aligned tweaks that should improve score without introducing new training loops or a different model family.'
- What this solution (achieved 0.66314) has done: 'Your current score (0.66288) is far below the 0.9699 target, so we should make a small, metric-aligned improvement while preserving the same core pipeline (RGB HOG features + OneVsRest LogisticRegression). The most direct lift with minimal semantic change is to add a simple train/validation split and use the validation to choose `C` (regularization strength) from a tiny grid, selecting by mean ROC AUC, then retrain on full data with the best `C`. This keeps the exact same model family, feature extraction, and training approach, but avoids a likely under/over-regularized setting that depresses AUC. I also ensure deterministic splits and keep the submission writer and external-ensemble behavior unchanged.'
- What this solution (achieved 0.4912) has done: 'I fix the crash by ensuring the HOG feature matrix never contains NaNs (they can arise if an image read fails or if any intermediate numeric operation produces non-finite values). This is done in a minimal, score-neutral way: robust image reading (convert to RGB), strict float conversion, and a final `np.nan_to_num` cleanup on the extracted feature arrays before model fitting/prediction. I also keep the existing pipeline (RGB HOG → StandardScaler → OneVsRest(LogisticRegression) with the tiny C-grid) unchanged otherwise, and still write a correctly formatted `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.65401) has done: 'Your current score dropped because the “robustness” change effectively made many feature vectors identical/near-identical (zeros) when image reading or HOG returned non-finite values, which collapses discrimination and tanks ROC AUC. To move back toward the target with minimal logic changes, I keep the exact same pipeline (RGB HOG → StandardScaler → OneVsRest(LogisticRegression) with the same tiny C-grid) but make feature extraction reliably non-degenerate by switching to `skimage.io.imread` (more reliable on Kaggle images) and removing the per-image z-score normalization that can amplify noise/edge cases. I also ensure class probabilities are returned in the exact target column order by explicitly mapping `ovr.classes_` to the required label order before writing the submission, preventing subtle column misalignment that can severely hurt AUC. These are small, directly score-relevant fixes and should raise performance materially without changing the modeling approach.'
- What this solution (achieved 0.65401) has done: 'Your current score is far below the target, so we should make a small but high-impact fix that keeps the exact same pipeline (RGB HOG → StandardScaler → OneVsRest LogisticRegression) while correcting two likely silent issues: image path resolution and class/probability column alignment. First, I make image loading robust to both `Train_*.jpg/Test_*.jpg` and plain `{image_id}.jpg` filenames so you don’t end up with many zero-filled feature rows that destroy AUC. Second, I explicitly map predicted probabilities to the required label order using the estimator’s learned class order, and I also align external submissions by `image_id` when ensembling to prevent row-order mismatches. These changes preserve your core logic and should move ROC AUC upward toward the target without changing the modeling approach.'
- What this solution (achieved 0.65401) has done: 'Your current score suggests many training/test images are still being missed or mis-read, which silently fills rows with zero features and destroys ROC AUC. I make a minimal but high-impact fix by resolving image paths via a one-time index of the `images/` folder (so `Train_*.jpg` / `Test_*.jpg` are always found reliably), while keeping the same HOG(RGB) + StandardScaler + OneVsRest(LogisticRegression) pipeline unchanged. I also ensure the probability columns are guaranteed to be in the required label order (defensive against any estimator class-order quirks) and keep the existing external-submission ensemble behavior intact. These changes are directly aimed at improving feature quality and label/prob alignment to move your score upward toward the 0.9699 target.'
- What this solution (achieved 0.66317) has done: 'Your current score is far below the target, so the smallest score-relevant lift while preserving the same core pipeline (RGB HOG → StandardScaler → OneVsRest(LogisticRegression)) is to (1) fix a likely silent probability/class-order bug by using the *pipeline’s* `predict_proba` (not the inner OVR estimator directly), and (2) reduce the number of “all-zero” feature rows by making image indexing recursive over nested `images/` folders and resolving IDs more reliably. These changes do not alter the model family, feature type, or training loop; they mainly ensure you’re actually training/predicting on the correct images and outputting correctly ordered probabilities. I keep the external-submission ensembling path intact and still always write a valid `submission.csv` aligned to `test.csv`. This should move ROC AUC upward toward your target without a major rewrite.'
- What this solution (achieved 0.66326) has done: 'We keep your exact core pipeline (RGB HOG → StandardScaler → OneVsRest(LogisticRegression)) and only make two score-relevant fixes that commonly suppress ROC AUC without changing the modeling approach. First, we use a stratified split based on the multiclass argmax label (still training the same multi-label targets) so validation-based C selection is stable and representative. Second, we address class imbalance more appropriately for ROC AUC by removing `class_weight="balanced"` (it can distort probability ranking) and by increasing `C_grid` slightly while keeping the same tiny grid-search logic. Everything else (paths, feature extraction, training loop structure, submission writing, and optional external-CSV ensembling) remains the same and still outputs a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

SUBMISSIONS_PATH = "/kaggle/input/submissions/"  # may not exist
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7/",
    "/kaggle/data/plant-pathology-2020-fgvc7/",
    "/kaggle/input/",
    "/kaggle/data/",
]


def _first_existing(path_list, filename):
    for root in path_list:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    return None


sample_sub_path = _first_existing(DATA_ROOT_CANDIDATES, "sample_submission.csv")
test_csv_path = _first_existing(DATA_ROOT_CANDIDATES, "test.csv")
train_csv_path = _first_existing(DATA_ROOT_CANDIDATES, "train.csv")

images_dir = None
for root in DATA_ROOT_CANDIDATES:
    cand = os.path.join(root, "images")
    if os.path.isdir(cand):
        images_dir = cand
        break

if (
    sample_sub_path is None
    or test_csv_path is None
    or train_csv_path is None
    or images_dir is None
):
    raise FileNotFoundError(
        "Could not locate sample_submission.csv, test.csv, train.csv and/or images/ in expected Kaggle paths."
    )

print("sample_submission.csv:", sample_sub_path)
print("test.csv:", test_csv_path)
print("train.csv:", train_csv_path)
print("images_dir:", images_dir)



## === cell 1
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[], test_csv_path=None):
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if test_csv_path is None:
        raise ValueError("test_csv_path is required for safe ensembling/alignment.")
    test_df = pd.read_csv(test_csv_path)
    test_ids = test_df["image_id"].values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx} but only {len(submissions_all)} files exist."
            )

        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])

        missing = [c for c in required_cols if c not in submission.columns]
        if missing:
            raise ValueError(
                f"Submission file {submissions_all[idx]} missing columns: {missing}"
            )

        submission = submission.set_index("image_id").reindex(test_ids)
        if submission.isna().any().any():
            submission[["healthy", "multiple_diseases", "rust", "scab"]] = submission[
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].fillna(0.25)

        submission_vals = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission_vals * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, template_csv_path, test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    template_df = pd.read_csv(template_csv_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in template_df.columns]
    if missing:
        raise ValueError(f"Template submission missing columns: {missing}")

    submission_df = template_df.loc[:, required_cols].copy()
    submission_df = submission_df.iloc[: len(test_df)].copy()

    submission_df["image_id"] = test_df["image_id"].values

    if submission_avg.shape != (len(test_df), 4):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected {(len(test_df), 4)}"
        )

    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)




## === cell 4
def _build_image_id_to_path_index(images_dir):
    mapping = {}
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            base = os.path.splitext(fn)[0]  # e.g., "Train_370" / "Test_59"
            full = os.path.join(root, fn)
            mapping[base] = full
    return mapping


def _resolve_image_path(image_id, images_dir, name_to_path):
    iid = str(image_id)

    if iid in name_to_path:
        return name_to_path[iid]

    if iid.startswith("Train_") or iid.startswith("Test_"):
        if iid in name_to_path:
            return name_to_path[iid]

    k1 = f"Train_{iid}"
    if k1 in name_to_path:
        return name_to_path[k1]
    k2 = f"Test_{iid}"
    if k2 in name_to_path:
        return name_to_path[k2]

    return os.path.join(images_dir, f"{iid}.jpg")


def _extract_hog_features_rgb(image_ids, images_dir, target_size=(160, 160)):
    import numpy as np
    from skimage.feature import hog
    from skimage.transform import resize
    from skimage import io
    from joblib import Parallel, delayed
    import multiprocessing as mp

    H, W = target_size
    name_to_path = _build_image_id_to_path_index(images_dir)

    def _hog_one(iid):
        p = _resolve_image_path(iid, images_dir, name_to_path)
        if not os.path.exists(p):
            return None

        try:
            img = io.imread(p)
        except Exception:
            return None

        try:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            elif img.shape[-1] != 3:
                return None
        except Exception:
            return None

        img = img.astype(np.float32, copy=False)
        if img.max() > 1.5:
            img = img / 255.0

        img = resize(
            img,
            (H, W, 3),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=False,  # float in [0,1]
        ).astype(np.float32, copy=False)

        f = hog(
            img,
            orientations=12,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
            channel_axis=-1,
        ).astype(np.float32, copy=False)

        if not np.all(np.isfinite(f)):
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
        return f

    n_jobs = min(4, mp.cpu_count())
    feats = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=32)(
        delayed(_hog_one)(iid) for iid in image_ids
    )

    feat_len = None
    missing_paths = 0
    for f in feats:
        if f is None:
            missing_paths += 1
        elif feat_len is None:
            feat_len = f.size
    if feat_len is None:
        raise RuntimeError("No images could be loaded to extract features.")

    X = np.zeros((len(image_ids), feat_len), dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            X[i, :] = f

    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if missing_paths:
        print(f"Warning: {missing_paths} images missing/unreadable; filled with zeros.")
    return X


def train_and_predict_probs(train_csv_path, test_csv_path, images_dir):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in ["image_id"] + target_cols:
        if c not in train_df.columns:
            raise ValueError(f"train.csv missing column: {c}")

    train_ids = train_df["image_id"].values
    test_ids = test_df["image_id"].values

    X_train_full = _extract_hog_features_rgb(train_ids, images_dir)
    X_test = _extract_hog_features_rgb(test_ids, images_dir)
    y_full = train_df[target_cols].values.astype(np.int32)

    if not np.all(np.isfinite(X_train_full)):
        X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(X_test)):
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    strat_labels = np.argmax(y_full, axis=1)

    idx = np.arange(len(train_ids))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=strat_labels,
    )

    X_tr, y_tr = X_train_full[tr_idx], y_full[tr_idx]
    X_va, y_va = X_train_full[va_idx], y_full[va_idx]

    def _make_clf(C_value):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "ovr",
                    OneVsRestClassifier(
                        LogisticRegression(
                            solver="lbfgs",
                            penalty="l2",
                            max_iter=5000,
                            C=C_value,
                            class_weight=None,
                            random_state=42,
                            n_jobs=1,
                        ),
                        n_jobs=4,
                    ),
                ),
            ]
        )

    def _mean_colwise_auc(y_true, y_pred):
        aucs = []
        for j in range(y_true.shape[1]):
            if len(np.unique(y_true[:, j])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
        return float(np.mean(aucs)) if len(aucs) else 0.5

    C_grid = [0.2, 0.5, 1.0, 2.0, 5.0]
    best_C, best_auc = None, -1.0
    for C_value in C_grid:
        clf = _make_clf(C_value)
        clf.fit(X_tr, y_tr)
        va_pred = clf.predict_proba(X_va)
        auc = _mean_colwise_auc(y_va, va_pred)
        print(f"Validation mean column-wise ROC AUC for C={C_value}: {auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_C = C_value

    print(
        f"Selected C={best_C} based on validation mean column-wise ROC AUC={best_auc:.6f}"
    )

    final_clf = _make_clf(best_C)
    final_clf.fit(X_train_full, y_full)

    proba_raw = final_clf.predict_proba(X_test)
    proba_raw = np.asarray(proba_raw, dtype=np.float32)

    if proba_raw.shape[1] != 4:
        raise RuntimeError(
            f"Expected 4 probability columns, got shape {proba_raw.shape}"
        )

    proba = np.clip(proba_raw, 1e-6, 1 - 1e-6).astype("float32")
    return proba




## === cell 5
if len(submissions_all) >= 3:
    submission_avg = ensemble(
        submissions_all, [0, 1, 2], [0.15, 0.8, 0.05], test_csv_path=test_csv_path
    )
    template_for_output = sample_sub_path
elif len(submissions_all) >= 1:
    submission_avg = ensemble(submissions_all, [0], [1.0], test_csv_path=test_csv_path)
    template_for_output = sample_sub_path
else:
    submission_avg = train_and_predict_probs(train_csv_path, test_csv_path, images_dir)
    template_for_output = sample_sub_path
    print(
        "No external submissions found; trained a simple HOG(RGB)+LogReg model for predictions."
    )

make_submission_file(submission_avg, template_for_output, test_csv_path)
