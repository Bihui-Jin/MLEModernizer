# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

FALLBACK_SAMPLE_SUB_1 = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
FALLBACK_SAMPLE_SUB_2 = "/kaggle/data/sample_submission.csv"



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission candidates:", submissions_all)



## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Submission file {path} missing columns: {missing}. Has: {list(df.columns)}"
        )
    return df[["image_id"] + TARGET_COLS]


def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average over submission files by index in submissions_all.
    Returns numpy array of shape (n_rows, 4) aligned to the first selected submission's row order.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx length ({len(sub_idx)}) must equal weights length ({len(weights)})"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    if any((i < 0 or i >= len(submissions_all)) for i in sub_idx):
        raise IndexError(
            f"Requested indices {sub_idx} out of range for {len(submissions_all)} files."
        )

    submission_with_weight = []
    base_ids = None

    for i, idx in enumerate(sub_idx):
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = _load_submission(path)
        if base_ids is None:
            base_ids = df["image_id"].values
        else:
            if not (df["image_id"].values == base_ids).all():
                df = df.set_index("image_id").loc[base_ids].reset_index()

        submission_with_weight.append(df[TARGET_COLS].values * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_submission_path):
    """
    Writes submission.csv with correct columns and row count, using template_submission_path for image_id order.
    """
    submission_df = _load_submission(template_submission_path)

    submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
def _pick_fallback_sample():
    if os.path.exists(FALLBACK_SAMPLE_SUB_1):
        return FALLBACK_SAMPLE_SUB_1
    if os.path.exists(FALLBACK_SAMPLE_SUB_2):
        return FALLBACK_SAMPLE_SUB_2
    raise FileNotFoundError("Could not find any sample_submission.csv fallback path.")


def _find_dataset_root():
    candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for root in candidates:
        train_csv = os.path.join(root, "train.csv")
        test_csv = os.path.join(root, "test.csv")
        images_dir = os.path.join(root, "images")
        if (
            os.path.exists(train_csv)
            and os.path.exists(test_csv)
            and os.path.isdir(images_dir)
        ):
            return root
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/images directory under expected Kaggle paths."
    )


def _load_image_as_feature_vector(image_path, size=(96, 96)):
    """
    Core logic preserved: low-res pixels + simple stats.
    - Per-image standardize the pixel block so illumination/background differences don't dominate.
    - Keep appended stats (means/std/min/max) to preserve absolute color/brightness information.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)

    flat = arr.reshape(-1).astype(np.float32)

    m = float(flat.mean())
    s = float(flat.std())
    if s < 1e-6:
        s = 1e-6
    flat_std = (flat - m) / s

    ch = arr.reshape(-1, 3)
    stats = np.concatenate(
        [ch.mean(axis=0), ch.std(axis=0), ch.min(axis=0), ch.max(axis=0)], axis=0
    ).astype(
        np.float32
    )  # 12 dims

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    gstats = np.array(
        [gray.mean(), gray.std(), gray.min(), gray.max()], dtype=np.float32
    )  # 4 dims

    return np.concatenate([flat_std, stats, gstats], axis=0).astype(np.float32)


def _build_features(image_ids, images_dir, size=(96, 96)):
    import numpy as np

    feat_dim = size[0] * size[1] * 3 + 12 + 4
    X = np.zeros((len(image_ids), feat_dim), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(image_ids):
        p = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(p):
            missing += 1
            continue
        X[i] = _load_image_as_feature_vector(p, size=size)
    if missing:
        print(
            f"Warning: {missing} images not found in {images_dir}; using zero features for them."
        )
    return X


def _train_and_predict_fallback_submission():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    root = _find_dataset_root()
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    images_dir = os.path.join(root, "images")
    sample_sub_path = os.path.join(root, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        sample_sub_path = _pick_fallback_sample()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train_df.columns and c != "image_id":
            raise ValueError(f"train.csv missing required target column: {c}")
    if "image_id" not in test_df.columns:
        raise ValueError("test.csv missing image_id column")

    X_train_96 = _build_features(
        train_df["image_id"].tolist(), images_dir, size=(96, 96)
    )
    X_test_96 = _build_features(test_df["image_id"].tolist(), images_dir, size=(96, 96))
    X_train_64 = _build_features(
        train_df["image_id"].tolist(), images_dir, size=(64, 64)
    )
    X_test_64 = _build_features(test_df["image_id"].tolist(), images_dir, size=(64, 64))

    def _split_pix_and_stats(X, size):
        pix_dim = size[0] * size[1] * 3
        stats_dim = 12 + 4
        pix = X[:, :pix_dim]
        stats = X[:, pix_dim : pix_dim + stats_dim]
        return pix, stats

    pix96_tr, st96_tr = _split_pix_and_stats(X_train_96, (96, 96))
    pix96_te, st96_te = _split_pix_and_stats(X_test_96, (96, 96))
    pix64_tr, st64_tr = _split_pix_and_stats(X_train_64, (64, 64))
    pix64_te, st64_te = _split_pix_and_stats(X_test_64, (64, 64))

    scaler96 = StandardScaler(with_mean=True, with_std=True)
    scaler64 = StandardScaler(with_mean=True, with_std=True)
    st96_tr_s = scaler96.fit_transform(st96_tr)
    st96_te_s = scaler96.transform(st96_te)
    st64_tr_s = scaler64.fit_transform(st64_tr)
    st64_te_s = scaler64.transform(st64_te)

    pix_tr = np.concatenate([pix96_tr, pix64_tr], axis=1).astype(np.float32)
    pix_te = np.concatenate([pix96_te, pix64_te], axis=1).astype(np.float32)

    def _per_row_standardize(A):
        m = A.mean(axis=1, keepdims=True)
        s = A.std(axis=1, keepdims=True)
        s = np.maximum(s, 1e-6)
        return (A - m) / s

    pix_tr = _per_row_standardize(pix_tr).astype(np.float32)
    pix_te = _per_row_standardize(pix_te).astype(np.float32)

    X_train = np.concatenate([pix_tr, st96_tr_s, st64_tr_s], axis=1).astype(np.float32)
    X_test = np.concatenate([pix_te, st96_te_s, st64_te_s], axis=1).astype(np.float32)

    y_onehot = train_df[TARGET_COLS].astype(int).values
    row_sums = y_onehot.sum(axis=1)
    if not np.all((row_sums == 1) | (row_sums == 0)):
        print(
            "Warning: Found rows with multiple positive labels; using argmax as class label."
        )
    y_class = np.argmax(y_onehot, axis=1).astype(int)
    y_class = np.where(row_sums == 0, 1, y_class).astype(int)

    clf = LogisticRegression(
        max_iter=2500,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        C=3.0,
        random_state=0,
    )
    clf.fit(X_train, y_class)

    proba = clf.predict_proba(X_test).astype(np.float32)  # (n_test, n_classes_seen)

    classes_ = clf.classes_.astype(int).tolist()
    full_proba = np.zeros((proba.shape[0], 4), dtype=np.float32)
    for j, cls_id in enumerate(classes_):
        if 0 <= cls_id < 4:
            full_proba[:, cls_id] = proba[:, j]

    alpha = 0.03
    full_proba = (1.0 - alpha) * full_proba + alpha * 0.25

    template_df = _load_submission(sample_sub_path)

    pred_df = pd.DataFrame(full_proba, columns=TARGET_COLS)
    pred_df.insert(0, "image_id", test_df["image_id"].values)
    pred_df = (
        pred_df.set_index("image_id").loc[template_df["image_id"].values].reset_index()
    )

    return (
        pred_df[["image_id"] + TARGET_COLS].values[:, 1:].astype(np.float32),
        sample_sub_path,
    )


if len(submissions_all) >= 4:
    submission_avg = ensemble(submissions_all, [1, 2, 3], [0.3, 0.3, 0.4])
    template_path = submissions_all[1]  # template aligned with ensemble base ids
    make_submission_file(submission_avg, template_path)
elif len(submissions_all) > 0:
    sub_idx = list(range(len(submissions_all)))
    weights = [1.0 / len(sub_idx)] * len(sub_idx)
    submission_avg = ensemble(submissions_all, sub_idx, weights)
    template_path = submissions_all[0]
    make_submission_file(submission_avg, template_path)
else:
    submission_avg, template_path = _train_and_predict_fallback_submission()
    make_submission_file(submission_avg, template_path)
