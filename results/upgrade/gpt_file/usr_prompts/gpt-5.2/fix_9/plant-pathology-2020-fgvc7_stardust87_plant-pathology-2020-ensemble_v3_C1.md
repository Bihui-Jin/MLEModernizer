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
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

print("SUBMISSIONS_PATH exists:", os.path.exists(SUBMISSIONS_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TEST_CSV_PATH exists:", os.path.exists(TEST_CSV_PATH))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("IMAGES_DIR exists:", os.path.exists(IMAGES_DIR))



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print("Found submission files:", len(submissions_all))
print(submissions_all[:10])




## === cell 3
def _make_fallback_submissions(
    sample_sub_path: str,
    train_csv_path: str,
    test_csv_path: str,
    images_dir: str,
    out_dir: str = ".",
):
    """
    Score-relevant minimal changes vs previous version (core logic preserved):
    - Still: fixed image features (PIL+NumPy) -> StandardScaler -> OneVsRest(LogisticRegression) -> predict_proba
      and create 2 submissions (orig and hflip-TTA) for later weighted CSV ensembling.
    - Improve feature signal minimally (still fast, still PIL+NumPy): add a second grayscale downsample resolution
      + simple gradient-magnitude summary stats; this typically improves separability and thus ROC AUC.
    - Use a stratified train/val split for sanity-check AUC (does not affect final submission), avoiding misleading
      splits that can hide training issues.
    - Use solver='saga' (stable with many features) and slightly tuned C/max_iter to reduce underfitting that can
      keep AUC near 0.5.
    """
    req_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]

    sample = pd.read_csv(sample_sub_path)
    missing = [c for c in req_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission is missing columns: {missing}")

    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)

    test_ids = test_df["image_id"].astype(str).tolist()

    sub_uniform = pd.DataFrame({"image_id": test_ids})
    for c in req_cols[1:]:
        sub_uniform[c] = 0.25
    path_uniform = os.path.join(out_dir, "fallback_uniform.csv")
    sub_uniform.to_csv(path_uniform, index=False)

    import numpy as np
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    def _img_path_from_id(image_id: str) -> str:
        if image_id.lower().endswith(".jpg"):
            return os.path.join(images_dir, image_id)
        return os.path.join(images_dir, f"{image_id}.jpg")

    def _grad_mag_stats(gray_arr_f32: np.ndarray):
        gx = np.diff(gray_arr_f32, axis=1, append=gray_arr_f32[:, -1:])
        gy = np.diff(gray_arr_f32, axis=0, append=gray_arr_f32[-1:, :])
        mag = np.sqrt(gx * gx + gy * gy)
        return np.array(
            [mag.mean(), mag.std(), np.quantile(mag, 0.5), np.quantile(mag, 0.9)],
            dtype=np.float32,
        )

    def _extract_features_from_pil(
        im,
        rgb_size=(128, 128),
        gray_small=(32, 32),
        gray_med=(64, 64),
        bins=32,
    ) -> np.ndarray:
        im_rgb = im.convert("RGB").resize(rgb_size)
        arr = np.asarray(im_rgb, dtype=np.uint8)

        feats = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
            h = h.astype(np.float32)
            h /= h.sum() + 1e-8
            feats.append(h)

        flat = arr.reshape(-1, 3).astype(np.float32) / 255.0
        feats.append(flat.mean(axis=0).astype(np.float32))
        feats.append(flat.std(axis=0).astype(np.float32))

        im_g1 = im.convert("L").resize(gray_small)
        g1 = np.asarray(im_g1, dtype=np.float32) / 255.0
        feats.append(g1.reshape(-1).astype(np.float32))
        feats.append(_grad_mag_stats(g1))

        im_g2 = im.convert("L").resize(gray_med)
        g2 = np.asarray(im_g2, dtype=np.float32) / 255.0
        feats.append(g2.reshape(-1).astype(np.float32))
        feats.append(_grad_mag_stats(g2))

        return np.concatenate(feats, axis=0)

    def _extract_features(image_id: str, flip: bool = False):
        p = _img_path_from_id(image_id)
        try:
            im = Image.open(p)
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            return _extract_features_from_pil(im)
        except Exception:
            return None

    _zero_feat = None
    for _id in train_df["image_id"].astype(str).tolist()[:20] + test_ids[:20]:
        tmp = _extract_features(_id, flip=False)
        if tmp is not None:
            _zero_feat = np.zeros_like(tmp, dtype=np.float32)
            break
    if _zero_feat is None:
        raise RuntimeError("Could not infer feature vector size from available images.")

    def _safe_extract(image_id: str, flip: bool = False) -> np.ndarray:
        v = _extract_features(image_id, flip=flip)
        return v if v is not None else _zero_feat

    train_ids = train_df["image_id"].astype(str).tolist()
    y = train_df[req_cols[1:]].astype(int).values

    X = np.vstack([_safe_extract(i, flip=False) for i in train_ids])

    strat = y.argmax(axis=1)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    base = LogisticRegression(
        solver="saga",
        C=4.0,
        max_iter=4000,
        n_jobs=1,
        random_state=42,
        class_weight="balanced",
    )
    clf = OneVsRestClassifier(
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False, with_std=True)),
                ("lr", base),
            ]
        )
    )
    clf.fit(X_tr, y_tr)

    try:
        va_proba = clf.predict_proba(X_va)
        aucs = []
        for j in range(y.shape[1]):
            aucs.append(roc_auc_score(y_va[:, j], va_proba[:, j]))
        print("Local validation mean ROC AUC (sanity only):", float(np.mean(aucs)))
        print("Per-class AUC:", [float(a) for a in aucs])
    except Exception as e:
        print("Validation AUC computation skipped due to:", repr(e))

    clf.fit(X, y)

    X_test = np.vstack([_safe_extract(i, flip=False) for i in test_ids])
    X_test_flip = np.vstack([_safe_extract(i, flip=True) for i in test_ids])

    proba = clf.predict_proba(X_test)
    proba_flip = clf.predict_proba(X_test_flip)
    proba_tta = 0.5 * (proba + proba_flip)

    sub_tta = pd.DataFrame({"image_id": test_ids})
    for j, c in enumerate(req_cols[1:]):
        sub_tta[c] = proba_tta[:, j]
    path_tta = os.path.join(out_dir, "fallback_img_pil_lr_tta.csv")
    sub_tta.to_csv(path_tta, index=False)

    sub_orig = pd.DataFrame({"image_id": test_ids})
    for j, c in enumerate(req_cols[1:]):
        sub_orig[c] = proba[:, j]
    path_orig = os.path.join(out_dir, "fallback_img_pil_lr_orig.csv")
    sub_orig.to_csv(path_orig, index=False)

    return [path_tta, path_orig, path_uniform]


if len(submissions_all) == 0:
    print(
        "No external submissions found; creating fallback submissions (PIL+NumPy image features LR, with/without TTA + uniform)."
    )
    submissions_all = _make_fallback_submissions(
        SAMPLE_SUB_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, IMAGES_DIR, out_dir="."
    )
    print("Fallback submissions:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("submissions_all is empty; cannot ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) == 0:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})"
        )

    test_df = pd.read_csv(TEST_CSV_PATH)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if not (0 <= idx < len(submissions_all)):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all of length {len(submissions_all)}"
            )

        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        submission = pd.read_csv(path)
        missing = [c for c in (["image_id"] + cols) if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission file {path} missing columns: {missing}")

        submission["image_id"] = submission["image_id"].astype(str)
        test_df["image_id"] = test_df["image_id"].astype(str)

        submission = test_df[["image_id"]].merge(submission, on="image_id", how="left")
        if submission[cols].isna().any().any():
            submission[cols] = submission[cols].fillna(0.25)

        mat = submission.loc[:, cols].values
        submission_with_weight.append(mat * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    test_df = pd.read_csv(TEST_CSV_PATH)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]

    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    submission_df["image_id"] = submission_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)

    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right"
    )
    if submission_df.shape[0] != test_df.shape[0]:
        raise ValueError(
            "Submission row count does not match test row count after alignment."
        )

    import numpy as np

    if (
        not isinstance(submission_avg, np.ndarray)
        or submission_avg.shape[0] != submission_df.shape[0]
        or submission_avg.shape[1] != 4
    ):
        raise ValueError(
            f"submission_avg has shape {getattr(submission_avg, 'shape', None)}, expected ({submission_df.shape[0]}, 4)"
        )

    submission_df[cols] = submission_avg
    submission_df[cols] = submission_df[cols].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.7, 0.3])
else:
    submission_avg = ensemble(submissions_all, [0], [1.0])

make_submission_file(submission_avg, submissions_all)
