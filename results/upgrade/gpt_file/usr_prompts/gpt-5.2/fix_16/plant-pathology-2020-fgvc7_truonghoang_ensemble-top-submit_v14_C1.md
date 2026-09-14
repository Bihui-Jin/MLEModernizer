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

0.9704488894833836

# 6. Current score

0.64199

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because it tries to read out-of-environment “../input/plantpathology/…” submission files that don’t exist here, so `dsub` never gets defined and the blend loop crashes. I keep the same core approach (simple averaging/blending into the sample_submission template with the same alpha smoothing), but replace the missing external submission inputs with a minimal, deterministic baseline built from `train.csv` label priors (class prevalence). This guarantees the notebook runs end-to-end, writes a valid `submission.csv` with the exact required columns/order, and yields a reasonable non-error score without changing evaluation semantics.'
- What this solution (achieved 0.65586) has done: 'Main bottlenecks are repeated per-image PIL decoding + multiple full-image array reshapes/reductions, and very slow blending via `map(...to_dict(orient="index"))` + DataFrame construction. I keep the exact same feature definitions and model, but make feature extraction single-pass (no `flat`, no extra intermediate arrays) and parallelize it safely with a thread pool (PIL releases the GIL during decode; NumPy reductions run in C). I also replace the blending step with an equivalent indexed join/NumPy accumulation that avoids Python dicts/row-wise mapping while producing identical values (up to negligible float differences). Paths, targets, model, and training semantics remain unchanged.'
- What this solution (achieved 0.69977) has done: 'We need to move the score up from 0.65586 toward 0.97045 (higher-is-better), and the gap is large (>30%), so we can make a modest but meaningful upgrade while keeping the overall approach (handcrafted image features + sklearn classifier + same submission build/blend). The biggest issue is model undercapacity for non-linear decision boundaries with the current tiny 10-feature vector; switching the classifier inside the existing OneVsRest/StandardScaler pipeline from LogisticRegression to an RBF-kernel SVC with calibrated probabilities typically gives a large AUC boost on small tabular feature sets without changing the feature extraction or training loop structure. To preserve stability and runtime, I keep the exact feature extraction and the same pipeline semantics (fit once on all train, predict_proba on test), only changing the estimator. I also keep the same blending and alpha-smoothing logic and ensure the submission columns/order exactly match sample_submission.'
- What this solution (achieved 0.71088) has done: 'Your current score (0.69977) is far below the target (0.97045), so we need a meaningful but still “same-core-logic” improvement: keep the handcrafted 10-D feature vector and the same sklearn pipeline structure, but make the classifier better calibrated and less brittle. The smallest high-impact change here is to replace `SVC(probability=True)` (which uses an internal Platt scaling that is often weak/unstable in multilabel OvR) with an explicitly calibrated SVM via `CalibratedClassifierCV`, keeping the same RBF SVC base estimator and training semantics (fit once, predict_proba once). I also add deterministic `random_state` where applicable and keep your blending/smoothing unchanged so evaluation semantics remain the same. This should move ROC AUC upward toward the target without changing the overall approach.'
- What this solution (achieved 0.6432) has done: 'Your score (0.71088) is far below the target (0.97045), so we need a meaningful uplift while keeping the same core approach (handcrafted 10-D features → StandardScaler → OvR classifier → probability predictions → same blending/smoothing/submission). The smallest high-impact, still-in-family change is to switch the calibrated SVM base estimator from RBF to linear: with such a tiny feature vector, an RBF SVM can overfit and then calibrate poorly, while a linear SVM often generalizes better and calibrates more reliably for ROC AUC. I’m also setting `class_weight="balanced"` on the SVM to reduce bias toward majority labels (important for multilabel AUC), without changing the pipeline/training semantics. Everything else (feature extraction, pipeline structure, calibration, blending, alpha smoothing, file paths, and submission schema) remains the same and still writes `submission.csv`.'
- What this solution (achieved 0.6432) has done: 'Your current score (0.6432) is far below the target (0.97045), so we need a meaningful uplift while keeping your core pipeline intact (same 10-D handcrafted features → StandardScaler → OvR calibrated linear SVM → same blending/smoothing/submission). The biggest low-risk gain is to stop training on the full train set without validation: instead, generate out-of-fold (OOF) probabilities for the train set and use them to learn a per-class temperature (logit scaling) that optimizes mean ROC AUC; then apply that scaling to test probabilities. This keeps the same model family and training semantics (no early stopping, no sampling) but improves probability calibration, which ROC AUC is sensitive to when models are miscalibrated. I also add deterministic seeds and a tiny positive floor before logit to avoid numerical issues, while leaving paths and submission schema unchanged.'
- What this solution (achieved 0.6432) has done: 'Your current score (0.6432) is far below the target (0.97045), so we should improve rather than degrade. The biggest issue is that you’re applying **two** probability calibrations/transformations (CalibratedClassifierCV sigmoid + a second per-class “temperature” search on OOF), which can easily hurt ROC AUC by distorting ranking; the minimal high-impact fix is to remove the second temperature step and use the calibrated probabilities directly. I also fix the stratification logic (argmax on multilabel can be misleading) by removing the now-unneeded OOF block entirely, keeping the same features, same linear SVM + calibration, same blending and smoothing, and still writing a valid `submission.csv`. This is a small change to post-processing only and is the most likely to move AUC upward toward the target without changing the core model/feature pipeline.'
- What this solution (achieved 0.69182) has done: 'We need to move the score up (0.6432 → target 0.97045), so the smallest meaningful change is to improve generalization/calibration without changing your feature extraction or overall sklearn pipeline structure. The current linear SVM + sigmoid calibration is likely underfitting on the tiny 10-D features; switching back to an RBF SVM and using isotonic calibration (still within the same “SVM → CalibratedClassifierCV → OvR → predict_proba” core logic) typically improves ROC AUC rankings. To keep runtime safe under 600s, I cap the RBF SVM complexity (moderate C/gamma) and keep cv=5 as you already do. Submission building/blending stays identical so evaluation semantics remain unchanged.'
- What this solution (achieved 0.69749) has done: 'I fix the runtime error caused by passing a generator (`cv.split(...)`) into `CalibratedClassifierCV`, which cannot be pickled/cloned inside `OneVsRestClassifier`. The minimal safe fix is to pass an integer `cv=5` (and keep the same `StratifiedKFold` settings for stratification separately) so sklearn can internally create splits without storing a generator. I also keep the rest of the pipeline (features → scaler → OvR calibrated RBF SVC → predict_proba → blending/smoothing → submission.csv) unchanged, ensuring `proba_test`, `dsub`, and the final `submission.csv` are produced end-to-end.'
- What this solution (achieved 0.69264) has done: 'Your score (0.69749) is far below the target (0.97045), so we should improve it with minimal, low-risk changes while keeping your handcrafted 10-D features and the same “StandardScaler → OvR calibrated SVM → predict_proba → blend/smooth” semantics. The biggest likely issue is poor probability calibration and ranking from sigmoid calibration on a small dataset; switching calibration to isotonic (still CalibratedClassifierCV) often improves ROC AUC without changing the model family. To make isotonic reliable and deterministic, I also use an explicit StratifiedKFold object (not a generator) for `cv`, and slightly increase `C` to reduce underfitting while keeping the same RBF SVC and pipeline structure. Submission generation, blending, alpha-smoothing, paths, and column order remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.69485) has done: 'We need to move your score up (0.69264 → target 0.97045), so the smallest likely gain without changing your core approach is to fix label handling: this dataset is single-label (exactly one of the four columns is 1), but you currently train a multilabel OvR setup, which can hurt ranking/AUC. I keep the exact same handcrafted 10-D features, scaler, SVM, calibration, and predict_proba workflow, but switch to a single multiclass calibrated classifier and then map the multiclass probabilities back into the 4 submission columns. I also keep your blending list, alpha-smoothing, paths, and output schema identical so it still writes a valid `submission.csv`. This should materially improve ROC AUC while staying within the same modeling family and training semantics.'
- What this solution (achieved 0.64199) has done: 'Your current score (0.69485) is far below the target (0.97045), so we should improve calibration/ranking while keeping the same handcrafted 10-D features and the same SVM+calibration pipeline. The smallest high-impact fix is to use a **multiclass probability model directly** (instead of calibrating an SVC and then mapping), by switching to `LogisticRegression(multi_class="multinomial")` inside the same `StandardScaler` pipeline; this preserves the “tabular features → scaler → classifier → predict_proba” core logic and usually yields much better ROC AUC on this dataset. We keep the same deterministic setup, the same blending list (still one model), and the same alpha-smoothing/submission schema. This should move AUC substantially upward toward the target without changing feature extraction or the submission semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "../kaggle/data",
]


def _find_file(filename: str):
    for root in CANDIDATE_ROOTS:
        path = os.path.join(root, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(root, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    return None


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

missing = [
    name
    for name, p in [
        ("train.csv", train_path),
        ("test.csv", test_path),
        ("sample_submission.csv", sample_sub_path),
    ]
    if p is None
]
if missing:
    raise FileNotFoundError(
        f"Could not locate required files: {missing}. Searched roots: {CANDIDATE_ROOTS}"
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

TARGETS = ["healthy", "multiple_diseases", "rust", "scab"]
for c in ["image_id"] + TARGETS:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv missing required column: {c}")

np.random.seed(0)



## === cell 1
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression


def _iter_existing_image_dirs():
    seen = set()
    for root in CANDIDATE_ROOTS:
        for d in (
            os.path.join(root, "images"),
            os.path.join(root, "plant-pathology-2020-fgvc7", "images"),
        ):
            if d in seen:
                continue
            seen.add(d)
            if os.path.isdir(d):
                yield d


def _build_image_path_index():
    idx = {}
    for img_dir in _iter_existing_image_dirs():
        try:
            for fn in os.listdir(img_dir):
                if not fn.endswith(".jpg"):
                    continue
                iid = fn[:-4]
                if iid not in idx:
                    idx[iid] = os.path.join(img_dir, fn)
        except FileNotFoundError:
            pass
    return idx


_IMAGE_PATH_INDEX = _build_image_path_index()


def _find_image_path(image_id: str):
    return _IMAGE_PATH_INDEX.get(image_id, None)


def _extract_features(image_path: str):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr_u8 = np.asarray(img)  # uint8, (H,W,3)

    arr = arr_u8.astype(np.float32) * (1.0 / 255.0)  # float32, (H,W,3)

    ch_mean = arr.mean(axis=(0, 1))
    ch_std = arr.std(axis=(0, 1))

    gray = arr.mean(axis=2)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())

    r, g, b = float(ch_mean[0]), float(ch_mean[1]), float(ch_mean[2])
    ngrdi = float((g - r) / (g + r + 1e-6))

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = float((mx - mn).mean())

    return np.array(
        [
            float(ch_mean[0]),
            float(ch_mean[1]),
            float(ch_mean[2]),
            float(ch_std[0]),
            float(ch_std[1]),
            float(ch_std[2]),
            gray_mean,
            gray_std,
            ngrdi,
            sat,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(image_ids):
    from concurrent.futures import ThreadPoolExecutor

    X = np.zeros((len(image_ids), 10), dtype=np.float32)
    paths = []
    missing = []

    for iid in image_ids:
        p = _find_image_path(iid)
        if p is None:
            missing.append(iid)
        paths.append(p)

    if missing:
        print(
            f"Warning: {len(missing)} images not found; features set to 0 for those ids. Example:",
            missing[:5],
        )

    max_workers = min(32, (os.cpu_count() or 4) * 2)

    def _job(i, p):
        return i, _extract_features(p)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_job, i, p) for i, p in enumerate(paths) if p is not None]
        for fut in futures:
            i, feat = fut.result()
            X[i, :] = feat

    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

y_train_mc = train_df[TARGETS].values.argmax(axis=1).astype(int)

lr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=2.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=0,
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", lr),
    ]
)

clf.fit(X_train, y_train_mc)

proba_mc = clf.predict_proba(X_test).astype(np.float64)  # (n_test, n_classes)
classes_seen = clf.named_steps["clf"].classes_.astype(int)

proba_test = np.zeros((len(test_df), len(TARGETS)), dtype=np.float64)
proba_test[:, classes_seen] = proba_mc



## === cell 2
d = pd.DataFrame({"image_id": test_df["image_id"].values})
for j, t in enumerate(TARGETS):
    d[t] = proba_test[:, j]

dsub = [d]
n = len(dsub)
print("Built", n, "model(s) for blending. Test preds shape:", d.shape)



## === cell 3
sub = sample_sub.copy()
sub = sub.merge(test_df[["image_id"]], on="image_id", how="right", sort=False)

for t in TARGETS:
    sub[t] = 0.0

sub_index = sub.set_index("image_id")
acc = np.zeros((len(sub_index), len(TARGETS)), dtype=np.float64)

for dm in dsub:
    dm2 = dm.set_index("image_id")[TARGETS]
    aligned = dm2.reindex(sub_index.index)
    acc += np.nan_to_num(aligned.to_numpy(dtype=np.float64), nan=0.0)

acc /= float(n)
alpha = 0.01
acc = (1.0 - alpha) * acc + (alpha / 4.0)
acc = np.clip(acc, 0.0, 1.0)

sub_index.loc[:, TARGETS] = acc
sub = sub_index.reset_index()[["image_id"] + TARGETS]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
