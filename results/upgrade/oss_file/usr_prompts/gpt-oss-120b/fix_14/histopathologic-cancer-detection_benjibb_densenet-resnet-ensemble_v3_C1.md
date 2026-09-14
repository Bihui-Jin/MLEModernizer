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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9648

# 6. Current score

0.75669

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the NumPy `full` call that caused a TypeError and ensure the fallback test‑probability array is correctly defined. Then I adjust the submission‑writing logic so it safely handles empty or mismatched prediction lengths, using a mean value or zero when needed. These fixes let the script run end‑to‑end, compute the validation AUC (expected ≈0.97), and produce a proper `.csv` submission, moving the score toward the target.'
- What this solution (achieved 0.5) has done: 'The changes add a safe fallback that uses the validation‑set ensemble probabilities when the test‑set predictions are missing, ensuring a non‑empty submission and producing AUC ≈ validation performance (≈ 0.97). The submission writer also fills missing values with the mean of the available validation probabilities instead of a constant zero, moving the score toward the target 0.9648.'
- What this solution (achieved 0.45758) has done: 'The timeout was caused by the single‑threaded loop that opens and processes every test image when no model predictions are available. I replaced that loop with a thread‑pool that loads and computes the centre‑region mean in parallel, pre‑allocating the result array and reusing the fallback mean already computed for validation. This keeps the exact same probability calculation logic while dramatically cutting I/O‑bound runtime.'
- What this solution (achieved 0.54242) has done: 'I add a lightweight calibration model that learns the relationship between the centre‑pixel intensity of an image and the true label using a sampled subset of the training data. This model then convert the simple centre‑intensity probabilities used in the fallback path into better‑calibrated probabilities, raising the AUC toward the target while keeping the original ensemble logic unchanged. The changes also import the needed sklearn class and keep all file‑handling and submission logic intact.'
- What this solution (achieved 0.54242) has done: 'I compute validation probabilities directly from the centre‑pixel logistic‑regression model (instead of relying on missing external CSVs). After fitting the model on a sampled subset of the training images, I obtain `avg_valid_prob` and the corresponding `auc_val` using those same samples. This gives a realistic AUC (well above 0.5) and a sensible fallback mean for test predictions, while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.54242) has done: 'The fix adds the missing imports (pandas, numpy, os, concurrent.futures, Pillow Image, and the scikit‑learn utilities) and restructures the notebook into a clear sequence of cells: import & helper, load ensemble CSVs, softmax conversion, validation AUC (or fallback logistic regression on centre intensity), baseline centre‑mean probability for test images, test‑set prediction aggregation, and finally safe submission writing. All variables are now defined before use, the fallback paths are handled correctly, and the script writes a proper `.csv` file with the validation AUC shown in the console.'
- What this solution (achieved 0.54693) has done: 'I enhance the fallback model while keeping the overall pipeline unchanged.  
- Add an overall‑image mean as a second feature together with the centre mean, and train the logistic‑regression on these two features (using a larger sample of the training set).  
- Update the test‑time baseline to compute both features and use the same calibrated model.  
These minimal changes give the model richer information, improving validation AUC and moving the score toward the target.'
- What this solution (achieved 0.73994) has done: 'I replace the simple two‑feature logistic‑regression fallback with a richer pixel‑level model: the centre 32×32 patch (RGB) is flattened and combined with the overall image mean as features, then a LogisticRegression is trained on a random 40 k sample of the training set. The same feature extraction is used at test time, so predictions are better calibrated and the validation AUC moves much closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.75669) has done: 'The update skips the expensive image‑feature extraction and logistic‑regression training when ensemble predictions are available, because those predictions already provide the validation AUC and test probabilities. By guarding the heavy computation with `if not has_ensemble`, we avoid reading >170 k training images and >45 k test images when they are unnecessary, keeping the core logic identical for cases where ensemble files are missing.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import concurrent.futures
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_csv(path):
    """Load a CSV if it exists; otherwise return an empty DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()




## === cell 1
dense161 = load_csv(
    "../input/cancer-densenet161-v2-for-ensemble/validation_0.976066529750824.csv"
)
dense161_test = load_csv(
    "../input/cancer-densenet161-v2-for-ensemble/submission_0.976066529750824.csv"
)

dense201 = load_csv(
    "../input/cancer-densenet201-v2-for-ensemble/validation_0.9749373197555542.csv"
)
dense201_test = load_csv(
    "../input/cancer-densenet201-v2-for-ensemble/submission_0.9749373197555542.csv"
)

res50 = load_csv(
    "../input/cancer-resnet50-v2-for-ensemble/validation_0.9727705717086792.csv"
)
res50_test = load_csv(
    "../input/cancer-resnet50-v2-for-ensemble/submission_0.9727705717086792.csv"
)




## === cell 2
def softmax_probs(df, model_name, test=False):
    """
    Convert logit columns to softmax probabilities for the positive class.
    Expected columns:
        - validation:  val_0, val_1
        - test:        pred_0, pred_1
    Returns a NumPy array of probabilities for class 1.
    """
    if test:
        logits0 = np.exp(df["pred_0"])
        logits1 = np.exp(df["pred_1"])
    else:
        logits0 = np.exp(df["val_0"])
        logits1 = np.exp(df["val_1"])
    sum_logits = logits0 + logits1
    return logits1 / sum_logits




## === cell 3
valid_probs = []
valid_labels = None

if not dense161.empty:
    valid_probs.append(softmax_probs(dense161, "dense161", test=False))
    if "ground_truth_label" in dense161.columns:
        valid_labels = dense161["ground_truth_label"].values
    else:
        valid_labels = dense161["label"].values

if not dense201.empty:
    valid_probs.append(softmax_probs(dense201, "dense201", test=False))

if not res50.empty:
    valid_probs.append(softmax_probs(res50, "res50", test=False))

has_ensemble = bool(valid_probs) and (valid_labels is not None)

if has_ensemble:
    avg_valid_prob = np.mean(np.column_stack(valid_probs), axis=1)
    auc_val = roc_auc_score(valid_labels, avg_valid_prob)
else:
    avg_valid_prob = np.array([])
    auc_val = None




## === cell 4
train_labels_path = "../input/histopathologic-cancer-detection/train_labels.csv"
if not os.path.isfile(train_labels_path):
    train_labels_path = "../input/train_labels.csv"

train_df = pd.read_csv(train_labels_path)
train_ids = train_df["id"].astype(str).tolist()
train_labels = train_df["label"].values

MAX_TRAIN = None
if MAX_TRAIN is not None and len(train_ids) > MAX_TRAIN:
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_ids), MAX_TRAIN, replace=False)
    sample_ids = [train_ids[i] for i in sample_idx]
    sample_labels = train_labels[sample_idx]
else:
    sample_ids = train_ids
    sample_labels = train_labels

if not has_ensemble:

    def compute_center_features(img_id, img_dir):
        """
        Returns a tuple (flattened_center_pixels, overall_mean).
        flattened_center_pixels is a 32*32*3 vector with values in [0,1].
        overall_mean is a single float in [0,1].
        """
        img_path = os.path.join(img_dir, f"{img_id}.tif")
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)  # shape HxWx3, uint8
            h, w = arr.shape[:2]
            half = 32 // 2
            cx, cy = w // 2, h // 2
            region = arr[cy - half : cy + half, cx - half : cx + half, :]  # 32x32x3
            centre_flat = region.astype(np.float32).reshape(-1) / 255.0  # 3072 values
            overall = arr.mean() / 255.0
            return centre_flat, overall
        except Exception:
            return np.full(32 * 32 * 3, np.nan, dtype=np.float32), np.nan

    train_dir = "../input/histopathologic-cancer-detection/train/"
    if not os.path.isdir(train_dir):
        train_dir = "../input/train/"

    feature_len = 32 * 32 * 3 + 1
    sample_features = np.empty((len(sample_ids), feature_len), dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor() as exec:
        futures = {
            exec.submit(compute_center_features, img_id, train_dir): idx
            for idx, img_id in enumerate(sample_ids)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            centre_flat, overall = fut.result()
            sample_features[idx, :-1] = centre_flat
            sample_features[idx, -1] = overall

    mask = ~np.isnan(sample_features).any(axis=1)
    sample_features = sample_features[mask]
    sample_labels = sample_labels[mask]

    if len(sample_features) > 0:
        centre_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=1, class_weight="balanced"),
        )
        centre_model.fit(sample_features, sample_labels)
    else:
        centre_model = None

    if not has_ensemble:
        if centre_model is not None and len(sample_features) > 0:
            avg_valid_prob = centre_model.predict_proba(sample_features)[:, 1]
            auc_val = roc_auc_score(sample_labels, avg_valid_prob)
        else:
            avg_valid_prob = np.array([])
            auc_val = None
else:
    centre_model = None




## === cell 5
def baseline_center_prob(test_ids, test_dir):
    """
    Compute calibrated probabilities for each test image using the
    centre‑pixel features and overall mean, through the same LogisticRegression
    model trained on the training set.
    """
    n = len(test_ids)
    probs = np.empty(n, dtype=np.float32)

    fallback = avg_valid_prob.mean() if avg_valid_prob.size > 0 else 0.0
    feature_len = 32 * 32 * 3 + 1

    def worker(idx_imgid):
        idx, img_id = idx_imgid
        img_path = os.path.join(test_dir, f"{img_id}.tif")
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            h, w = arr.shape[:2]
            half = 32 // 2
            cx, cy = w // 2, h // 2
            region = arr[cy - half : cy + half, cx - half : cx + half, :]
            centre_flat = region.astype(np.float32).reshape(-1) / 255.0
            overall = arr.mean() / 255.0
        except Exception:
            centre_flat = np.full(32 * 32 * 3, fallback, dtype=np.float32)
            overall = fallback

        if centre_model is not None:
            x = np.concatenate([centre_flat, [overall]]).reshape(1, -1)
            calibrated = centre_model.predict_proba(x)[0, 1]
        else:
            calibrated = centre_flat.mean()  # very rough fallback
        return idx, calibrated

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, prob in executor.map(worker, enumerate(test_ids)):
            probs[idx] = prob

    return probs




## === cell 6
test_probs = []
if not dense161_test.empty:
    test_probs.append(softmax_probs(dense161_test, "dense161", test=True))
if not dense201_test.empty:
    test_probs.append(softmax_probs(dense201_test, "dense201", test=True))
if not res50_test.empty:
    test_probs.append(softmax_probs(res50_test, "res50", test=True))

if test_probs:
    avg_test_prob = np.mean(np.column_stack(test_probs), axis=1)
else:
    sub_path = "../input/histopathologic-cancer-detection/sample_submission.csv"
    if not os.path.isfile(sub_path):
        sub_path = "../input/sample_submission.csv"
    submission_df = pd.read_csv(sub_path)

    test_ids = submission_df["id"].astype(str).tolist()
    test_dir = "../input/histopathologic-cancer-detection/test/"
    if not os.path.isdir(test_dir):
        test_dir = "../input/test/"

    avg_test_prob = baseline_center_prob(test_ids, test_dir)




## === cell 7
sub_path = "../input/histopathologic-cancer-detection/sample_submission.csv"
if not os.path.isfile(sub_path):
    sub_path = "../input/sample_submission.csv"
submission = pd.read_csv(sub_path)

if avg_test_prob.size == len(submission):
    submission["label"] = avg_test_prob
else:
    if avg_test_prob.size > 0:
        fill_val = avg_test_prob.mean()
    elif avg_valid_prob.size > 0:
        fill_val = avg_valid_prob.mean()
    else:
        fill_val = 0.0
    submission["label"] = fill_val

out_name = f"submission_{auc_val:.6f}.csv" if auc_val is not None else "submission.csv"
submission.to_csv(out_name, index=False, header=True)
print(f"Saved submission to {out_name}")
if auc_val is not None:
    print(f"Validation AUC: {auc_val:.6f}")
