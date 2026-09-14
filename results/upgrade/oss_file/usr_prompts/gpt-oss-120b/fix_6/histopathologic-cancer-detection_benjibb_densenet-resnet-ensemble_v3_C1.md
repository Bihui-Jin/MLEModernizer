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

0.45758

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the NumPy `full` call that caused a TypeError and ensure the fallback test‑probability array is correctly defined. Then I adjust the submission‑writing logic so it safely handles empty or mismatched prediction lengths, using a mean value or zero when needed. These fixes let the script run end‑to‑end, compute the validation AUC (expected ≈0.97), and produce a proper `.csv` submission, moving the score toward the target.'
- What this solution (achieved 0.5) has done: 'The changes add a safe fallback that uses the validation‑set ensemble probabilities when the test‑set predictions are missing, ensuring a non‑empty submission and producing AUC ≈ validation performance (≈ 0.97). The submission writer also fills missing values with the mean of the available validation probabilities instead of a constant zero, moving the score toward the target 0.9648.'
- What this solution (achieved 0.45758) has done: 'The timeout was caused by the single‑threaded loop that opens and processes every test image when no model predictions are available. I replaced that loop with a thread‑pool that loads and computes the centre‑region mean in parallel, pre‑allocating the result array and reusing the fallback mean already computed for validation. This keeps the exact same probability calculation logic while dramatically cutting I/O‑bound runtime.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from PIL import Image
import concurrent.futures  # added for parallel image processing




## === cell 1
def load_csv(path):
    """Load a CSV if it exists; otherwise return an empty DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


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

if valid_probs and valid_labels is not None:
    avg_valid_prob = np.mean(np.column_stack(valid_probs), axis=1)
    auc_val = roc_auc_score(valid_labels, avg_valid_prob)
else:
    avg_valid_prob = np.array([])
    auc_val = None




## === cell 4
def baseline_center_prob(test_ids, test_dir):
    """
    Compute a simple probability for each test image using parallel I/O:
    - Load the image.
    - Extract the centre 32×32 region.
    - Compute the mean pixel intensity across all RGB channels.
    - Normalise by 255 to obtain a probability in [0,1].
    Returns a NumPy array aligned with ``test_ids``.
    """
    n = len(test_ids)
    probs = np.empty(n, dtype=np.float32)

    fallback = avg_valid_prob.mean() if avg_valid_prob.size > 0 else 0.0
    half = 32 // 2  # =16

    def worker(idx_imgid):
        idx, img_id = idx_imgid
        img_path = os.path.join(test_dir, f"{img_id}.tif")
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            h, w = arr.shape[:2]
            cx, cy = w // 2, h // 2
            region = arr[cy - half : cy + half, cx - half : cx + half]
            prob = region.mean() / 255.0
        except Exception:
            prob = fallback
        return idx, prob

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, prob in executor.map(worker, enumerate(test_ids)):
            probs[idx] = prob

    return probs




## === cell 5
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




## === cell 6
sub_path = "../input/histopathologic-cancer-detection/sample_submission.csv"
if not os.path.isfile(sub_path):
    sub_path = "../input/sample_submission.csv"  # fallback location
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
