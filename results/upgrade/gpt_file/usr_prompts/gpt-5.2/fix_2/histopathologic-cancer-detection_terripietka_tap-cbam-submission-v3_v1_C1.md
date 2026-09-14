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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.7704608760998947

# 6. Current score

0.45758

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.45758) has done: 'I remove the TensorFlow/Keras dependency because it is currently failing to import in this environment and the external pretrained model file path is missing, which prevents any valid submission from being created. To keep the pipeline working end-to-end with only installed/standard libraries, I replace the model inference with a lightweight baseline that reads each `.tif` and predicts a probability based on the mean intensity of the center 32×32 region (aligned with the competition’s label definition). I also fix the data-loading logic so IDs are handled consistently (no `.tif` suffix in the submission), ensure predictions are probabilities (not hard class labels) for ROC-AUC, and guarantee that `submission.csv` is written with the required columns and row order. The plotting/inspection cells be kept but made safe so they don’t block execution.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




## === cell 1
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found: {SAMPLE_SUB_PATH}"

test_df = pd.read_csv(SAMPLE_SUB_PATH)
print("Sample submission shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
test_df.head()



## === cell 2
test_df_img = test_df.copy()
test_df_img["filename"] = test_df_img["id"].astype(str) + ".tif"

first_file = os.path.join(TEST_DIR, test_df_img.loc[0, "filename"])
print("First test image path:", first_file)
print("Exists:", os.path.exists(first_file))

print("Test Set Size:", test_df_img.shape)
test_df_img.head()



## === cell 3
try:
    sample_images = test_df_img.sample(16, random_state=1).reset_index(drop=True)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.tight_layout(pad=1.0)

    for i, ax in enumerate(axes.flat):
        fname = sample_images.loc[i, "filename"]
        img = mpimg.imread(os.path.join(TEST_DIR, fname))
        ax.imshow(img)
        ax.set_title(f"ID: {sample_images.loc[i,'id']}")
        ax.axis("off")

    plt.show()
except Exception as e:
    print("Visualization skipped due to error:", repr(e))




## === cell 4
def center_crop_mean_intensity(img: np.ndarray, crop_size: int = 32) -> float:
    """
    Compute mean intensity of center crop (crop_size x crop_size).
    Returns a float in [0, 1] approximately after normalization.
    """
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3]  # drop alpha if present
        img = img.mean(axis=-1)

    img = img.astype(np.float32)

    if img.max() > 1.5:
        img = img / 255.0

    h, w = img.shape[:2]
    ch = crop_size // 2
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - ch), min(h, cy + ch)
    x0, x1 = max(0, cx - ch), min(w, cx + ch)

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return float(np.clip(img.mean(), 0.0, 1.0))
    return float(np.clip(crop.mean(), 0.0, 1.0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))




## === cell 5
probs = np.zeros(len(test_df_img), dtype=np.float32)

bias = -0.2
scale = 6.0

missing = 0
for i, fname in enumerate(test_df_img["filename"].values):
    path = os.path.join(TEST_DIR, fname)
    if not os.path.exists(path):
        missing += 1
        probs[i] = 0.5
        continue
    img = mpimg.imread(path)
    m = center_crop_mean_intensity(img, crop_size=32)
    probs[i] = float(sigmoid(scale * (m - 0.5) + bias))

print("Done. Missing files:", missing)
print("Pred stats:", float(probs.min()), float(probs.mean()), float(probs.max()))



## === cell 6
submission = pd.DataFrame(
    {"id": test_df["id"].astype(str), "label": probs.astype(np.float32)}
)

submission["label"] = submission["label"].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()



## === cell 7
try:
    plt.figure(figsize=(6, 4))
    plt.hist(submission["label"].values, bins=30, color="steelblue", edgecolor="white")
    plt.title("Predicted probability distribution")
    plt.xlabel("label (probability)")
    plt.ylabel("count")
    plt.show()
except Exception as e:
    print("Plot skipped due to error:", repr(e))
