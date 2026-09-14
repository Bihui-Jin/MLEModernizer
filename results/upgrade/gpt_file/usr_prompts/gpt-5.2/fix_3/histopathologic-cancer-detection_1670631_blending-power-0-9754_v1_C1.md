# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE = "/kaggle/input/histopathologic-cancer-detection"
if not os.path.exists(BASE):
    BASE = "/kaggle/data/histopathologic-cancer-detection"

sample_path = os.path.join(BASE, "sample_submission.csv")
test_dir = os.path.join(BASE, "test")

sub_template = pd.read_csv(sample_path)
sub_template = sub_template[["id", "label"]].copy()

df2 = sub_template.copy()




## === cell 1
def _read_center_crop_rgb(path, half=16):
    """
    Read only the center (2*half x 2*half) crop in RGB.
    Falls back to full read+crop if ROI read isn't available.
    """
    try:
        import cv2  # type: ignore

        im0 = cv2.imread(path, cv2.IMREAD_COLOR)
        if im0 is None:
            raise ValueError("cv2.imread returned None")
        h, w = im0.shape[:2]
        cy, cx = h // 2, w // 2
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        crop_bgr = im0[y0:y1, x0:x1, :]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return crop_rgb
    except Exception:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            w, h = img.size
            cx, cy = w // 2, h // 2
            left, upper = cx - half, cy - half
            right, lower = cx + half, cy + half
            return np.array(img.convert("RGB").crop((left, upper, right, lower)))


def center_brightness_score_from_crop(crop_rgb):
    gray = (
        0.2989 * crop_rgb[..., 0]
        + 0.5870 * crop_rgb[..., 1]
        + 0.1140 * crop_rgb[..., 2]
    ).astype(np.float32)

    mean_brightness = float(gray.mean()) / 255.0
    score = 1.0 - mean_brightness  # higher => darker => more likely tumor
    score = 1.0 / (1.0 + np.exp(-(score - 0.5) * 6.0))
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)




## === cell 2
ids = sub_template["id"].astype(str).tolist()
n = len(ids)
preds = np.zeros(n, dtype=np.float32)

try:
    available = set(os.listdir(test_dir))
except Exception:
    available = None  # fallback to exists

missing = 0

_join = os.path.join
_test_dir = test_dir
_available = available
_read_crop = _read_center_crop_rgb
_score_crop = center_brightness_score_from_crop

for i, img_id in enumerate(ids):
    fname = f"{img_id}.tif"
    if _available is not None:
        if fname not in _available:
            missing += 1
            preds[i] = 0.0
            continue
        img_path = _join(_test_dir, fname)
    else:
        img_path = _join(_test_dir, fname)
        if not os.path.exists(img_path):
            missing += 1
            preds[i] = 0.0
            continue

    crop = _read_crop(img_path, half=16)
    preds[i] = _score_crop(crop)

df1 = pd.DataFrame({"id": ids, "label": preds})

df1 = df2[["id"]].merge(df1, on="id", how="left")
df1["label"] = df1["label"].fillna(0.0).astype(float)

df1["label"] = 0.8 * df1["label"].values + 0.2 * df2["label"].astype(float).values

assert df1.shape[0] == df2.shape[0]
assert list(df1.columns) == ["id", "label"]
df1["label"] = df1["label"].clip(0.0, 1.0)

df1.head()



## === cell 3
df1.to_csv("sub.csv", index=False)
print("Wrote sub.csv with shape:", df1.shape)
print("Missing test images (if any):", missing)
print(df1.head())
