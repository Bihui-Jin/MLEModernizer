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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

INPUT_DIR = "/kaggle/input/histopathologic-cancer-detection"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_CSV = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing sample submission: {SAMPLE_SUB_CSV}"



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print("Train labels:", train_df.shape, train_df.columns.tolist())
train_df.head()



## === cell 2
test_images = TEST_DIR

test_df = pd.read_csv(SAMPLE_SUB_CSV)
print("Sample submission:", test_df.shape, test_df.columns.tolist())

test_df["filename"] = test_df["id"].astype(str) + ".tif"

print("Test Set Size:", test_df.shape)
test_df.head()




## === cell 3
def center_crop(arr, crop=32):
    h, w = arr.shape[:2]
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    return arr[y0 : y0 + crop, x0 : x0 + crop]


def rgb_to_gray(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def image_feature(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
    c = center_crop(arr, 32)
    gray = rgb_to_gray(c)

    ch_std = np.std(c, axis=2).mean()

    g_mean = float(gray.mean())
    g_std = float(gray.std())

    return np.array([g_mean / 255.0, g_std / 255.0, ch_std / 255.0], dtype=np.float32)




## === cell 4
sample_images = test_df.sample(16, random_state=1)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
fig.tight_layout(pad=1.0)

for i, ax in enumerate(axes.flat):
    fn = sample_images.iloc[i]["filename"]
    img = mpimg.imread(os.path.join(test_images, fn))
    ax.imshow(img)
    ax.set_title("test")
    ax.axis("off")

plt.show()



## === cell 5

rng = np.random.default_rng(1)
n_fit = 6000  # balanced-ish subset size for calibration
idx = rng.choice(len(train_df), size=min(n_fit, len(train_df)), replace=False)
fit_df = train_df.iloc[idx].reset_index(drop=True)

X_list = []
y_list = []
missing = 0
for i, row in fit_df.iterrows():
    path = os.path.join(TRAIN_DIR, row["id"] + ".tif")
    if not os.path.exists(path):
        missing += 1
        continue
    X_list.append(image_feature(path))
    y_list.append(int(row["label"]))

print("Calibration subset size used:", len(X_list), "missing:", missing)
X = np.vstack(X_list)  # (n, 3)
y = np.asarray(y_list, dtype=np.float32)  # (n,)

mu = X.mean(axis=0, keepdims=True)
sigma = X.std(axis=0, keepdims=True) + 1e-6
Xn = (X - mu) / sigma


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


w = np.zeros((Xn.shape[1],), dtype=np.float32)
b = np.float32(0.0)

lr = 0.1
steps = 400  # small but sufficient for 3 features
for t in range(steps):
    z = Xn @ w + b
    p = sigmoid(z)
    err = p - y
    gw = (Xn.T @ err) / Xn.shape[0]
    gb = err.mean()
    w -= lr * gw.astype(np.float32)
    b -= lr * np.float32(gb)

p_train = sigmoid(Xn @ w + b)
print("Train subset mean pred:", float(p_train.mean()), "label mean:", float(y.mean()))



## === cell 6

test_probs = np.zeros((len(test_df),), dtype=np.float32)
missing_test = 0

for i, fn in enumerate(test_df["filename"].tolist()):
    path = os.path.join(TEST_DIR, fn)
    if not os.path.exists(path):
        missing_test += 1
        test_probs[i] = 0.5
        continue
    f = image_feature(path)[None, :]  # (1,3)
    fnorm = (f - mu) / sigma
    test_probs[i] = sigmoid((fnorm @ w)[0] + b)

print("Missing test files:", missing_test)
print("Pred range:", float(test_probs.min()), float(test_probs.max()))



## === cell 7
submission = pd.DataFrame(
    {"id": test_df["id"].astype(str), "label": test_probs.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 8
frequency_distribution = (
    pd.cut(submission["label"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 4))
frequency_distribution.plot(kind="bar", color="steelblue")
plt.title("Prediction Probability Distribution (binned)")
plt.xlabel("Predicted probability bin")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
