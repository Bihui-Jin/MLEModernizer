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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, re, math, random, pathlib
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    raise RuntimeError("decode_image is unused in this TF-free fallback pipeline.")




## === cell 2
BATCH_SIZE = 32



## === cell 3
candidate_roots = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvcvc8/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
data_root = None
for r in candidate_roots:
    if os.path.exists(r):
        data_root = r
        break
if data_root is None:
    data_root = "../input/plant-pathology-2021-fgvc8"

source = os.path.join(data_root, "test_images")
sample_path = os.path.join(data_root, "sample_submission.csv")
train_path = os.path.join(data_root, "train.csv")

sample_sub = pd.read_csv(sample_path)
image_files = sample_sub["image"].astype(str).tolist()
IMAGE_PATHS = [os.path.join(source, f) for f in image_files]

missing = [f for f, p in zip(image_files, IMAGE_PATHS) if not os.path.exists(p)]
if len(missing) > 0:
    print(
        f"WARNING: Missing {len(missing)} test images in filesystem. Example: {missing[:3]}"
    )
else:
    print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", image_files[:3])



## === cell 4
IMAGE_PATHS[:5]



## === cell 5
existing_paths = [p for p in IMAGE_PATHS if os.path.exists(p)]
if len(existing_paths) == 0:
    print(
        f"WARNING: No readable images found in {source}. Will still generate a valid submission via priors."
    )
else:
    print(f"Readable test images found: {len(existing_paths)}/{len(IMAGE_PATHS)}")



## === cell 6
AUTO = None
existing_mask = np.array([os.path.exists(p) for p in IMAGE_PATHS], dtype=bool)
existing_indices = np.where(existing_mask)[0].tolist()
have_images = len(existing_indices) > 0
test_dataset = None



## === cell 7
print("Using TF-free prior-based predictor (fits class prevalences on train.csv).")



## === cell 8
train_df = pd.read_csv(train_path)

classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(classes)}


def parse_labels(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return str(s).strip().split()


counts = np.zeros(len(classes), dtype=np.float64)
for lab in train_df["labels"].astype(str).tolist():
    labs = set(parse_labels(lab))
    for c in labs:
        if c in class_to_idx:
            counts[class_to_idx[c]] += 1.0

prevalence = counts / max(1.0, float(len(train_df)))
prev_series = pd.Series(prevalence, index=classes).sort_values(ascending=False)

print("Class prevalence (descending):")
print(prev_series)



## === cell 9
disease_classes = [c for c in classes if c != "healthy"]
disease_prev = prev_series.loc[disease_classes]

top1 = disease_prev.index[0]
base_labels = [top1]

print("Base labels used for disease predictions:", base_labels)



## === cell 10
from PIL import Image


def image_stats_gray(path, max_side=256):
    """
    Deterministic, cheap features:
    - mean brightness (0..255)
    - std (contrast)
    Uses a downscaled grayscale thumbnail for speed.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            im = im.resize(
                (int(round(w / scale)), int(round(h / scale))), resample=Image.BILINEAR
            )
        arr = np.asarray(im, dtype=np.float32)
        return float(arr.mean()), float(arr.std())


means = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)
stds = np.full(len(IMAGE_PATHS), np.nan, dtype=np.float32)

if have_images:
    for i in existing_indices:
        try:
            m, s = image_stats_gray(IMAGE_PATHS[i], max_side=256)
            means[i] = m
            stds[i] = s
        except Exception:
            pass

valid_means = means[np.isfinite(means)]
valid_stds = stds[np.isfinite(stds)]

train_img_root = os.path.join(data_root, "train_images")
train_paths = [
    os.path.join(train_img_root, f) for f in train_df["image"].astype(str).tolist()
]
train_exists = [p for p in train_paths if os.path.exists(p)]
have_train_images = len(train_exists) > 0

mean_thr, std_thr = None, None
train_mean_thr, train_std_thr = None, None

if have_train_images:
    N_CAL = 1200
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(train_paths))
    rng.shuffle(idx)
    idx = idx[: min(N_CAL, len(train_paths))]

    train_means = []
    train_stds = []
    train_is_healthy = []

    for j in idx:
        p = train_paths[j]
        if not os.path.exists(p):
            continue
        try:
            m, s = image_stats_gray(p, max_side=256)
            train_means.append(m)
            train_stds.append(s)
            labs = set(parse_labels(train_df.iloc[j]["labels"]))
            train_is_healthy.append(1 if ("healthy" in labs and len(labs) == 1) else 0)
        except Exception:
            continue

    train_means = np.asarray(train_means, dtype=np.float32)
    train_stds = np.asarray(train_stds, dtype=np.float32)
    train_is_healthy = np.asarray(train_is_healthy, dtype=np.int32)

    if train_means.size > 50 and train_stds.size > 50:
        hm = train_means[train_is_healthy == 1]
        hs = train_stds[train_is_healthy == 1]
        dm = train_means[train_is_healthy == 0]
        ds = train_stds[train_is_healthy == 0]

        if hm.size > 10 and hs.size > 10 and dm.size > 10 and ds.size > 10:
            train_mean_thr = float(np.quantile(hm, 0.35))
            train_std_thr = float(np.quantile(hs, 0.65))

if train_mean_thr is not None and train_std_thr is not None:
    mean_thr, std_thr = train_mean_thr, train_std_thr
elif have_images and valid_means.size > 0 and valid_stds.size > 0:
    mean_thr = float(np.quantile(valid_means, 0.60))
    std_thr = float(np.quantile(valid_stds, 0.40))

print(
    "Heuristic thresholds:",
    {
        "mean_thr": mean_thr,
        "std_thr": std_thr,
        "calibrated_on_train": (train_mean_thr is not None),
    },
)



## === cell 11
complex_common = float(prev_series.get("complex", 0.0)) >= 0.20

pred_string = []
for i, img_name in enumerate(image_files):
    if (
        have_images
        and np.isfinite(means[i])
        and np.isfinite(stds[i])
        and mean_thr is not None
        and std_thr is not None
    ):
        is_healthy_like = (means[i] >= mean_thr) and (stds[i] <= std_thr)

        if is_healthy_like:
            pred_string.append("healthy")
        else:
            labs = list(base_labels)
            if complex_common:
                if (means[i] < (mean_thr - 10.0)) or (stds[i] > (std_thr + 12.0)):
                    if "complex" not in labs:
                        labs.append("complex")
            pred_string.append(" ".join(labs))
    else:
        pred_string.append(" ".join(base_labels) if len(base_labels) > 0 else "healthy")

print("Num predictions:", len(pred_string))
print("First 5 predictions:", pred_string[:5])



## === cell 12
pred_string[:10]



## === cell 13
if len(pred_string) < len(image_files):
    pred_string = pred_string + ["healthy"] * (len(image_files) - len(pred_string))
elif len(pred_string) > len(image_files):
    pred_string = pred_string[: len(image_files)]

pred_string = [
    ("healthy" if (s is None or str(s).strip() == "") else str(s).strip())
    for s in pred_string
]



## === cell 14
df = pd.DataFrame({"image": image_files, "labels": pred_string})
assert (
    len(df) == len(image_files) == len(pred_string)
), "Submission lengths do not match."
assert list(df.columns) == [
    "image",
    "labels",
], "Submission columns do not match required format."

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
