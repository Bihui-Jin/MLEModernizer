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

# 5. Target score

0.807700831024933

# 6. Current score

0.20384

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20126) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to the pure-Python backend before importing TF, which avoids the `MessageFactory.GetPrototype` error in this environment. Then, because your notebook expects a pre-trained `.h5` model that is not available, I keep the same overall “predict → threshold → space-delimited labels → write submission.csv” flow but replace model loading/inference with a simple, deterministic image-based heuristic so the pipeline runs end-to-end and produces a valid `submission.csv`. I also make the test image directory resolution robust to the two possible dataset mount points shown in your paths, and ensure `n_test`, `preds`, and `class_names` are always defined before use. This is score-neutral in intent (it won’t reach the target), but it unblocks submission generation reliably without changing the output format semantics.'
- What this solution (achieved 0.20384) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t needed for this heuristic pipeline) while keeping the same overall flow: read CSVs → build class list → iterate over test images → produce per-class scores → threshold into space-delimited labels → write `submission.csv`. I replace the TF `ImageDataGenerator` with a small PIL-based batch loader so execution is stable in this Kaggle environment without changing the prediction semantics beyond negligible preprocessing differences. I also make dataset path resolution include `/kaggle/data/...` based on your provided filesystem, ensuring the script finds the images/CSVs reliably. This should run end-to-end and generate a valid submission file; score should improve versus the current broken/low-signal setup simply by correctly processing all test images deterministically (but it won’t change the core heuristic logic).'

# 9. Code solution

## === cell 0
import os

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Setup OK (TensorFlow intentionally not imported).")




## === cell 1
def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


TRAIN_CSV = first_existing(
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/train.csv",
)
SAMPLE_SUB = first_existing(
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)
TEST_DIR = first_existing(
    "../input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/data/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/data/test_images",
)

if TRAIN_CSV is None or SAMPLE_SUB is None or TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not resolve dataset paths. TRAIN_CSV={TRAIN_CSV}, SAMPLE_SUB={SAMPLE_SUB}, TEST_DIR={TEST_DIR}"
    )

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print("Resolved paths:")
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TEST_DIR:", TEST_DIR)
print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(mlb.transform(label_split), columns=class_names)
print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 3
h_target = 384
w_target = 384
batch_size = 32


def iter_test_batches(df, directory, target_size=(384, 384), batch_size=32):
    """
    Replacement for TF/Keras ImageDataGenerator to avoid TF import/runtime issues.
    Yields float32 arrays in [0,1] of shape (bs, H, W, 3), preserving submission order.
    """
    H, W = target_size
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bs = end - start
        batch = np.empty((bs, H, W, 3), dtype=np.float32)

        for i, fname in enumerate(df["image"].iloc[start:end].tolist()):
            path = os.path.join(directory, fname)
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im = im.resize((W, H), resample=Image.BILINEAR)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
            except Exception:
                arr = np.zeros((H, W, 3), dtype=np.float32)
            batch[i] = arr

        yield batch


print("PIL-based generator ready.")



## === cell 4
n_test = len(submissions)

n_classes = len(class_names)
preds = np.zeros((n_test, n_classes), dtype=np.float32)

idx_healthy = class_names.index("healthy") if "healthy" in class_names else None


def squash(x):
    return 1.0 / (1.0 + np.exp(-x))


seen = 0
for batch in iter_test_batches(
    submissions, TEST_DIR, target_size=(h_target, w_target), batch_size=batch_size
):
    bs = batch.shape[0]
    mean = batch.mean(axis=(1, 2, 3))  # (bs,)
    std = batch.std(axis=(1, 2, 3))  # (bs,)
    g = batch[:, :, :, 1].mean(axis=(1, 2))
    rb = 0.5 * (
        batch[:, :, :, 0].mean(axis=(1, 2)) + batch[:, :, :, 2].mean(axis=(1, 2))
    )
    green = g - rb

    for j, cname in enumerate(class_names):
        lc = cname.lower()
        score = -1.0 + 0.0 * mean

        if lc == "healthy":
            score = 3.0 * green - 2.0 * std + 0.5 * (0.5 - np.abs(mean - 0.5))
        elif "rust" in lc:
            score = 2.0 * (mean - 0.45) + 1.0 * std - 1.0 * green
        elif "scab" in lc:
            score = 1.5 * std + 0.5 * (mean - 0.5) - 0.5 * green
        elif "frog" in lc:
            score = 1.0 * std - 0.2 * mean
        elif "complex" in lc:
            score = 1.2 * std + 0.8 * np.abs(mean - 0.5)
        elif "powdery" in lc:
            score = 1.0 * (mean - 0.5) + 0.8 * std
        else:
            score = 0.5 * std - 0.5 * np.abs(mean - 0.5)

        preds[seen : seen + bs, j] = squash(score).astype(np.float32)

    seen += bs

print("Preds shape:", preds.shape)
print("Preds aligned shape:", preds.shape)



## === cell 5
thresh = 0.3
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

out_labels = []
for i in range(n_test):
    p = preds[i]

    c = min(len(class_names), p.shape[-1])
    p = p[:c]

    if healthy_idx is not None and healthy_idx < c and p[healthy_idx] == np.max(p):
        lab = "healthy"
    else:
        chosen = [class_names[j] for j in range(c) if p[j] >= thresh]

        if (len(chosen) == 0) or ("healthy" in chosen):
            j = int(np.argmax(p))
            lab = class_names[j]
        else:
            lab = " ".join(chosen)

    out_labels.append(lab)

submissions_out = submissions.copy()
submissions_out["labels"] = out_labels

submissions_out = submissions_out[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)

print(submissions_out.head())
print("Wrote submission.csv with shape:", submissions_out.shape)



## === cell 6
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions_out)
chk.head()
