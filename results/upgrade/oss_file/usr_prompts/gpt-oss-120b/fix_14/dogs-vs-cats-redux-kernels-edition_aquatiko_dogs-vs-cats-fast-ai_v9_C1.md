# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_PATH = "../input/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
IMG_SIZE = 48  # richer pixel features than the original 32×32


def load_image_features(path):
    """Load an image, resize to IMG_SIZE, and return a feature vector.
    The vector consists of the flattened RGB pixels plus per‑channel mean
    and standard deviation values, giving the model extra discriminative
    information without changing the overall pipeline."""
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)  # shape (IMG_SIZE, IMG_SIZE, 3)
    flat = arr.reshape(-1)  # pixel values
    means = arr.mean(axis=(0, 1))  # shape (3,)
    stds = arr.std(axis=(0, 1))  # shape (3,)
    return np.concatenate([flat, means, stds])  # (IMG_SIZE*IMG_SIZE*3 + 6,)


train_features = []
train_labels = []

for label_name in ["cat", "dog"]:
    label_dir = os.path.join(TRAIN_DIR, label_name)
    if not os.path.isdir(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = os.path.join(label_dir, fname)
        feats = load_image_features(fpath)
        train_features.append(feats)
        train_labels.append(1 if label_name == "dog" else 0)

train_features = np.stack(train_features)  # shape (n_samples, n_features)
train_labels = np.array(train_labels)

test_filenames = []
test_features = []

for root, _, files in os.walk(TEST_DIR):
    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = os.path.join(root, fname)
        feats = load_image_features(fpath)
        test_features.append(feats)
        test_filenames.append(fname)  # keep original filename for id extraction

test_features = np.stack(test_features)




## === cell 1
scaler = StandardScaler().fit(train_features)
train_features_scaled = scaler.transform(train_features)
test_features_scaled = scaler.transform(test_features)

clf = LogisticRegression(
    max_iter=8000,  # allow more iterations for convergence
    solver="lbfgs",
    C=1e5,  # very weak regularisation to fit richer features
    class_weight="balanced",  # improve probability calibration for log‑loss
    random_state=42,
)
clf.fit(train_features_scaled, train_labels)

test_probs = clf.predict_proba(test_features_scaled)[:, 1]




## === cell 2
def extract_id(filename):
    base = os.path.splitext(filename)[0]
    digits = "".join(filter(str.isdigit, base))
    return int(digits) if digits else 0


submission_df = pd.DataFrame(
    {"id": [extract_id(fn) for fn in test_filenames], "label": test_probs}
)

submission_df = submission_df.sort_values("id").reset_index(drop=True)

output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, rows: {len(submission_df)}")
