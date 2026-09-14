# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pandas as pd
from csv import writer
from collections import Counter
from PIL import Image
import numpy as np

TRAIN_CSV_PATH = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGES_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TEST_IMAGES_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV_PATH)
most_common_label = int(train_df["label"].mode()[0])


def leading_char(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    return base[0] if base else ""


def prefix_two(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    return base[:2] if len(base) >= 2 else base


def suffix_two(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    return base[-2:] if len(base) >= 2 else base


def remainder_mod(fname: str, mod: int) -> int:
    base = os.path.splitext(fname)[0]
    try:
        return int(base) % mod
    except ValueError:
        return -1


def remainder_mod5(fname: str) -> int:
    return remainder_mod(fname, 5)


def remainder_mod7(fname: str) -> int:
    return remainder_mod(fname, 7)


def remainder_mod3(fname: str) -> int:
    return remainder_mod(fname, 3)


def remainder_mod11(fname: str) -> int:
    return remainder_mod(fname, 11)


def remainder_mod13(fname: str) -> int:
    return remainder_mod(fname, 13)


def suffix_char(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    return base[-1] if base else ""


def name_length(fname: str) -> int:
    base = os.path.splitext(fname)[0]
    return len(base)


def _mode_and_count(series):
    """Return (mode_label, occurrence_count) for a pandas Series."""
    m = series.mode()
    if not m.empty:
        label = int(m.iloc[0])
        cnt = int((series == label).sum())
        return label, cnt
    else:
        return most_common_label, 0


train_df["lead_char"] = train_df["image_id"].apply(leading_char)
lead_char_mode = {
    k: _mode_and_count(g["label"]) for k, g in train_df.groupby("lead_char")
}

train_df["prefix2"] = train_df["image_id"].apply(prefix_two)
prefix2_mode = {k: _mode_and_count(g["label"]) for k, g in train_df.groupby("prefix2")}

train_df["suffix2"] = train_df["image_id"].apply(suffix_two)
suffix2_mode = {k: _mode_and_count(g["label"]) for k, g in train_df.groupby("suffix2")}

train_df["rem5"] = train_df["image_id"].apply(remainder_mod5)
rem5_mode = {
    r: _mode_and_count(g["label"]) for r, g in train_df.groupby("rem5") if r != -1
}

train_df["rem7"] = train_df["image_id"].apply(remainder_mod7)
rem7_mode = {
    r: _mode_and_count(g["label"]) for r, g in train_df.groupby("rem7") if r != -1
}

train_df["rem3"] = train_df["image_id"].apply(remainder_mod3)
rem3_mode = {
    r: _mode_and_count(g["label"]) for r, g in train_df.groupby("rem3") if r != -1
}

train_df["rem11"] = train_df["image_id"].apply(remainder_mod11)
rem11_mode = {
    r: _mode_and_count(g["label"]) for r, g in train_df.groupby("rem11") if r != -1
}

train_df["rem13"] = train_df["image_id"].apply(remainder_mod13)
rem13_mode = {
    r: _mode_and_count(g["label"]) for r, g in train_df.groupby("rem13") if r != -1
}

train_df["suffix"] = train_df["image_id"].apply(suffix_char)
suffix_mode = {k: _mode_and_count(g["label"]) for k, g in train_df.groupby("suffix")}

train_df["len"] = train_df["image_id"].apply(name_length)
len_mode = {k: _mode_and_count(g["label"]) for k, g in train_df.groupby("len")}


def average_rgb(image_path: str) -> np.ndarray:
    """Return a (3,) array with mean R, G, B values."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        return arr.mean(axis=(0, 1))


def downsample_vector(image_path: str, size: int = 16) -> np.ndarray:
    """
    Resize the image to (size, size), convert to RGB, and flatten.
    Returns a 3*size*size vector of float32.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.reshape(-1)  # shape (3*size*size,)


print(
    "Computing average colours and down‑sample vectors for training images (this may take a minute)..."
)
train_means = []
train_vectors = []
train_labels = []
for _, row in train_df.iterrows():
    img_file = os.path.join(TRAIN_IMAGES_DIR, row["image_id"])
    if os.path.isfile(img_file):
        train_means.append(average_rgb(img_file))
        train_vectors.append(downsample_vector(img_file, size=16))
        train_labels.append(int(row["label"]))
    else:
        train_means.append(np.array([0, 0, 0], dtype=np.float32))
        train_vectors.append(np.zeros(3 * 16 * 16, dtype=np.float32))
        train_labels.append(most_common_label)

train_means = np.stack(train_means)  # (N_train, 3)
train_vectors = np.stack(train_vectors)  # (N_train, 3*size*size)
train_labels = np.array(train_labels)

valid_exts = {".jpg", ".jpeg", ".png"}
test_images = sorted(
    [
        f
        for f in os.listdir(TEST_IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
)




## === cell 1
with open(SUBMISSION_PATH, mode="w", newline="") as f:
    writer_obj = writer(f)
    writer_obj.writerow(["image_id", "label"])  # header

    for img_name in test_images:
        weighted_candidates = []

        pref2 = prefix_two(img_name)
        if pref2 in prefix2_mode:
            weighted_candidates.append(prefix2_mode[pref2])

        suff2 = suffix_two(img_name)
        if suff2 in suffix2_mode:
            weighted_candidates.append(suffix2_mode[suff2])

        lead = leading_char(img_name)
        if lead in lead_char_mode:
            weighted_candidates.append(lead_char_mode[lead])

        rem13 = remainder_mod13(img_name)
        if rem13 in rem13_mode:
            weighted_candidates.append(rem13_mode[rem13])

        rem11 = remainder_mod11(img_name)
        if rem11 in rem11_mode:
            weighted_candidates.append(rem11_mode[rem11])

        rem7 = remainder_mod7(img_name)
        if rem7 in rem7_mode:
            weighted_candidates.append(rem7_mode[rem7])

        rem5 = remainder_mod5(img_name)
        if rem5 in rem5_mode:
            weighted_candidates.append(rem5_mode[rem5])

        rem3 = remainder_mod3(img_name)
        if rem3 in rem3_mode:
            weighted_candidates.append(rem3_mode[rem3])

        suff = suffix_char(img_name)
        if suff in suffix_mode:
            weighted_candidates.append(suffix_mode[suff])

        ln = name_length(img_name)
        if ln in len_mode:
            weighted_candidates.append(len_mode[ln])

        test_path = os.path.join(TEST_IMAGES_DIR, img_name)

        try:
            test_mean = average_rgb(test_path)  # (3,)
            dists = np.linalg.norm(train_means - test_mean, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_label = train_labels[nearest_idx]
            weighted_candidates.append((nearest_label, 5))
        except Exception:
            pass

        try:
            test_vec = downsample_vector(test_path, size=16)
            dists_vec = np.linalg.norm(train_vectors - test_vec, axis=1)
            nearest_idx_vec = np.argmin(dists_vec)
            nearest_label_vec = train_labels[nearest_idx_vec]
            weighted_candidates.append((nearest_label_vec, 10))
        except Exception:
            pass

        label_weights = {}
        for lbl, wgt in weighted_candidates:
            label_weights[lbl] = label_weights.get(lbl, 0) + wgt

        if label_weights:
            pred_label = max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            pred_label = most_common_label

        writer_obj.writerow([img_name, int(pred_label)])

print(f"Submission written to {SUBMISSION_PATH} with {len(test_images)} rows.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
