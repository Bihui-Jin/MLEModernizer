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

# 5. Target score

0.870806890299184

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20366) has done: 'We make the submission generation more robust and add a tiny heuristic prediction: instead of always using the overall most‑common label we also compute a label from the image file name (numeric part % 5). This keeps the core logic unchanged, guarantees a valid CSV (filters non‑image files, correct header), and should move the accuracy slightly toward the target without altering any model‑training code.'
- What this solution (achieved 0.61099) has done: 'We replace the random “numeric % 5” heuristic with a simple label‑frequency mapping that uses the most common label overall and, when possible, a per‑leading‑digit majority label derived from the training CSV. This keeps the original workflow (reading the CSV, scanning the test folder, writing a CSV) but gives a smarter default prediction, moving the accuracy upward toward the target while preserving the core logic.'
- What this solution (achieved 0.61099) has done: 'The fix adds richer heuristics derived from the training set: a two‑character prefix‑to‑label map, and a remainder‑of‑5 map based on the numeric part of the image name. During submission generation the code first tries the most specific prefix‑2 rule, then the original leading‑character rule, then the remainder rule, and finally falls back to the overall most‑common label. These extra but still lightweight rules are expected to raise accuracy toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight heuristic that uses the mode label for the image ID modulo 7 (an extra numeric pattern) and apply it before falling back to the modulo‑5 rule. This keeps the original workflow unchanged while giving the model another data‑driven rule that should raise accuracy toward the target.'
- What this solution (achieved 0.61099) has done: 'I keep the original workflow (reading the train CSV, scanning the test folder, and writing a CSV) but add a few extra lightweight heuristics derived from the training data – suffix‑character, filename‑length, and modulo‑3 rules – and combine all available rule‑based predictions by taking the most common label for each image (falling back to the overall most‑common label). These additional patterns give the classifier more chances to guess the correct label without changing any model‑training code, moving the accuracy closer to the target score.'
- What this solution (achieved 0.61099) has done: 'I add a few extra, still‑lightweight heuristics derived from the training set – suffix‑two characters, and numeric‑modulo 11 and 13 rules – and combine them with the existing candidates. These extra patterns give the vote‑based predictor more chances to hit the true label, moving the accuracy toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.61099) has done: 'I strengthen the heuristic by recording how often each candidate label appears for a given pattern (e.g., leading‑char, prefix‑2, modulo 5, etc.) and, when predicting, choose the label with the highest accumulated support rather than just the most frequent among the raw candidates. This keeps the overall workflow unchanged while giving more weight to patterns that proved reliable on the training set, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight image‑based heuristic that computes the average RGB colour of every training image once, then for each test image finds the nearest training image in colour space and uses its label. This new signal is combined with the existing filename‑pattern votes, giving the predictor a much stronger clue without changing the overall workflow. The rest of the code (reading CSVs, generating the CSV file) stays the same.'

# 9. Code solution

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


print("Computing average colours for training images (this may take a few seconds)...")
train_means = []
train_labels = []
for _, row in train_df.iterrows():
    img_file = os.path.join(TRAIN_IMAGES_DIR, row["image_id"])
    if os.path.isfile(img_file):
        train_means.append(average_rgb(img_file))
        train_labels.append(int(row["label"]))
    else:
        train_means.append(np.array([0, 0, 0], dtype=np.float32))
        train_labels.append(most_common_label)

train_means = np.stack(train_means)  # shape (N_train, 3)
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

        label_weights = {}
        for lbl, wgt in weighted_candidates:
            label_weights[lbl] = label_weights.get(lbl, 0) + wgt

        if label_weights:
            pred_label = max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            pred_label = most_common_label

        writer_obj.writerow([img_name, int(pred_label)])

print(f"Submission written to {SUBMISSION_PATH} with {len(test_images)} rows.")
