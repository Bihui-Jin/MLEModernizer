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
import os
import pandas as pd
from collections import Counter

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images"
train_images_path = "../input/plant-pathology-2021-fgvc8/train_images"



## === cell 1
train_df = pd.read_csv(train_csv_path)

train_label_dict = dict(zip(train_df["image"], train_df["labels"]))

all_labels = train_df["labels"].str.split().explode()
label_counter = Counter(all_labels)
fallback_label = label_counter.most_common(1)[0][0]  # most common disease



## === cell 2
try:
    from PIL import Image
    import numpy as np

    def primary_label(label_str):
        return label_str.split()[0]

    brightness_vals = {}
    for idx, row in train_df.iterrows():
        img_name = row["image"]
        img_path = os.path.join(train_images_path, img_name)
        if not os.path.isfile(img_path):
            continue
        try:
            with Image.open(img_path).convert("L") as im:  # grayscale
                arr = np.asarray(im, dtype=np.float32)
                mean_bright = arr.mean()
        except Exception:
            continue
        lbl = primary_label(row["labels"])
        brightness_vals.setdefault(lbl, []).append(mean_bright)

    label_median_brightness = {
        lbl: np.median(vals) for lbl, vals in brightness_vals.items() if vals
    }
except Exception:
    label_median_brightness = {}




## === cell 3
def predict_with_brightness(
    test_path, fallback_label, label_dict, label_median_brightness
):
    """
    Predict labels for test images.
    - Exact match: use training label if image exists in the dict.
    - Otherwise: compute image brightness (if possible) and assign the label with
      closest median brightness. If brightness info is unavailable, use fallback_label.
    Returns sorted image ids and corresponding label strings.
    """
    image_ids = sorted(os.listdir(test_path))
    labels = []
    can_compute = bool(label_median_brightness)
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        can_compute = False

    for img in image_ids:
        if img in label_dict:
            labels.append(label_dict[img])
        else:
            chosen = fallback_label
            if can_compute:
                img_path = os.path.join(test_path, img)
                try:
                    with Image.open(img_path).convert("L") as im:
                        arr = np.asarray(im, dtype=np.float32)
                        img_bright = arr.mean()
                    best_label = min(
                        label_median_brightness.items(),
                        key=lambda kv: abs(kv[1] - img_bright),
                    )[0]
                    chosen = best_label
                except Exception:
                    chosen = fallback_label
            labels.append(chosen)
    return image_ids, labels




## === cell 4
image_ids, labels = predict_with_brightness(
    test_dir, fallback_label, train_label_dict, label_median_brightness
)



## === cell 5
submission_file = pd.DataFrame({"image": image_ids, "labels": labels})
submission_file.to_csv("submission.csv", index=False)



## === cell 6
submission_file.head()
