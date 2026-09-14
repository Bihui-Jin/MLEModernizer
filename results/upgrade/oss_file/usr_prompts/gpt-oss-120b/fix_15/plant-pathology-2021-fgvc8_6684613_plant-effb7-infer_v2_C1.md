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

0.7559372114496771

# 6. Current score

0.27019

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'The fix removes the problematic TensorFlow import (setting `tf = None`), and filters the test‑image directory so only actual image files are kept, preventing extra rows that cause a submission size mismatch. These minimal changes allow the notebook to run end‑to‑end and produce a valid `submission.csv` with the correct number of rows.'
- What this solution (achieved 0.38173) has done: 'The baseline was predicting a single most‑common label for every test image, yielding a very low F1. To move the score toward the target we keep the same overall pipeline but replace that naïve prediction with a multi‑label guess using the three most frequent classes from the training set. Predicting several common labels improves recall and typically raises the mean F1 while still respecting the original logic and without adding new dependencies.'
- What this solution (achieved 0.38173) has done: 'I replace the static “top‑3 most common labels” heuristic with a small routine that evaluates every possible K (1 … number of classes) on the training data, picks the K that gives the highest mean‑sample F1 when using the K most frequent classes for every image, and then uses that K to build the submission. This keeps the original pipeline (no model training) but should raise recall without adding external libraries, moving the score toward the target.'
- What this solution (achieved 0.35308) has done: 'I keep the overall pipeline unchanged but improve the heuristic for the submission. After selecting the optimal K most‑common labels, I also always add the “complex” class (if it isn’t already among the chosen labels). This modest change increases recall for images that contain many diseases, which should raise the mean F1‑score toward the target without altering the core logic or adding new dependencies.'
- What this solution (achieved 0.27019) has done: 'I replace the fixed “top‑K most common labels + always add ‘complex’” heuristic with a data‑driven choice: compute the average number of true labels per training image and use that count `avg_k` to select the most frequent classes. The “complex” class only be added if it appears in at least 10 % of the training samples, which keeps precision higher while still improving recall. This small, targeted change keeps the original pipeline intact but better aligns the predicted label set with the training‑set label distribution, moving the mean‑F1 score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random
import math

tf = None




## === cell 1
def auto_select_accelerator():
    """
    Dummy accelerator selector kept for compatibility.
    Returns a simple namespace with one replica when TF is unavailable.
    """

    class DummyStrategy:
        num_replicas_in_sync = 1

    return DummyStrategy()




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[0]  # unchanged

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
print("Number of classes:", len(class_name))

df["labels"] = df["labels"].astype(str)
n_labels = len(class_name)



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

test_files = [
    f
    for f in os.listdir(test_dir)
    if os.path.isfile(os.path.join(test_dir, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]

test_df = pd.DataFrame()
test_df["image"] = test_files



## === cell 4
test_paths = [os.path.join(test_dir, fname) for fname in test_df["image"].values]



## === cell 5
if tf is not None:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

    with tf.distribute.get_strategy().scope():
        base = tf.keras.applications.EfficientNetB7(
            weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3)
        )
        model = tf.keras.Sequential(
            [
                base,
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Dense(n_labels, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
            metrics=["accuracy"],
        )
        model.summary()
else:
    model = None



## === cell 6
most_common_labels = df["labels"].value_counts().index.tolist()

true_label_sets = df["labels"].apply(lambda x: set(x.split())).tolist()

avg_len = int(round(np.mean([len(s) for s in true_label_sets])))
avg_len = max(1, min(avg_len, n_labels))  # keep within sensible bounds

complex_freq = (df["labels"].str.contains(r"\bcomplex\b")).mean()
include_complex = complex_freq >= 0.10  # add only if at least 10 % of samples have it

selected_labels = most_common_labels[:avg_len]
if include_complex and "complex" not in selected_labels:
    selected_labels.append("complex")

multi_label = " ".join(selected_labels)

print(f"Average true label count per image: {avg_len}")
print(
    f"'complex' frequency: {complex_freq:.3f} – {'adding' if include_complex else 'not adding'} it"
)
print(f"Predicting labels: {multi_label}")

test_df["labels"] = multi_label



## === cell 7
if model is not None:
    weight_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
    weights_loaded = False
    if os.path.exists(weight_path):
        try:
            model.load_weights(weight_path)
            weights_loaded = True
            print("Loaded external weights, skipping training.")
        except Exception as e:
            print(f"Could not load external weights: {e}")

    if not weights_loaded:
        print("Skipping model training in baseline mode.")
else:
    print("TensorFlow not available; model training is omitted.")



## === cell 8
submission_path = "submission.csv"
test_df[["image", "labels"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
test_df.head()
