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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.6415143120960296

# 6. Current score

0.52778

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I remove the problematic `tensorflow_addons` import, replace the missing model load with a simple baseline that predicts the most common label from the training data, and adjust the code so the prediction variable is defined before creating the submission. These fixes eliminate the import error, the file‑not‑found error, and the NameError, allowing the notebook to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.26208) has done: 'The changes speed up feature extraction by processing images in parallel with a thread pool and replace the explicit Python loop that builds the label matrix with `MultiLabelBinarizer`, which creates the binary target array directly. Both modifications keep the exact same feature representation and label ordering, so model training and predictions remain unchanged while drastically reducing I/O‑bound runtime. The rest of the code—including model architecture, training, and submission creation—is untouched.'
- What this solution (achieved 0.42258) has done: 'We enrich the image features by adding channel‑wise standard deviations (so each image is represented by 6 numbers instead of just 3) and train the one‑vs‑rest LogisticRegression with balanced class weights. A small validation split is used to pick a global probability threshold that maximizes the macro F1 on the validation set; this threshold is then applied to the test predictions before creating the submission. These modest tweaks keep the original model structure while expectedly moving the F1 score toward the target.'
- What this solution (achieved 0.44907) has done: 'I enhance the feature extraction by adding mean and standard‑deviation statistics for the HSV colour space (in addition to the existing RGB stats) and slightly increase the LogisticRegression regularisation strength (C = 2.0). These changes keep the overall model pipeline unchanged while giving the classifier richer colour information, which should raise the validation macro‑F1 and move the score closer to the target.'
- What this solution (achieved 0.48061) has done: 'The changes add richer colour statistics (min / max values) to each image’s feature vector, broaden the regularisation parameter slightly, and search the probability threshold over a finer, wider range. These modest enhancements keep the same logistic‑regression‑based pipeline while giving the classifier more informative inputs and a better‑tuned decision cut‑off, which should raise the validation macro F1 and move the score closer to the target.'
- What this solution (achieved 0.49158) has done: 'I enhance the image feature vector by adding a normalized 16‑bin grayscale histogram (giving the model richer texture information) and slightly increase the LogisticRegression regularisation strength (C = 5.0) to allow the classifier to fit these extra features. The threshold search is also refined to 0.005 steps for a marginally better cut‑off. These minimal, targeted tweaks keep the overall pipeline unchanged while aiming to raise the validation macro F1 toward the target score.'
- What this solution (achieved 0.52577) has done: 'I add a small but useful feature (mean grayscale value) to the image statistics, increase the LogisticRegression regularisation strength to C=10.0, and replace the single global probability threshold with an optimal per‑class threshold searched on the validation split. These changes keep the overall pipeline unchanged while giving the model richer input and a better calibrated decision rule, which should raise the macro‑F1 toward the target. The script is also renumbered to start from cell 1 and now reliably writes a valid `submission.csv`.'
- What this solution (achieved 0.52653) has done: 'I keep the overall pipeline unchanged but add a few lightweight feature enhancements and a slightly stronger regularisation to raise the macro‑F1. The image extractor now also returns the grayscale standard deviation (giving the model more texture information) and the threshold sweep is expanded to start from 0.01, allowing finer per‑class cut‑offs. LogisticRegression’s C is increased to 30 (still balanced and linear) and the max‑iterations are raised to 500 to ensure convergence. These minimal, targeted tweaks keep the original architecture while expectedly moving the validation score closer to the target.'
- What this solution (achieved 0.52778) has done: 'I add a few lightweight statistical features (channel‑wise skewness) to the image descriptor, raise the logistic‑regression regularisation a bit and allow more optimizer iterations so the model can fit the richer vector. I also fine‑tune the probability‑threshold search with a finer step (0.001) to get a slightly better macro‑F1 while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import concurrent.futures
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



## === cell 1
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images"

train_df = pd.read_csv(train_csv_path)

train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
all_labels = sorted({lbl for sublist in train_df["label_list"] for lbl in sublist})
num_classes = len(all_labels)

label2idx = {lbl: i for i, lbl in enumerate(all_labels)}
idx2label = {i: lbl for lbl, i in label2idx.items()}




## === cell 2
def extract_rgb_hsv_stats(image_path, size=(64, 64)):
    """
    Load an image, resize, and return extended colour statistics:
    mean, std, min, max for each RGB and HSV channel (24 values),
    skewness for each RGB and HSV channel (6 values),
    a 16‑bin normalized grayscale histogram (16 values),
    mean and std of the grayscale channel (2 values),
    and the mean grayscale intensity (1 value).
    Output shape: (49,)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(size)
        arr_rgb = np.asarray(img) / 255.0  # (H, W, 3)

        mean_rgb = arr_rgb.mean(axis=(0, 1))
        std_rgb = arr_rgb.std(axis=(0, 1))
        min_rgb = arr_rgb.min(axis=(0, 1))
        max_rgb = arr_rgb.max(axis=(0, 1))
        skew_rgb = ((arr_rgb - mean_rgb) ** 3).mean(axis=(0, 1))

        img_hsv = img.convert("HSV")
        arr_hsv = np.asarray(img_hsv) / 255.0  # (H, W, 3)

        mean_hsv = arr_hsv.mean(axis=(0, 1))
        std_hsv = arr_hsv.std(axis=(0, 1))
        min_hsv = arr_hsv.min(axis=(0, 1))
        max_hsv = arr_hsv.max(axis=(0, 1))
        skew_hsv = ((arr_hsv - mean_hsv) ** 3).mean(axis=(0, 1))

        img_gray = img.convert("L")
        arr_gray = np.asarray(img_gray) / 255.0  # (H, W)
        hist, _ = np.histogram(arr_gray, bins=16, range=(0.0, 1.0))
        hist = hist.astype(float) / (hist.sum() + 1e-12)  # normalize
        mean_gray = arr_gray.mean()
        std_gray = arr_gray.std()

        return np.concatenate(
            [
                mean_rgb,
                std_rgb,
                min_rgb,
                max_rgb,
                skew_rgb,
                mean_hsv,
                std_hsv,
                min_hsv,
                max_hsv,
                skew_hsv,
                hist,
                np.array([mean_gray, std_gray]),
            ]
        )
    except Exception:
        return np.full(49, 0.5)


train_image_paths = [
    os.path.join(train_images_dir, row["image"]) for _, row in train_df.iterrows()
]


def parallel_extract(paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        features = list(
            tqdm(
                executor.map(extract_rgb_hsv_stats, paths),
                total=len(paths),
                desc="Building train features",
            )
        )
    return np.stack(features)


X = parallel_extract(train_image_paths)

mlb = MultiLabelBinarizer(classes=all_labels)
y = mlb.fit_transform(train_df["label_list"]).astype(int)



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

base_clf = LogisticRegression(
    max_iter=1000,  # allow more iterations for convergence
    solver="lbfgs",
    class_weight="balanced",
    n_jobs=-1,
    C=50.0,  # a bit stronger regularisation for richer features
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_train, y_train)

val_proba = clf.predict_proba(X_val)

thresholds = np.arange(0.01, 0.991, 0.001)
best_thresh_per_class = np.full(num_classes, 0.5)

for c in range(num_classes):
    best_f1_c = 0.0
    best_t = 0.5
    for t in thresholds:
        pred_c = (val_proba[:, c] >= t).astype(int)
        f1_c = f1_score(y_val[:, c], pred_c, zero_division=1)
        if f1_c > best_f1_c:
            best_f1_c = f1_c
            best_t = t
    best_thresh_per_class[c] = best_t

val_pred_bin = (val_proba >= best_thresh_per_class[np.newaxis, :]).astype(int)
overall_f1 = f1_score(y_val, val_pred_bin, average="macro")
print(f"Per‑class threshold optimisation done (validation macro F1={overall_f1:.4f})")



## === cell 4
test_images_dir = "../input/plant-pathology-2021-fgvc8/test_images"
test_filenames = sorted(
    [
        os.path.join(test_images_dir, f)
        for f in os.listdir(test_images_dir)
        if f.lower().endswith(".jpg")
    ]
)

X_test = parallel_extract(test_filenames)

test_proba = clf.predict_proba(X_test)
pred_binary = (test_proba >= best_thresh_per_class[np.newaxis, :]).astype(int)


def binary_to_labels(vec):
    idxs = np.where(vec == 1)[0]
    if len(idxs) == 0:
        return "healthy"
    return " ".join([idx2label[i] for i in idxs])


pred_labels = [binary_to_labels(row) for row in pred_binary]

submission = pd.DataFrame(
    {
        "image": [os.path.basename(p) for p in test_filenames],
        "labels": pred_labels,
    }
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission.head()
