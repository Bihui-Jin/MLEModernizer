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

2.7

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

0.8847083711090964

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61248) has done: 'I add a lightweight image‑based model: read a limited number of training images, resize them to a small resolution, flatten the pixels and train a RandomForest classifier (fallback to the majority label if any required library is missing). This simple visual model should give a noticeably higher accuracy than always predicting the majority class, moving the score toward the target while keeping the original workflow intact. The script now loads images, trains the model, generates predictions for the test set, and writes a proper `submission.csv`.'
- What this solution (achieved 0.6151) has done: 'Increasing the number of training images and the capacity of the RandomForest raise validation accuracy, moving the score closer to the target.  
The script now processes all available training samples (removing the 5 k limit), uses a larger forest (500 trees) with balanced class weighting, and stores image pixels as uint8 to keep memory low.  
Minor dtype conversions keep the core “flatten‑pixels‑RF” approach unchanged while improving predictive power.  
The rest of the pipeline (prediction, fallback handling, CSV export) remains the same.'
- What this solution (achieved 0.6151) has done: 'The script was slowed mainly by training a 1000‑tree RandomForest on high‑dimensional image vectors. Reducing the number of trees dramatically cuts runtime while keeping the same model type, feature extraction, and training logic, so the predictions remain comparable. The only change is the `n_estimators` value; all other code, paths, and defaults stay identical.'
- What this solution (achieved 0.6151) has done: 'We boost the simple RandomForest model while preserving its core “flatten‑pixels‑RF” pipeline: increase the image resolution to capture more detail (112 × 112 instead of 96 × 96) and raise the forest size from 200 to 500 trees, which should improve validation accuracy and move the Kaggle score closer to the target without altering the overall workflow. The rest of the script – data loading, fallback handling, and CSV export – stays unchanged.'
- What this solution (achieved 0.6151) has done: 'The changes focus on the two main slow spots: image loading (by using a slightly smaller resize that still preserves the model’s intent) and the RandomForest fit (by returning to the original 500‑tree setting which dramatically cuts training time while keeping the same classifier type). The rest of the workflow, data paths, and prediction logic stay exactly the same, preserving deterministic behaviour and result semantics.'
- What this solution (achieved 0.61547) has done: 'The changes keep the same overall workflow but speed up the two expensive stages: image loading and model training.  The thread pool now uses all CPU cores, and the RandomForest uses far fewer trees (300 instead of 1000), which dramatically cuts training time while preserving the same model type and evaluation logic.  These adjustments are purely performance‑tuning and do not alter the algorithmic core or result semantics.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight PCA dimensionality‑reduction step before the RandomForest so the model works on a compact set of informative features, which usually raises validation accuracy without changing the overall “flatten‑pixels‑RF” workflow. I also increase the forest size to 500 trees for a modest boost. The test data are transformed with the same PCA before prediction, keeping the submission format unchanged.'
- What this solution (achieved 0.61099) has done: 'The fix raises the model capacity slightly while keeping the same “flatten‑pixels → PCA → RandomForest” pipeline: we increase the PCA dimensionality from 150 to 300 components to retain more visual information, and we double the forest size from 500 to 1000 trees. These minimal adjustments are expected to boost validation accuracy and move the Kaggle score closer to the target without altering any other workflow or output format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)




## === cell 1
train_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_path, dtype={"label": np.int32})
train_df["label"] = train_df["label"].astype(int)
majority_label = train_df["label"].value_counts().idxmax()




## === cell 2
try:
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    libs_available = True
except Exception as e:
    print("Required ML libraries not available: {}".format(e))
    libs_available = False




## === cell 3
if libs_available:
    import concurrent.futures

    IMAGE_SIZE = (128, 128)
    MAX_TRAIN_SAMPLES = None  # use all images

    train_images_dir = "../input/cassava-leaf-disease-classification/train_images/"

    def _load_image(path_label):
        """Load, resize and flatten an image; return (array, label) or (None, label) on failure."""
        path, label = path_label
        if not os.path.isfile(path):
            return None, label
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.uint8).flatten()
            return arr, label
        except Exception:
            return None, label

    image_label_pairs = [
        (os.path.join(train_images_dir, row["image_id"]), row["label"])
        for _, row in train_df.iterrows()
    ]

    if MAX_TRAIN_SAMPLES is not None:
        image_label_pairs = image_label_pairs[:MAX_TRAIN_SAMPLES]

    max_workers = max(1, os.cpu_count())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_load_image, image_label_pairs))

    X_list = [arr for arr, lbl in results if arr is not None]
    y_list = [lbl for arr, lbl in results if arr is not None]

    X = np.stack(X_list, axis=0).astype(np.uint8)
    y = np.array(y_list, dtype=np.int32)

    if len(X) == 0:
        print("No training images could be loaded – falling back to majority label.")
        libs_available = False
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        pca = PCA(n_components=300, random_state=42)
        X_train_pca = pca.fit_transform(X_train.astype(np.float32))
        X_val_pca = pca.transform(X_val.astype(np.float32))

        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_train_pca, y_train)
        val_acc = (rf.predict(X_val_pca) == y_val).mean()
        print("Validation accuracy of RF+PCA model: {:.4f}".format(val_acc))




## === cell 4
test_dir = "../input/cassava-leaf-disease-classification/test_images/"
test_filenames = sorted(
    [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
)

if libs_available:
    import concurrent.futures

    def _load_test_image(fname):
        img_path = os.path.join(test_dir, fname)
        if not os.path.isfile(img_path):
            return None
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            return np.asarray(img, dtype=np.uint8).flatten()
        except Exception:
            return None

    max_workers = max(1, os.cpu_count())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        test_arrays = list(executor.map(_load_test_image, test_filenames))

    valid_idx = [i for i, arr in enumerate(test_arrays) if arr is not None]

    if valid_idx:
        valid_arrays = np.stack([test_arrays[i] for i in valid_idx])
        valid_arrays_pca = pca.transform(valid_arrays.astype(np.float32))
        preds = rf.predict(valid_arrays_pca)
    else:
        preds = np.array([])

    final_preds = []
    pred_iter = iter(preds)
    for i in range(len(test_filenames)):
        if i in valid_idx:
            final_preds.append(next(pred_iter))
        else:
            final_preds.append(majority_label)
else:
    final_preds = [majority_label] * len(test_filenames)




## === cell 5
output_path = "submission.csv"
submission_df = pd.DataFrame({"image_id": test_filenames, "label": final_preds})
submission_df.to_csv(output_path, index=False)
print("Submission written to {} with {} rows.".format(output_path, len(submission_df)))
