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
import numpy as np
from collections import Counter

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except Exception as e:
    load_model = None
    load_img = None
    img_to_array = None
    print("TensorFlow/Keras import failed:", e)




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_df = pd.read_csv(sample_submission_path)
train_df = pd.read_csv(train_csv_path)
majority_label = train_df["label"].mode()[0]  # most common label in training set




## === cell 2
models_info = [
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
        (550, 550),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5",
        (512, 512),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5",
        (448, 448),
    ),
    (
        "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5",
        (512, 512),
    ),
]

models = []
if load_model is not None:
    for path, size in models_info:
        if os.path.isfile(path):
            try:
                mdl = load_model(path)
                models.append((mdl, size))
                print(f"Loaded model: {path}")
            except Exception as e:
                print(f"Failed to load model {path}: {e}")
        else:
            print(f"Model file not found, skipping: {path}")
else:
    print("Skipping model loading because TensorFlow/Keras could not be imported.")

fallback_model = None


def extract_features(
    img_path, bins=32, rgb_size=(32, 32), hsv_size=(32, 32), flat_size=(32, 32)
):
    """Extract concatenated RGB/HSV histograms and flattened RGB pixels."""
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img_rgb = img.resize(rgb_size)
    arr_rgb = np.array(img_rgb)
    rgb_hist = []
    for ch in range(3):
        h, _ = np.histogram(arr_rgb[:, :, ch], bins=bins, range=(0, 256))
        rgb_hist.append(h.astype(np.float32))
    img_hsv = img.convert("HSV").resize(hsv_size)
    arr_hsv = np.array(img_hsv)
    hsv_hist = []
    for ch in range(3):
        h, _ = np.histogram(arr_hsv[:, :, ch], bins=bins, range=(0, 256))
        hsv_hist.append(h.astype(np.float32))
    img_flat = img.resize(flat_size)
    flat_arr = np.array(img_flat).astype(np.float32).flatten()
    return np.concatenate(rgb_hist + hsv_hist + [flat_arr])


if not models:
    print(
        "Training enhanced fallback classifier on color & HSV histograms + pixel data."
    )
    try:
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from concurrent.futures import ThreadPoolExecutor

        train_image_dir = (
            "/kaggle/input/cassava-leaf-disease-classification/train_images"
        )

        def process_row(row):
            img_path = os.path.join(train_image_dir, row["image_id"])
            if os.path.isfile(img_path):
                return (extract_features(img_path), int(row["label"]))
            else:
                return (None, None)

        rows = [row for _, row in train_df.iterrows()]
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_row, rows))

        X_list = [feat for feat, lbl in results if feat is not None]
        y_list = [lbl for feat, lbl in results if feat is not None]

        X = np.stack(X_list)
        y = np.array(y_list)

        rf = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
        )
        rf.fit(X, y)
        print("RandomForest fallback model trained successfully.")

        et = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
        )
        et.fit(X, y)
        print("ExtraTrees fallback model trained successfully.")

        class EnsembleFallback:
            """Simple majority‑vote ensemble for fallback models."""

            def __init__(self, models):
                self.models = models

            def predict(self, X):
                preds = [model.predict(X) for model in self.models]
                preds = np.array(preds)  # shape: (n_models, n_samples)
                final = []
                for col in preds.T:
                    vals, counts = np.unique(col, return_counts=True)
                    final.append(vals[np.argmax(counts)])
                return np.array(final)

        fallback_model = EnsembleFallback([rf, et])
    except Exception as e:
        print("Failed to train tree‑based fallback models:", e)
        print("Falling back to custom KNN classifier.")

        class SimpleKNN:
            """Very lightweight K‑Nearest‑Neighbors classifier."""

            def __init__(self, n_neighbors=3):
                self.n_neighbors = n_neighbors
                self.X_train = None
                self.y_train = None

            def fit(self, X, y):
                self.X_train = X
                self.y_train = y

            def predict(self, X):
                preds = []
                for x in X:
                    dists = np.linalg.norm(self.X_train - x, axis=1)
                    nn_idx = np.argpartition(dists, self.n_neighbors)[
                        : self.n_neighbors
                    ]
                    nn_labels = self.y_train[nn_idx]
                    vals, counts = np.unique(nn_labels, return_counts=True)
                    pred = vals[np.argmax(counts)]
                    preds.append(pred)
                return np.array(preds)

        train_image_dir = (
            "/kaggle/input/cassava-leaf-disease-classification/train_images"
        )

        def process_row_knn(row):
            img_path = os.path.join(train_image_dir, row["image_id"])
            if os.path.isfile(img_path):
                return (extract_features(img_path), int(row["label"]))
            else:
                return (None, None)

        rows_knn = [row for _, row in train_df.iterrows()]
        X_knn = []
        y_knn = []
        for row in rows_knn:
            feat, lbl = process_row_knn(row)
            if feat is not None:
                X_knn.append(feat)
                y_knn.append(lbl)

        if X_knn:
            X_knn = np.stack(X_knn)
            y_knn = np.array(y_knn)
            knn = SimpleKNN(n_neighbors=3)
            knn.fit(X_knn, y_knn)
            fallback_model = knn
            print("Custom KNN fallback model trained successfully.")
        else:
            print("No training images found; fallback will use majority label.")
            fallback_model = None




## === cell 3
image_predictions = []

test_images = sorted(
    [
        f
        for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

if models:
    per_model_results = [[] for _ in range(len(models))]

    BATCH_SIZE = 512  # larger batch to improve throughput and reduce total loops
    distinct_sizes = {}
    for idx, (_, size) in enumerate(models):
        distinct_sizes.setdefault(size, []).append(idx)

    for batch_start in range(0, len(test_images), BATCH_SIZE):
        batch_names = test_images[batch_start : batch_start + BATCH_SIZE]

        raw_imgs = [
            load_img(os.path.join(test_image_dir, name)) for name in batch_names
        ]

        size_to_batch = {}
        for size in distinct_sizes.keys():
            batch_arrays = []
            for img in raw_imgs:
                resized_img = img.resize(size)
                arr = np.expand_dims(img_to_array(resized_img) / 255.0, axis=0)[0]
                batch_arrays.append(arr)
            size_to_batch[size] = np.stack(batch_arrays, axis=0)

        for m_idx, (model, input_size) in enumerate(models):
            batch_np = size_to_batch[input_size]
            batch_preds = model.predict(batch_np, verbose=0)
            for probs in batch_preds:
                pred_cls = int(np.argmax(probs))
                conf = float(probs[pred_cls])
                per_model_results[m_idx].append((pred_cls, conf))

    for idx, image_id in enumerate(test_images):
        model_predictions = []
        confidence_scores = {}
        for model_res in per_model_results:
            pred_cls, conf = model_res[idx]
            model_predictions.append(pred_cls)
            confidence_scores.setdefault(pred_cls, []).append(conf)

        vote_counts = Counter(model_predictions)
        most_common = vote_counts.most_common()
        final_pred = most_common[0][0]

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
            final_pred = max(tied, key=lambda cls: np.mean(confidence_scores[cls]))

        image_predictions.append({"image_id": image_id, "label": final_pred})

elif fallback_model is not None:
    for image_id in test_images:
        feat = extract_features(os.path.join(test_image_dir, image_id)).reshape(1, -1)
        final_pred = int(fallback_model.predict(feat)[0])
        image_predictions.append({"image_id": image_id, "label": final_pred})
else:
    for image_id in test_images:
        final_pred = int(majority_label)
        image_predictions.append({"image_id": image_id, "label": final_pred})

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]




## === cell 4
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 5
submission_df.head()
