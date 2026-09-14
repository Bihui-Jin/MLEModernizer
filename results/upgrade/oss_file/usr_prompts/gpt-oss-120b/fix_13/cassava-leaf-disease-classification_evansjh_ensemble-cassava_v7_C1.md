# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8893925657298277

# 6. Current score

0.61622

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I guard all TensorFlow imports and model loading so missing libraries or files don’t crash the notebook. If no pretrained models are available, the script fall back to predicting the most frequent class from the training labels, guaranteeing a valid `submission.csv` file. Minor adjustments also fix variable name issues and ensure the CSV is written with the correct columns.'
- What this solution (achieved 0.62369) has done: 'I keep the original workflow but add a lightweight fallback classifier that trains on the training images when the TensorFlow models cannot be loaded. This classifier uses simple color‑histogram features and a RandomForest, which gives a better-than‑majority accuracy without altering the core ensemble logic. The script now always creates a valid `submission.csv`.'
- What this solution (achieved 0.62145) has done: 'Implemented a richer fallback model and unified feature extraction to boost prediction quality while keeping the original workflow intact.  
- Added HSV‑based histograms to the feature set (RGB + HSV).  
- Increased RandomForest trees to 500 and enabled balanced class weighting.  
- Defined a single `extract_features` function used for both training and inference, removing duplicate definitions.  
- Minor clean‑ups ensure the submission CSV is always written correctly.'
- What this solution (achieved 0.61547) has done: 'We add a lightweight custom K‑Nearest‑Neighbors fallback that is used when scikit‑learn isn’t available (or its import fails). This guarantees a working model without external dependencies, while still giving a better‑than‑majority prediction and keeping the original workflow unchanged. The rest of the script (model loading, ensemble voting, CSV writing) stays the same.'
- What this solution (achieved 0.61622) has done: 'The changes focus on removing unnecessary Python‑level loops and repeated work while keeping the exact same models, feature functions, and voting logic.  
- In the fallback training we pre‑allocate the feature matrix and use a `ThreadPoolExecutor` with a sensible `chunksize` to speed up image feature extraction.  
- `EnsembleFallback.predict` is rewritten to compute the majority vote with NumPy instead of a Python loop, preserving the same tie‑breaking rule.  
- During inference with the fallback model we now extract features for **all** test images in one parallel pass and call the ensemble’s `predict` once, eliminating per‑image overhead.  
- Minor micro‑optimisations (caching `os.path.join`, using list comprehensions, avoiding repeated `os.path.isfile` checks) further reduce I/O overhead. All these adjustments keep the original architecture, data handling, and prediction semantics intact.'

# 9. Code solution

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        train_rows = [row for _, row in train_df.iterrows()]
        n_samples = len(train_rows)

        sample_feat = extract_features(
            os.path.join(train_image_dir, train_rows[0]["image_id"])
        )
        feat_dim = sample_feat.shape[0]
        X = np.empty((n_samples, feat_dim), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.int32)

        def load_feat(idx_row):
            idx, row = idx_row
            img_path = os.path.join(train_image_dir, row["image_id"])
            if os.path.isfile(img_path):
                return idx, extract_features(img_path), int(row["label"])
            else:
                return idx, None, None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for idx, feat, lbl in executor.map(
                load_feat, enumerate(train_rows), chunksize=32
            ):
                if feat is not None:
                    X[idx] = feat
                    y[idx] = lbl

        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

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
                preds = np.vstack([m.predict(X) for m in self.models])
                n_samples = preds.shape[1]
                final = np.empty(n_samples, dtype=preds.dtype)
                for i in range(n_samples):
                    cols, counts = np.unique(preds[:, i], return_counts=True)
                    final[i] = cols[np.argmax(counts)]
                return final

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

        rows_knn = [row for _, row in train_df.iterrows()]
        X_knn = []
        y_knn = []
        for row in rows_knn:
            img_path = os.path.join(train_image_dir, row["image_id"])
            if os.path.isfile(img_path):
                X_knn.append(extract_features(img_path))
                y_knn.append(int(row["label"]))

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

    BATCH_SIZE = 512
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
            pred_cls = np.argmax(batch_preds, axis=1)
            conf = np.max(batch_preds, axis=1).astype(float)
            per_model_results[m_idx].extend(zip(pred_cls.tolist(), conf.tolist()))

    for idx, image_id in enumerate(test_images):
        model_predictions = [res[idx][0] for res in per_model_results]
        confidence_scores = {}
        for pred_cls, conf in [res[idx] for res in per_model_results]:
            confidence_scores.setdefault(pred_cls, []).append(conf)

        vote_counts = Counter(model_predictions)
        most_common = vote_counts.most_common()
        final_pred = most_common[0][0]

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
            final_pred = max(tied, key=lambda cls: np.mean(confidence_scores[cls]))

        image_predictions.append({"image_id": image_id, "label": final_pred})

elif fallback_model is not None:
    from concurrent.futures import ThreadPoolExecutor

    test_paths = [os.path.join(test_image_dir, name) for name in test_images]

    def feat_path(p):
        return extract_features(p)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        feats = list(executor.map(feat_path, test_paths, chunksize=32))

    X_test = np.stack(feats)
    preds = fallback_model.predict(X_test)

    if isinstance(fallback_model, type(fallback_model)) and hasattr(
        fallback_model, "models"
    ):
        confs = []
        for sub_model in fallback_model.models:
            prob = sub_model.predict_proba(X_test)
            confs.append(np.max(prob, axis=1))
        confs = np.stack(confs, axis=1)  # shape: n_samples x n_models

        for i, pred in enumerate(preds):
            tied = [cls for cls, cnt in Counter([pred]).items() if cnt > 1]
            if tied:
                class_conf = {}
                for cls in tied:
                    mask = np.array([p == cls for p in preds])
                    class_conf[cls] = np.mean(confs[mask, :])
                pred = max(class_conf, key=class_conf.get)
                preds[i] = pred

    for image_id, label in zip(test_images, preds):
        image_predictions.append({"image_id": image_id, "label": int(label)})

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
