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

3.9

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
import os, glob
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf

    print("Tensorflow version", tf.__version__)
except Exception as e:
    print("Tensorflow import failed (ignored):", e)
    tf = None  # ensure tf is defined even if import fails.

tf = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    print("scikit‑learn import failed (ignored):", e)
    LogisticRegression = None
    RandomForestClassifier = None



## === cell 1
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_CLASSES = 5



## === cell 2
import concurrent.futures

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)

train_tfrecords_dir = "../input/cassava-leaf-disease-classification/train_tfrecords"
if not os.path.isdir(train_tfrecords_dir):
    train_tfrecords_dir = (
        "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
    )

tfrecord_paths = sorted(glob.glob(os.path.join(train_tfrecords_dir, "*.tfrec")))
if tf is not None and tfrecord_paths:
    raw_dataset = tf.data.TFRecordDataset(
        tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE
    )

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        img = tf.io.decode_jpeg(parsed["image"], channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0  # normalize
        label = tf.cast(parsed["label"], tf.int32)
        return img, label

    parsed_dataset = raw_dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    img_list = []
    label_list = []
    for img_tensor, lbl in parsed_dataset:
        img_list.append(img_tensor.numpy())
        label_list.append(int(lbl.numpy()))
    img_arrays = np.stack(img_list)  # (N, H, W, C)
    img_labels = np.array(label_list, dtype=int)  # (N,)
else:
    train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_images_dir):
        train_images_dir = (
            "/kaggle/input/cassava-leaf-disease-classification/train_images"
        )

    def _load_img(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.array(img).astype(np.float32) / 255.0  # normalize

    paths_labels = []
    for _, row in train_df.iterrows():
        img_path = os.path.join(train_images_dir, row["image_id"])
        if os.path.exists(img_path):
            paths_labels.append((img_path, int(row["label"])))

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        img_arrays = list(executor.map(lambda pl: _load_img(pl[0]), paths_labels))
    img_labels = np.array([lbl for _, lbl in paths_labels], dtype=int)
    img_arrays = np.stack(img_arrays)

centroids_imgs = np.zeros((NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
counts = np.zeros(NUM_CLASSES, dtype=int)

train_imgs_list = []
train_labels_list = []

for img_arr, label in zip(img_arrays, img_labels):
    centroids_imgs[label] += img_arr
    counts[label] += 1
    train_imgs_list.append(img_arr.reshape(-1))  # flattened
    train_labels_list.append(label)

centroids_imgs = centroids_imgs / np.maximum(counts[:, None, None, None], 1)
rgb_means = centroids_imgs.mean(axis=(1, 2))  # shape (NUM_CLASSES, 3)

X_train_full = np.stack(train_imgs_list)  # (N, D)
y_train_full = np.array(train_labels_list)  # (N,)




## === cell 3
class CentroidImageModel:
    """Nearest‑centroid classifier using flattened mean images."""

    def __init__(self, centroids):
        self.centroids = centroids.reshape(NUM_CLASSES, -1)

    def predict(self, x):
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        dists = np.linalg.norm(x_flat[:, None, :] - self.centroids[None, :, :], axis=2)
        preds = np.argmin(dists, axis=1)
        probs = np.zeros((batch, NUM_CLASSES), dtype=np.float32)
        probs[np.arange(batch), preds] = 1.0
        return probs


class CentroidRGBModel:
    """Nearest‑centroid classifier using per‑class mean RGB values."""

    def __init__(self, rgb_means):
        self.rgb_means = rgb_means  # (NUM_CLASSES, 3)

    def predict(self, x):
        batch = x.shape[0]
        rgb_avg = x.mean(axis=(1, 2))  # (batch, 3)
        dists = np.linalg.norm(rgb_avg[:, None, :] - self.rgb_means[None, :, :], axis=2)
        preds = np.argmin(dists, axis=1)
        probs = np.zeros((batch, NUM_CLASSES), dtype=np.float32)
        probs[np.arange(batch), preds] = 1.0
        return probs


model_image = CentroidImageModel(centroids_imgs)
model_rgb = CentroidRGBModel(rgb_means)




## === cell 4
class LinearSoftmaxModel:
    """Simple linear classifier trained with SGD on flattened images."""

    def __init__(self, input_dim, num_classes, lr=0.05):
        self.W = np.zeros((input_dim, num_classes), dtype=np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)
        self.lr = lr

    def _softmax(self, logits):
        exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def fit(self, X, y, epochs=150, batch_size=256, verbose=False):
        N = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(N)
            X_shuff = X[idx]
            y_shuff = y[idx]
            for start in range(0, N, batch_size):
                end = start + batch_size
                xb = X_shuff[start:end]
                yb = y_shuff[start:end]
                logits = xb @ self.W + self.b
                probs = self._softmax(logits)
                y_one = np.zeros_like(probs)
                y_one[np.arange(yb.shape[0]), yb] = 1.0
                grad_logits = (probs - y_one) / yb.shape[0]
                grad_W = xb.T @ grad_logits
                grad_b = grad_logits.sum(axis=0)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
            if verbose:
                pred = np.argmax(self._softmax(X @ self.W + self.b), axis=1)
                acc = (pred == y).mean()
                print(f"Epoch {epoch+1}/{epochs} - Train acc: {acc:.4f}")

    def predict(self, x):
        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1)
        logits = x @ self.W + self.b
        return self._softmax(logits)


input_dim = IMAGE_SIZE * IMAGE_SIZE * 3
linear_model = LinearSoftmaxModel(input_dim, NUM_CLASSES, lr=0.05)
linear_model.fit(X_train_full, y_train_full, epochs=150, batch_size=256, verbose=False)



## === cell 5
if LogisticRegression is not None:
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        verbose=0,
        n_jobs=-1,
    )
    logreg.fit(X_train_full, y_train_full)

    class LogisticModelWrapper:
        """Wrap scikit‑learn LogisticRegression to match our predict API."""

        def __init__(self, clf):
            self.clf = clf

        def predict(self, x):
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
            return self.clf.predict_proba(x).astype(np.float32)

    logistic_model = LogisticModelWrapper(logreg)
else:
    logistic_model = None

if RandomForestClassifier is not None:
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_clf.fit(X_train_full, y_train_full)

    class RandomForestWrapper:
        """Wrap RandomForest to provide the same predict interface."""

        def __init__(self, clf):
            self.clf = clf

        def predict(self, x):
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
            return self.clf.predict_proba(x).astype(np.float32)

    rf_model = RandomForestWrapper(rf_clf)
else:
    rf_model = None



## === cell 6
test_dir = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"




## === cell 7
def load_test_images(image_dir):
    """
    Load all test images from TFRecords (if present) into a NumPy array (N, H, W, C)
    and return it together with the image filenames.
    """
    test_tfrecords_dir = "../input/cassava-leaf-disease-classification/test_tfrecords"
    if not os.path.isdir(test_tfrecords_dir):
        test_tfrecords_dir = (
            "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
        )

    tfrec_paths = sorted(glob.glob(os.path.join(test_tfrecords_dir, "*.tfrec")))
    if tf is not None and tfrec_paths:
        raw_dataset = tf.data.TFRecordDataset(
            tfrec_paths, num_parallel_reads=tf.data.AUTOTUNE
        )

        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            img = tf.io.decode_jpeg(parsed["image"], channels=3)
            img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
            img = tf.cast(img, tf.float32) / 255.0
            img_id = tf.cast(parsed["image_id"], tf.string)
            return img, img_id

        parsed_dataset = raw_dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

        imgs = []
        ids = []
        for img_tensor, id_tensor in parsed_dataset:
            imgs.append(img_tensor.numpy())
            ids.append(id_tensor.numpy().decode())
        X = np.stack(imgs)  # (N, H, W, C)
        return X, ids
    else:
        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        N = len(img_paths)
        X = np.empty((N, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        ids = [os.path.basename(p) for p in img_paths]

        def _load(p):
            img = Image.open(p).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            return np.array(img).astype(np.float32) / 255.0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            results = list(executor.map(_load, img_paths))

        for i, arr in enumerate(results):
            X[i] = arr
        return X, ids


def get_preds_model_list(image_dir, model_obj_list):
    """
    Predict labels for all images in `image_dir` using the provided list of
    model objects. All models receive the whole batch at once, their
    probabilities are averaged, and the final argmax gives the label.
    """
    X_batch, img_ids = load_test_images(image_dir)  # (N, H, W, C)
    probs_list = []
    for mod in model_obj_list:
        if mod is None:
            continue
        probs = mod.predict(X_batch)  # (N, NUM_CLASSES)
        probs_list.append(probs)
    avg_prob = np.mean(np.stack(probs_list, axis=0), axis=0)  # (N, NUM_CLASSES)
    preds = np.argmax(avg_prob, axis=1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 8
mod_lst = [model_image, model_rgb, linear_model]
if logistic_model is not None:
    mod_lst.append(logistic_model)
if rf_model is not None:
    mod_lst.append(rf_model)

predict_df = get_preds_model_list(test_dir, mod_lst)



## === cell 9
output_path = "submission.csv"
predict_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")



## === cell 10
print(predict_df.head())
