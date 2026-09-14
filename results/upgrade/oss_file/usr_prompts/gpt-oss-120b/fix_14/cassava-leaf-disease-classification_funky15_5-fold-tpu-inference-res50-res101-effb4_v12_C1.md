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

# 5. Target score

0.8590208522212148

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script failed because it tried to import an unused Kaggle helper that crashes with the current protobuf version and because it attempted to load non‑existent pretrained model files. I removed the problematic import, replaced the missing models with lightweight dummy models that always predict the most frequent class from the training set, and ensured images are converted to NumPy arrays before scaling. This fixes the runtime errors and guarantees a correctly‑formatted `submission.csv` is written.'
- What this solution (achieved 0.13004) has done: 'The fix removes the failing TensorFlow import by wrapping it in a safe try/except, adds a lightweight “centroid” model that classifies images based on their average RGB colour (computed from the training set), and replaces the dummy models with this single model to improve accuracy while keeping the original pipeline unchanged. All other logic and file paths remain the same, and a correctly‑named `submission.csv` is written.'
- What this solution (achieved 0.18423) has done: 'Implemented a more informative image‑centroid model and aligned image size throughout the pipeline.  
- Set `IMAGE_SIZE` to 32 so training and inference operate on the same resolution.  
- Accumulate full resized images per class to build mean‑image centroids instead of only mean RGB.  
- Replaced the previous `CentroidModel` with `CentroidImageModel` that predicts via nearest‑centroid distance on the flattened image vectors.  
These changes keep the original workflow intact while substantially improving classification accuracy, moving the score toward the target.'
- What this solution (achieved 0.13752) has done: 'Implemented a higher‑resolution (64 px) image size for richer feature centroids and added a complementary per‑class mean‑RGB model. Both models are ensembled by averaging their one‑hot predictions, boosting discrimination without altering the original workflow. The script now reliably creates a correctly named `submission.csv` while moving the validation score closer to the target.'
- What this solution (achieved 0.142) has done: 'Implemented a lightweight linear classifier trained on flattened 32×32 images and added it to the ensemble while reducing the image size to 32 px (memory‑friendly).  
The training loop runs a few SGD epochs on the full training set, producing a model that captures more discriminative patterns than the simple centroid methods.  
Predictions now average the three models (image‑centroid, RGB‑centroid, linear) to improve accuracy, and the script reliably writes `submission.csv`.'
- What this solution (achieved 0.2216) has done: 'Implemented a lightweight Logistic Regression classifier, wrapped it for consistent prediction handling, and added it to the model ensemble. This boosts discrimination without altering the existing workflow, keeping all original logic intact while improving the validation score toward the target. Added safe import for scikit‑learn and a simple wrapper class to match the other models' `predict` signature.'
- What this solution (achieved 0.46824) has done: 'The changes parallelize the heavy image‑loading steps for both training and test data using a thread pool, which dramatically cuts I/O time while keeping the exact same data order and values. The rest of the pipeline—including centroid computation, all model definitions, training loops, and ensembling—remains unchanged, so prediction accuracy and core logic are preserved.'

# 9. Code solution

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


try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    print("scikit‑learn import failed (ignored):", e)
    LogisticRegression = None
    RandomForestClassifier = None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

    max_workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2086612164.py in <cell line: 0>()
     35     img_list = []
     36     label_list = []
---> 37     for img_tensor, lbl in parsed_dataset:
     38         img_list.append(img_tensor.numpy())
     39         label_list.append(int(lbl.numpy()))

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786850792.py in <cell line: 0>()
     31 
     32 
---> 33 model_image = CentroidImageModel(centroids_imgs)
     34 model_rgb = CentroidRGBModel(rgb_means)
     35 

NameError: name 'centroids_imgs' is not defined

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
        idx = np.arange(N)
        for epoch in range(epochs):
            np.random.shuffle(idx)  # in‑place shuffle avoids copying X each epoch
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]
                xb = X[batch_idx]
                yb = y[batch_idx]
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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1856424696.py in <cell line: 0>()
     44 input_dim = IMAGE_SIZE * IMAGE_SIZE * 3
     45 linear_model = LinearSoftmaxModel(input_dim, NUM_CLASSES, lr=0.05)
---> 46 linear_model.fit(X_train_full, y_train_full, epochs=150, batch_size=256, verbose=False)
     47 
     48 

NameError: name 'X_train_full' is not defined

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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/688546038.py in <cell line: 0>()
      8         n_jobs=-1,
      9     )
---> 10     logreg.fit(X_train_full, y_train_full)
     11 
     12     class LogisticModelWrapper:

NameError: name 'X_train_full' is not defined

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

        max_workers = min(8, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1113872318.py in <cell line: 0>()
----> 1 mod_lst = [model_image, model_rgb, linear_model]
      2 if logistic_model is not None:
      3     mod_lst.append(logistic_model)
      4 if rf_model is not None:
      5     mod_lst.append(rf_model)

NameError: name 'model_image' is not defined

## === cell 9
output_path = "submission.csv"
predict_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1342122951.py in <cell line: 0>()
      1 output_path = "submission.csv"
----> 2 predict_df.to_csv(output_path, index=False)
      3 print(f"Submission written to {output_path}")
      4 
      5 

NameError: name 'predict_df' is not defined

## === cell 10
print(predict_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2160939148.py in <cell line: 0>()
----> 1 print(predict_df.head())

NameError: name 'predict_df' is not defined
