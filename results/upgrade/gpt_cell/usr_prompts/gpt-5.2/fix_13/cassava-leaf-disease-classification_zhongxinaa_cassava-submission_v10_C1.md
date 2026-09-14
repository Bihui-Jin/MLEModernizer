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

3.12

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0

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

0.8482

# 6. Current score

0.59903

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06502) has done: 'Diagnosis: Cell 4 iterates over every entry returned by `os.listdir(WORK_DIR + 'test_images')`. In this dataset layout there is a nested directory named `test_images` inside the `test_images` folder, so the loop eventually tries to load that directory path as an image, causing `IsADirectoryError`.  
Patch summary: Filter the directory listing to include only actual files (and optionally common image extensions) before calling `load_img`, keeping the prediction and submission logic unchanged.  
Updated cells: Only cell 4 is modified to skip directories.  
Compatibility notes for cell k+1: No interface or variable changes; `submission` and `submission.csv` are produced exactly as before.  
Assumptions: Test images are stored as files within `WORK_DIR/test_images`, and any directories inside it should be ignored.'
- What this solution (achieved 0.10164) has done: 'Your current score is far below the target, and the main issue is that when no pretrained model is found you instantiate an untrained EfficientNetV2B0 (weights=None) and still predict, which yields near-random labels. To move the score toward the target while preserving your architecture and inference flow, I switch the fallback model to use ImageNet pretrained weights (same EfficientNetV2B0 backbone, same pooling + Dense softmax head), which is a minimal change but yields a large accuracy jump. I also fix the input preprocessing to match EfficientNetV2 expectations (use `img_to_array` + `preprocess_input`) without changing prediction semantics. Finally, I ensure test filenames are processed in a stable sorted order while continuing to skip nested directories and producing `submission.csv` with the required columns.'
- What this solution (achieved 0.42526) has done: 'Your current score is far below the target, so we should improve accuracy with minimal risk while keeping your overall inference pipeline intact. The biggest remaining gap is that the fallback EfficientNetV2B0 is ImageNet-pretrained but your Dense(5) head is randomly initialized, so predictions are still nearly random; we instead use the pretrained EfficientNetV2B0 *with its original ImageNet head* and map its 1000-class outputs to your 5 cassava classes using a lightweight calibration step. Specifically, we compute class prototypes by averaging ImageNet softmax vectors over the training images (grouped by cassava label), then classify test images by highest cosine similarity to these prototypes—no training loops, no architecture changes, and it runs fast. If a saved cassava model is found, we keep using it exactly as before; otherwise, this prototype mapping should move your score substantially toward the 0.8482 target.'
- What this solution (achieved 0.4488) has done: 'Your score is far below the target, so we should safely increase accuracy while keeping your current “ImageNet EfficientNetV2B0 + prototype mapper fallback” core logic intact. The biggest low-risk gain is to make the prototype mapping more representative by sampling training images per class uniformly at random (instead of taking the first sorted filenames) and increasing the per-class cap moderately within the time budget. I also make the prototype vectors L2-normalized from the accumulated sums (to match cosine similarity better) and keep everything deterministic via fixed seeds. The submission format and inference flow remain unchanged, and the script still produces `submission.csv`.'
- What this solution (achieved 0.57922) has done: 'Your score is well below the target, so we should improve accuracy with the smallest change that preserves your current “ImageNet EfficientNetV2B0 + prototype mapper fallback” approach. The lowest-risk gain is to compute prototypes from the model’s pre-softmax logits (not softmax probabilities), because cosine similarity in a logit space is typically more discriminative than in a saturated probability space while keeping the exact same backbone and inference loop. I keep the same sampling cap, determinism, and submission writing, only changing prototype/prediction extraction to use the penultimate layer output. This should move your score upward toward the target without altering the overall pipeline structure.'
- What this solution (achieved 0.59903) has done: 'The timeout is dominated by per-image Python loops calling `model.predict()` (and especially the 4× TTA embedding calls inside the prototype path), plus the expensive recursive `os.walk` model search and slow row-by-row DataFrame appends. I keep the exact same model/prototype logic but batch inference: for the prototype path, compute embeddings with TTA in batches by stacking the 4 flips and doing one forward pass per batch; for the direct-model path, predict on batches of test images. I also replace the recursive model search with a fixed, non-recursive candidate list (same semantics, far less I/O) and build the submission via preallocated Python lists (identical output). Finally, I add `tf.data`-style parallel image decoding/preprocessing using TensorFlow ops to remove PIL overhead while preserving the same EfficientNetV2 preprocessing.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
    if int(_pb_ver.split(".", 1)[0]) >= 5:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
import shutil
import cv2, json
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.applications import efficientnet_v2
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
try:
    tf.config.optimizer.set_jit(False)  # keep numerics stable; don't change core logic
except Exception:
    pass

AUTO = tf.data.AUTOTUNE




## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - self.alpha) * (
            1 - y_true
        )
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)




## === cell 3
custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}

candidate_paths = [
    "/kaggle/input/finalmodel3/my_model.h5",
    "../input/finalmodel3/my_model.h5",
    "./my_model.h5",
    "./my_model.keras",
    "./model.h5",
    "./model.keras",
]
model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

use_prototype_mapper = False
embedding_model = None

if model_path is None:
    model = efficientnet_v2.EfficientNetV2B0(
        include_top=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    use_prototype_mapper = True

    embedding_model = tf.keras.Model(
        inputs=model.input, outputs=model.layers[-2].output
    )
else:
    model = load_model(model_path, custom_objects=custom_objects)




## === cell 4
def _iter_image_files(directory):
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        yield name, path


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [224, 224], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    return img  # (224,224,3)


def _load_preprocess_224_batch(file_paths):
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=AUTO)
    ds = ds.map(_decode_resize_preprocess_bytes, num_parallel_calls=AUTO)
    ds = ds.batch(len(file_paths), drop_remainder=False)
    return next(iter(ds)).numpy()


def _tta_stack_tf(batch_bhw3):
    a = batch_bhw3
    a_lr = tf.reverse(a, axis=[2])
    a_ud = tf.reverse(a, axis=[1])
    a_lrud = tf.reverse(a, axis=[1, 2])
    return tf.concat([a, a_lr, a_ud, a_lrud], axis=0)


def _embed_with_tta_batch(batch_np_bhw3, batch_size_pred=64):
    x = tf.convert_to_tensor(batch_np_bhw3, dtype=tf.float32)
    x4 = _tta_stack_tf(x)  # (4B,224,224,3)

    emb4 = embedding_model.predict(x4, batch_size=batch_size_pred, verbose=0)
    B = batch_np_bhw3.shape[0]
    emb4 = emb4.reshape((4, B, emb4.shape[-1])).astype(np.float32)
    emb = emb4.mean(axis=0)  # (B,D)
    return emb


prototypes = None
if use_prototype_mapper:
    train_csv = os.path.join(WORK_DIR, "train.csv")
    train_img_dir = os.path.join(WORK_DIR, "train_images")
    train_df = pd.read_csv(train_csv)

    MAX_PER_CLASS = 600  # keep the same cap

    rng = np.random.default_rng(42)

    embed_dim = int(embedding_model.output_shape[-1])
    protos = np.zeros((5, embed_dim), dtype=np.float32)
    counts = np.zeros((5,), dtype=np.int32)

    BATCH_LOAD = 32
    PRED_BATCH = 64

    for cls in range(5):
        sub = train_df.loc[train_df["label"] == cls, "image_id"].to_numpy()
        if sub.size == 0:
            continue
        take = min(int(MAX_PER_CLASS), int(sub.size))
        chosen = rng.choice(sub, size=take, replace=False)

        paths = []
        for img_id in chosen:
            p = os.path.join(train_img_dir, str(img_id))
            if os.path.isfile(p):
                paths.append(p)

        for i in range(0, len(paths), BATCH_LOAD):
            batch_paths = paths[i : i + BATCH_LOAD]
            if not batch_paths:
                continue
            batch_np = _load_preprocess_224_batch(batch_paths)  # (B,224,224,3)
            emb = _embed_with_tta_batch(batch_np, batch_size_pred=PRED_BATCH)  # (B,D)
            protos[cls] += emb.sum(axis=0)
            counts[cls] += emb.shape[0]

    for cls in range(5):
        if counts[cls] > 0:
            protos[cls] /= float(counts[cls])

    norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12
    prototypes = protos / norms




## === cell 5
test_dir = os.path.join(WORK_DIR, "test_images")
test_items = list(_iter_image_files(test_dir))
test_names = [n for n, _ in test_items]
test_paths = [p for _, p in test_items]

labels_out = []

BATCH_LOAD = 64
PRED_BATCH = 128  # inference batch size; doesn't change core logic

if use_prototype_mapper:
    for i in range(0, len(test_paths), BATCH_LOAD):
        batch_paths = test_paths[i : i + BATCH_LOAD]
        batch_np = _load_preprocess_224_batch(batch_paths)  # (B,224,224,3)
        emb = _embed_with_tta_batch(batch_np, batch_size_pred=PRED_BATCH)  # (B,D)

        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        sims = emb @ prototypes.T  # (B,5)
        pred = np.argmax(sims, axis=1).astype(int).tolist()
        labels_out.extend(pred)
else:
    for i in range(0, len(test_paths), BATCH_LOAD):
        batch_paths = test_paths[i : i + BATCH_LOAD]
        batch_np = _load_preprocess_224_batch(batch_paths)  # (B,224,224,3)
        y_pred = model.predict(batch_np, batch_size=PRED_BATCH, verbose=0)
        pred = np.argmax(y_pred, axis=-1).astype(int).tolist()
        labels_out.extend(pred)

submission = pd.DataFrame({"image_id": test_names, "label": labels_out})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")
