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

0.9038984587488668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by per-image Python overhead in the test inference loop (loading/decoding each JPEG and running three separate model calls for every single image). I preserve the exact ensemble logic and weights, but move test inference to a `tf.data` pipeline with batched decoding/resizing and vectorized model calls, which is equivalent but far faster. I also avoid building unused generators/datasets (training/validation pipelines and callbacks) since no training occurs in this script, and those steps waste time at startup. Finally, I keep determinism (seeds) and ensure output shapes/label semantics remain identical.'
- What this solution (achieved 0.11584) has done: 'The crash happens before any modeling because TensorFlow’s import triggers a protobuf incompatibility in this Kaggle runtime (`MessageFactory.GetPrototype` missing). The smallest reliable fix is to pin the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that failing C++ API path and lets the rest of your pipeline run unchanged. I also remove the unused `matplotlib` import to avoid extra heavy imports, but keep all ensemble logic, preprocessing, weights, and submission formatting identical. After this, the notebook should execute end-to-end and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.61099) has done: 'We need to fix the TensorFlow import crash caused by a protobuf API mismatch (`MessageFactory.GetPrototype`), which currently prevents any model code from running and leads to a very low score. The minimal reliable fix is to force TensorFlow to use the pure-Python protobuf backend and to avoid the C++ backend entirely by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus ensuring `google.protobuf` uses the Python runtime before importing TensorFlow. Additionally, your current script likely falls back to an untrained EfficientNetB0 head (random Dense) when the three SavedModels aren’t found, which yields near-random predictions and explains the 0.11584 score; the smallest score-improving fix is to load the original competition’s ImageNet-pretrained EfficientNetB0 and use it only as a feature extractor with a deterministic, label-frequency prior head when external SavedModels are missing (keeps inference stable and should move accuracy up, without changing any ensemble logic when SavedModels exist). Finally, I ensure the submission is aligned with `sample_submission.csv` order and always writes `/kaggle/working/submission.csv` with correct columns.'
- What this solution (achieved 0.61099) has done: 'I fix the runtime crash that happens before any modeling by ensuring TensorFlow never touches the incompatible protobuf C++ API in this environment: we force the Python protobuf implementation and monkey‑patch the missing `MessageFactory.GetPrototype` symbol before importing TensorFlow. This is the minimal change that unblocks end-to-end execution without changing your ensemble/inference logic. I also make the fallback prior model output explicitly match the expected logits/probabilities shape/dtype and keep submission ordering aligned to `sample_submission.csv`, so a valid `/kaggle/working/submission.csv` is always produced. No changes are made to the ensemble weights, image preprocessing, batching, or argmax label semantics.'
- What this solution (achieved 0.61099) has done: 'I fix the crash in the very first cell by correctly monkey‑patching the protobuf `MessageFactory` class (the current check is wrong because the symbol is missing on the *instance/class actually used* in this runtime). Then I make the SavedModel wrapper robust by always extracting a single tensor from dict/struct outputs and ensuring it’s shaped `[batch, 5]`, so the ensemble arithmetic can’t silently break. Finally, I keep your exact ensemble logic/weights and submission formatting, but add a safe softmax normalization when combining logits vs probabilities (score-stable/improving without changing argmax semantics when outputs are already probabilities), and guarantee `/kaggle/working/submission.csv` is written aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 score strongly suggests the three external SavedModels aren’t being found, so you’re mostly submitting the label-prior fallback (which typically lands around the majority-class baseline). To move toward the 0.9039 target with minimal core-logic change, I keep the exact same ensemble structure/weights and only improve the fallback path: when a SavedModel is missing, we instead use an ImageNet-pretrained EfficientNetB0 + a small linear head trained on `train.csv` (simple softmax regression on fixed features). This preserves the overall “no heavy training loop” approach (only a lightweight closed-form/fast iterative fit on extracted features) and keeps inference semantics identical (still ensembling 3 model outputs). I also keep the existing tf.data batched inference and submission ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the crash in the fallback “linear head training” by ensuring the head variables are properly tracked by Keras (so `head.trainable_variables` is non-empty) and by building the head once before the first gradient step. This is a minimal bug fix that unblocks end-to-end execution when the external SavedModels are missing, which should substantially improve accuracy versus the pure label-prior fallback and move your score toward the 0.9039 target. I keep the ensemble logic/weights, preprocessing, and submission formatting identical, only changing the broken training plumbing. I also add a small safety filter to drop `(None, var)` gradient pairs so `apply_gradients` can’t error in edge cases.'
- What this solution (achieved 0.61099) has done: 'I fix the crash in cell 6 caused by calling `tf.nn.softmax` on a symbolic `KerasTensor` while building a Functional model by replacing it with a Keras-compatible operation (`tf.keras.layers.Softmax`), which preserves identical semantics. I also add a small robustness guard so the linear head training uses integer labels and the feature tensors are materialized consistently, without changing the model/ensemble logic. These changes unblock end-to-end execution so your stronger ImageNet-based fallback can be used when SavedModels are missing, which should improve accuracy versus the pure label-prior fallback and move the score toward the 0.9039 target. The submission formatting and ordering remain aligned to `sample_submission.csv` and the file is always written to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

print("TensorFlow version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

label_counts = train_csv["label"].astype(int).value_counts().sort_index()
label_prior = (
    (label_counts / label_counts.sum())
    .reindex(range(5), fill_value=0.0)
    .values.astype(np.float32)
)
print("Train label prior:", label_prior)
print("Split sizes:", len(train), len(valid))




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model


def find_savedmodel_dir(base_path: str) -> str | None:
    """
    Return a directory that directly contains saved_model.pb or saved_model.pbtxt.
    Searches base_path recursively. Returns None if not found.
    """
    if not base_path or not os.path.exists(base_path):
        return None

    if os.path.isfile(os.path.join(base_path, "saved_model.pb")) or os.path.isfile(
        os.path.join(base_path, "saved_model.pbtxt")
    ):
        return base_path

    for root, dirs, files in os.walk(base_path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
    return None


def build_tfsm_model_from_dataset_root(
    dataset_root: str, call_endpoint: str = "serving_default"
):
    sm_dir = find_savedmodel_dir(dataset_root)
    if sm_dir is None:
        return None, None
    layer = TFSMLayer(sm_dir, call_endpoint=call_endpoint)
    input_layer = Input(shape=(224, 224, 3))
    output_layer = layer(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model, sm_dir




## === cell 4
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

CROPNET_DATASET_ROOT = "/kaggle/input/cropnet_from_kaggle"
OLD_DENSENET_DATASET_ROOT = "/kaggle/input/old_densenet"
OLD_EFFICIENTNET_DATASET_ROOT = "/kaggle/input/old_efficient_net"

cropnet_model, cropnet_sm_dir = build_tfsm_model_from_dataset_root(CROPNET_DATASET_ROOT)
old_densenet_model, densenet_sm_dir = build_tfsm_model_from_dataset_root(
    OLD_DENSENET_DATASET_ROOT
)
old_efficientnet_model, effnet_sm_dir = build_tfsm_model_from_dataset_root(
    OLD_EFFICIENTNET_DATASET_ROOT
)

print("Located SavedModel dirs:")
print(" cropnet:", cropnet_sm_dir)
print(" densenet:", densenet_sm_dir)
print(" effnet :", effnet_sm_dir)

prior_const = tf.constant(label_prior.reshape((1, 5)), dtype=tf.float32)
inp = Input(shape=(224, 224, 3))
out = Lambda(lambda x: tf.tile(prior_const, [tf.shape(x)[0], 1]), name="tile_prior")(
    inp
)
fallback_model = Model(inputs=inp, outputs=out, name="prior_fallback_model")

need_stronger_fallback = (
    (cropnet_model is None)
    or (old_densenet_model is None)
    or (old_efficientnet_model is None)
)

feature_extractor = None
if need_stronger_fallback:
    feature_extractor = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )




## === cell 5
import numpy as np
import pandas as pd

cropnet_weight = 0.5
densenet_weight = 0.3
efficientnet_weight = 0.2

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

img_size = (224, 224)
submission_path = "/kaggle/working/submission.csv"

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64  # batching only; does not alter model semantics

test_paths = [os.path.join(image_dir, iid) for iid in test_image_ids]


def _decode_resize_normalize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


options = tf.data.Options()
options.experimental_deterministic = True

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = (
    test_ds.map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 6
def _make_train_ds(paths, labels, batch_size=64, shuffle=True, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=42, reshuffle_each_iteration=False)
    ds = ds.map(
        lambda p, y: (_decode_resize_normalize(p), tf.cast(y, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32),
    ],
)
def _feats_step(model, xb):
    return model(xb, training=False)


def _extract_features(model, ds, n_samples: int, feat_dim: int):
    """
    Speed: preallocate feature/label arrays and fill them by batch instead of
    appending tensors + concatenating (which is O(#batches) overhead).
    Semantics are identical: same order, same values.
    """
    X = np.empty((n_samples, feat_dim), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.int32)

    offset = 0
    for xb, yb in ds:
        fb = _feats_step(model, xb)
        fb_np = fb.numpy()  # unavoidable host materialization for numpy prealloc
        yb_np = yb.numpy()
        bs = fb_np.shape[0]
        X[offset : offset + bs] = fb_np
        y[offset : offset + bs] = yb_np
        offset += bs

    if offset != n_samples:
        X = X[:offset]
        y = y[:offset]
    return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(
        y, dtype=tf.int32
    )


class LinearSoftmaxHead(tf.keras.Model):
    def __init__(self, in_dim, num_classes=5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.w = self.add_weight(
            name="w",
            shape=(self.in_dim, self.num_classes),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.num_classes,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self.w) + self.b


def train_linear_head_on_features(X, y, num_classes=5, l2=1e-4, lr=0.5, steps=300):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.cast(y, tf.int32)

    head = LinearSoftmaxHead(int(X.shape[-1]), num_classes=num_classes)
    _ = head(X[:1])

    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    @tf.function(
        reduce_retracing=True,
        input_signature=[],
    )
    def train_step():
        with tf.GradientTape() as tape:
            logits = head(X)
            ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            )
            reg = l2 * tf.nn.l2_loss(head.w)
            loss = ce + reg
        grads = tape.gradient(loss, head.trainable_variables)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, head.trainable_variables) if g is not None
        ]
        opt.apply_gradients(grads_and_vars)
        return loss

    for _ in range(steps):
        _ = train_step()
    return head


def build_imagenet_fallback_model(train_df: pd.DataFrame):
    train_paths = train_df["path"].tolist()
    train_labels = train_df["label"].astype(int).tolist()

    train_ds = _make_train_ds(
        train_paths, train_labels, batch_size=64, shuffle=True, cache=False
    )

    feat_dim = 1280
    X, y = _extract_features(
        feature_extractor, train_ds, n_samples=len(train_paths), feat_dim=feat_dim
    )

    head = train_linear_head_on_features(
        X, y, num_classes=5, l2=1e-4, lr=0.5, steps=250
    )

    inp = tf.keras.Input(shape=(224, 224, 3))
    feats = feature_extractor(inp, training=False)
    logits = head(feats)
    probs = tf.keras.layers.Softmax(axis=-1, name="softmax")(logits)
    return tf.keras.Model(inp, probs, name="imagenet_linear_fallback")


imagenet_fallback_model = None
if need_stronger_fallback:
    imagenet_fallback_model = build_imagenet_fallback_model(train)

crop_model = (
    cropnet_model
    if cropnet_model is not None
    else (
        imagenet_fallback_model
        if imagenet_fallback_model is not None
        else fallback_model
    )
)
dens_model = (
    old_densenet_model
    if old_densenet_model is not None
    else (
        imagenet_fallback_model
        if imagenet_fallback_model is not None
        else fallback_model
    )
)
eff_model = (
    old_efficientnet_model
    if old_efficientnet_model is not None
    else (
        imagenet_fallback_model
        if imagenet_fallback_model is not None
        else fallback_model
    )
)


def _as_2d_logits_or_probs(pred):
    """
    Make SavedModel outputs safe for ensembling:
    - If dict: take first value (common TF Serving signature).
    - If structure/tuple/list: take first element.
    - Ensure float32 tensor of shape [B, 5] WITHOUT accidentally slicing wrong dims.
    """
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    elif isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred = tf.convert_to_tensor(pred)
    pred = tf.cast(pred, tf.float32)

    if pred.shape.rank is not None and pred.shape.rank > 2:
        pred = tf.reshape(pred, [tf.shape(pred)[0], -1])
    elif pred.shape.rank is None:
        pred = tf.reshape(pred, [tf.shape(pred)[0], -1])

    pred = tf.reshape(pred, [tf.shape(pred)[0], tf.shape(pred)[-1]])
    if pred.shape[-1] is not None and int(pred.shape[-1]) == 5:
        return pred
    pred = pred[:, :5]
    return pred


def _softmax_temp(x, t: tf.Tensor):
    x = tf.cast(x, tf.float32)
    t = tf.cast(t, tf.float32)
    return tf.nn.softmax(x / t, axis=-1)


@tf.function(reduce_retracing=True)
def _ensemble_probs(batch, temperature: tf.Tensor):
    crop_raw = crop_model(batch, training=False)
    dens_raw = dens_model(batch, training=False)
    eff_raw = eff_model(batch, training=False)

    crop_pred = _softmax_temp(_as_2d_logits_or_probs(crop_raw), temperature)
    dens_pred = _softmax_temp(_as_2d_logits_or_probs(dens_raw), temperature)
    eff_pred = _softmax_temp(_as_2d_logits_or_probs(eff_raw), temperature)

    avg_pred = (
        cropnet_weight * crop_pred
        + densenet_weight * dens_pred
        + efficientnet_weight * eff_pred
    )
    return avg_pred


def _make_valid_ds(valid_df, batch_size=64):
    paths = valid_df["path"].tolist()
    labels = valid_df["label"].astype(int).tolist()
    return _make_train_ds(
        paths, labels, batch_size=batch_size, shuffle=False, cache=True
    )


@tf.function(reduce_retracing=True)
def _valid_correct_total(valid_ds, t_tensor):
    correct = tf.constant(0, tf.int32)
    total = tf.constant(0, tf.int32)
    for xb, yb in valid_ds:
        probs = _ensemble_probs(xb, t_tensor)
        pred = tf.argmax(probs, axis=-1, output_type=tf.int32)
        yb = tf.cast(yb, tf.int32)
        correct += tf.reduce_sum(tf.cast(tf.equal(pred, yb), tf.int32))
        total += tf.shape(yb)[0]
    return correct, total


def _choose_temperature_on_valid(valid_df):
    valid_ds = _make_valid_ds(valid_df, batch_size=BATCH_SIZE)

    temps = [0.8, 1.0, 1.2, 1.5]
    best_t = 1.0
    best_acc = -1.0

    for t in temps:
        t_tensor = tf.constant(t, dtype=tf.float32)
        correct, total = _valid_correct_total(valid_ds, t_tensor)
        correct_i = int(correct.numpy())
        total_i = int(total.numpy())
        acc = correct_i / max(total_i, 1)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"Chosen temperature on valid: t={best_t} (valid acc={best_acc:.4f})")
    return tf.constant(best_t, dtype=tf.float32)


temperature = _choose_temperature_on_valid(valid)


@tf.function(reduce_retracing=True)
def _ensemble_predict(batch, temperature: tf.Tensor):
    probs = _ensemble_probs(batch, temperature)
    return tf.argmax(probs, axis=-1, output_type=tf.int32)


n_test = len(test_image_ids)
predictions_np = np.empty((n_test,), dtype=np.int32)

offset = 0
for batch in test_ds:
    pb = _ensemble_predict(batch, temperature).numpy()
    bs = pb.shape[0]
    predictions_np[offset : offset + bs] = pb
    offset += bs

predictions = predictions_np.astype(int).tolist()

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Unique labels:", submission_df["label"].value_counts().to_dict())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/698646453.py in <cell line: 0>()
     15 # Speed + correctness: compile the feature extraction step once, and enforce a stable
     16 # input signature to avoid retracing overhead; computation is identical.
---> 17 @tf.function(
     18     reduce_retracing=True,
     19     input_signature=[

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in decorated(inner_function)
   1651         inner_function,
   1652         decorator_name="tf.function",
-> 1653         decorator_func=Function(
   1654             inner_function,
   1655             name,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in __init__(self, python_function, name, input_signature, autograph, jit_compile, reduce_retracing, experimental_implements, experimental_autograph_options, experimental_attributes)
    490     self._python_function = python_function
    491     self._function_type, self._default_values = (
--> 492         function_type_utils.make_function_type(python_function, input_signature)
    493     )
    494     self._function_cache = function_cache.FunctionCache()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/function_type_utils.py in make_function_type(python_function, input_signature)
    353   if input_signature is not None:
    354     input_signature = tuple(input_signature)
--> 355     function_type = function_type_lib.add_type_constraints(
    356         function_type, input_signature, default_values
    357     )

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in add_type_constraints(function_type, input_signature, default_values)
    640                       type_constraint))
    641       else:
--> 642         raise TypeError(
    643             f"input_signature missing type constraint for {param.name}")
    644 

TypeError: input_signature missing type constraint for xb
