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

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by per-image Python overhead in the test inference loop (loading/decoding each JPEG and running three separate model calls for every single image). I preserve the exact ensemble logic and weights, but move test inference to a `tf.data` pipeline with batched decoding/resizing and vectorized model calls, which is equivalent but far faster. I also avoid building unused generators/datasets (training/validation pipelines and callbacks) since no training occurs in this script, and those steps waste time at startup. Finally, I keep determinism (seeds) and ensure output shapes/label semantics remain identical.'
- What this solution (achieved 0.11584) has done: 'The crash happens before any modeling because TensorFlow’s import triggers a protobuf incompatibility in this Kaggle runtime (`MessageFactory.GetPrototype` missing). The smallest reliable fix is to pin the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that failing C++ API path and lets the rest of your pipeline run unchanged. I also remove the unused `matplotlib` import to avoid extra heavy imports, but keep all ensemble logic, preprocessing, weights, and submission formatting identical. After this, the notebook should execute end-to-end and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.61099) has done: 'We need to fix the TensorFlow import crash caused by a protobuf API mismatch (`MessageFactory.GetPrototype`), which currently prevents any model code from running and leads to a very low score. The minimal reliable fix is to force TensorFlow to use the pure-Python protobuf backend and to avoid the C++ backend entirely by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus ensuring `google.protobuf` uses the Python runtime before importing TensorFlow. Additionally, your current script likely falls back to an untrained EfficientNetB0 head (random Dense) when the three SavedModels aren’t found, which yields near-random predictions and explains the 0.11584 score; the smallest score-improving fix is to load the original competition’s ImageNet-pretrained EfficientNetB0 and use it only as a feature extractor with a deterministic, label-frequency prior head when external SavedModels are missing (keeps inference stable and should move accuracy up, without changing any ensemble logic when SavedModels exist). Finally, I ensure the submission is aligned with `sample_submission.csv` order and always writes `/kaggle/working/submission.csv` with correct columns.'
- What this solution (achieved 0.61099) has done: 'I fix the runtime crash that happens before any modeling by ensuring TensorFlow never touches the incompatible protobuf C++ API in this environment: we force the Python protobuf implementation and monkey‑patch the missing `MessageFactory.GetPrototype` symbol before importing TensorFlow. This is the minimal change that unblocks end-to-end execution without changing your ensemble/inference logic. I also make the fallback prior model output explicitly match the expected logits/probabilities shape/dtype and keep submission ordering aligned to `sample_submission.csv`, so a valid `/kaggle/working/submission.csv` is always produced. No changes are made to the ensemble weights, image preprocessing, batching, or argmax label semantics.'
- What this solution (achieved 0.61099) has done: 'I fix the crash in the very first cell by correctly monkey‑patching the protobuf `MessageFactory` class (the current check is wrong because the symbol is missing on the *instance/class actually used* in this runtime). Then I make the SavedModel wrapper robust by always extracting a single tensor from dict/struct outputs and ensuring it’s shaped `[batch, 5]`, so the ensemble arithmetic can’t silently break. Finally, I keep your exact ensemble logic/weights and submission formatting, but add a safe softmax normalization when combining logits vs probabilities (score-stable/improving without changing argmax semantics when outputs are already probabilities), and guarantee `/kaggle/working/submission.csv` is written aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 score strongly suggests the three external SavedModels aren’t being found, so you’re mostly submitting the label-prior fallback (which typically lands around the majority-class baseline). To move toward the 0.9039 target with minimal core-logic change, I keep the exact same ensemble structure/weights and only improve the fallback path: when a SavedModel is missing, we instead use an ImageNet-pretrained EfficientNetB0 + a small linear head trained on `train.csv` (simple softmax regression on fixed features). This preserves the overall “no heavy training loop” approach (only a lightweight closed-form/fast iterative fit on extracted features) and keeps inference semantics identical (still ensembling 3 model outputs). I also keep the existing tf.data batched inference and submission ordering aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory  # noqa: E402

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    if hasattr(_message_factory.MessageFactory, "GetMessageClass"):
        _message_factory.MessageFactory.GetPrototype = (  # type: ignore[attr-defined]
            _message_factory.MessageFactory.GetMessageClass
        )
    else:

        def _missing_getprototype(self, descriptor):  # type: ignore[no-redef]
            raise AttributeError(
                "MessageFactory.GetPrototype is missing in this runtime."
            )

        _message_factory.MessageFactory.GetPrototype = _missing_getprototype  # type: ignore[attr-defined]

try:
    _default_factory = _message_factory.MessageFactory()
    _mf_type = type(_default_factory)
    if not hasattr(_mf_type, "GetPrototype") and hasattr(_mf_type, "GetMessageClass"):
        _mf_type.GetPrototype = _mf_type.GetMessageClass  # type: ignore[attr-defined]
except Exception:
    pass

import tensorflow as tf  # noqa: E402

tf.random.set_seed(42)
np.random.seed(42)

tf.config.run_functions_eagerly(False)

print("TensorFlow version:", tf.__version__)



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
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (
    test_ds.map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 6


def _make_train_ds(paths, labels, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=42, reshuffle_each_iteration=False)
    ds = ds.map(
        lambda p, y: (_decode_resize_normalize(p), tf.cast(y, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _extract_features(model, ds):
    feats = []
    ys = []
    for xb, yb in ds:
        fb = model(xb, training=False)
        feats.append(tf.convert_to_tensor(fb))
        ys.append(tf.convert_to_tensor(yb))
    X = tf.concat(feats, axis=0)
    y = tf.concat(ys, axis=0)
    return X, y


class LinearSoftmaxHead(tf.keras.Model):
    def __init__(self, in_dim, num_classes=5):
        super().__init__()
        self.w = tf.Variable(tf.zeros([in_dim, num_classes], dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32))

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self.w) + self.b


def train_linear_head_on_features(X, y, num_classes=5, l2=1e-4, lr=0.5, steps=300):
    head = LinearSoftmaxHead(int(X.shape[-1]), num_classes=num_classes)
    y = tf.cast(y, tf.int32)
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            logits = head(X)
            ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            )
            reg = l2 * tf.nn.l2_loss(head.w)
            loss = ce + reg
        grads = tape.gradient(loss, head.trainable_variables)
        opt.apply_gradients(zip(grads, head.trainable_variables))
        return loss

    for _ in range(steps):
        _ = train_step()
    return head


def build_imagenet_fallback_model(train_df: pd.DataFrame):
    train_paths = train_df["path"].tolist()
    train_labels = train_df["label"].astype(int).tolist()

    train_ds = _make_train_ds(train_paths, train_labels, batch_size=64, shuffle=True)
    X, y = _extract_features(feature_extractor, train_ds)

    head = train_linear_head_on_features(
        X, y, num_classes=5, l2=1e-4, lr=0.5, steps=250
    )

    inp = tf.keras.Input(shape=(224, 224, 3))
    feats = feature_extractor(inp, training=False)
    logits = head(feats)
    probs = tf.nn.softmax(logits, axis=-1)
    return tf.keras.Model(inp, probs, name="imagenet_linear_fallback")


need_stronger_fallback = (
    (cropnet_model is None)
    or (old_densenet_model is None)
    or (old_efficientnet_model is None)
)
imagenet_fallback_model = None
if need_stronger_fallback:
    imagenet_fallback_model = build_imagenet_fallback_model(train_csv)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1613241182.py in <cell line: 0>()
     88 imagenet_fallback_model = None
     89 if need_stronger_fallback:
---> 90     imagenet_fallback_model = build_imagenet_fallback_model(train_csv)
     91 

/tmp/ipykernel_11/1613241182.py in build_imagenet_fallback_model(train_df)
     70     X, y = _extract_features(feature_extractor, train_ds)
     71 
---> 72     head = train_linear_head_on_features(
     73         X, y, num_classes=5, l2=1e-4, lr=0.5, steps=250
     74     )

/tmp/ipykernel_11/1613241182.py in train_linear_head_on_features(X, y, num_classes, l2, lr, steps)
     58 
     59     for _ in range(steps):
---> 60         _ = train_step()
     61     return head
     62 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filedplaqshm.py in tf__train_step()
     19                     loss = ag__.ld(ce) + ag__.ld(reg)
     20                 grads = ag__.converted_call(ag__.ld(tape).gradient, (ag__.ld(loss), ag__.ld(head).trainable_variables), None, fscope)
---> 21                 ag__.converted_call(ag__.ld(opt).apply_gradients, (ag__.converted_call(ag__.ld(zip), (ag__.ld(grads), ag__.ld(head).trainable_variables), None, fscope),), None, fscope)
     22                 try:
     23                     do_return = True

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in apply_gradients(self, grads_and_vars)
    380 
    381     def apply_gradients(self, grads_and_vars):
--> 382         grads, trainable_variables = zip(*grads_and_vars)
    383         self.apply(grads, trainable_variables)
    384         # Return iterations for compat with tf.keras.

ValueError: in user code:

    File "/tmp/ipykernel_11/1613241182.py", line 56, in train_step  *
        opt.apply_gradients(zip(grads, head.trainable_variables))
    File "/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py", line 382, in apply_gradients  **
        grads, trainable_variables = zip(*grads_and_vars)

    ValueError: not enough values to unpack (expected 2, got 0)


## === cell 7
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
    - Ensure float32 tensor of shape [B, 5].
    """
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    elif isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred = tf.convert_to_tensor(pred)
    pred = tf.cast(pred, tf.float32)

    pred = tf.reshape(pred, [tf.shape(pred)[0], -1])
    pred = pred[:, :5]
    return pred


def _maybe_softmax(pred):
    """
    If outputs are logits, softmax improves comparability across models.
    If outputs are already probabilities, softmax is nearly identity for well-formed probs.
    """
    return tf.nn.softmax(pred, axis=-1)


@tf.function(reduce_retracing=True)
def _ensemble_predict(batch):
    crop_raw = crop_model(batch, training=False)
    dens_raw = dens_model(batch, training=False)
    eff_raw = eff_model(batch, training=False)

    crop_pred = _maybe_softmax(_as_2d_logits_or_probs(crop_raw))
    dens_pred = _maybe_softmax(_as_2d_logits_or_probs(dens_raw))
    eff_pred = _maybe_softmax(_as_2d_logits_or_probs(eff_raw))

    avg_pred = (
        cropnet_weight * crop_pred
        + densenet_weight * dens_pred
        + efficientnet_weight * eff_pred
    )
    return tf.argmax(avg_pred, axis=-1, output_type=tf.int32)


pred_batches = []
for batch in test_ds:
    pred_batches.append(_ensemble_predict(batch))

predictions = tf.concat(pred_batches, axis=0).numpy().astype(int).tolist()

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Unique labels:", submission_df["label"].value_counts().to_dict())
