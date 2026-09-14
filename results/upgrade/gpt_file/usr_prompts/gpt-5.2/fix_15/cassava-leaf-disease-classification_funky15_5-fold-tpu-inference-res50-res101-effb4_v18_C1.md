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

0.8792686612269568

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13827) has done: 'Main bottlenecks are (1) expensive 512×512 bilinear resize + inference and (2) Python-side per-batch work (building filename lists, repeated `os.path.basename` decoding, and eager loops). To fit the 600s limit without changing the model or prediction semantics, I keep TTA off as you already do, but push more work into the `tf.data` graph: decode+resize+optional normalize in a single fused `map`, add `cache()` to avoid any accidental re-decode, and precompute image_ids once (so the loop only handles tensors and writes into preallocated numpy arrays). I also wrap the per-batch forward pass in a `tf.function` to reduce eager overhead while keeping deterministic settings and identical argmax-based labeling.'
- What this solution (achieved 0.61099) has done: 'The immediate blocker is the crash at import time caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` AttributeError) that prevents TensorFlow from importing; we fix this by forcing TensorFlow to use the pure-Python protobuf implementation via environment variables set before importing `tensorflow`. Next, your very low score is consistent with a preprocessing mismatch: you’re currently running EfficientNetB4 with `normalize=False` (feeding 0–255 pixels) even though the model (and the ImageNet backbone fallback) expects normalized/preprocessed inputs; we switch inference to `normalize=True` (divide by 255) to restore correct semantics without changing architecture or inference logic. Finally, we keep the same submission merge/alignment logic and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.15471) has done: 'The timeout is dominated by expensive 512×512 JPEG decoding + resizing for ~2.7k test images, so the key is to maximize input pipeline throughput and remove avoidable overhead while keeping the exact same inference logic (same model, same preprocessing, same argmax). I (1) enable the default fast protobuf C-implementation instead of forcing the slow Python one, (2) switch JPEG decode to the faster `tf.io.decode_image(..., expand_animations=False)` path while preserving 3-channel output, (3) add TF `map` vectorization for resize+preprocess after batching (same math, fewer kernel launches), and (4) avoid dataset `.cache()` (it doesn’t help for single-pass inference and can add overhead/memory pressure). All changes preserve determinism settings and keep paths and prediction semantics identical.'
- What this solution (achieved 0.09417) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation via environment variables that must be set before importing TensorFlow. This is an execution blocker preventing any submission from being generated, and fixing it is score-positive because it enables the intended EfficientNet inference to run. I keep the model/inference logic the same (same EfficientNetB4, same preprocessing/argmax semantics) and only add small robustness checks for file discovery so the pipeline reliably writes `submission.csv` with the required columns. No training, architecture, or prediction post-processing changes are introduced beyond the protobuf fix and safe path handling.'
- What this solution (achieved 0.20142) has done: 'The timeout is dominated by per-image JPEG decoding and 512×512 resizing done serially enough to underutilize the CPU, plus repeated small overheads in the input pipeline (extra maps and Python-side basename extraction). I keep the exact same inference logic (same model, same preprocessing, same argmax) but make the tf.data pipeline more throughput-oriented: set deterministic options once, enable dataset caching (safe for test-only inference), fuse decode+resize+preprocess into a single `map`, and ensure aggressive parallelism + prefetch. I also remove unused augmentation functions from the non-TTA path and avoid Python loops for basename decoding inside the hot loop by emitting image_id strings directly from the dataset. These changes are equivalent in results and should cut end-to-end inference time substantially within the 600s limit.'
- What this solution (achieved 0.20329) has done: 'I fix the execution blocker by changing the protobuf environment variable setup so TensorFlow can import in this Kaggle runtime (your current `"cpp"` setting causes the `_message` import failure). Then, with TensorFlow successfully imported, the downstream `keras`/`tf` NameErrors disappear and the existing EfficientNetB4 inference pipeline can run unchanged. I also make the path resolution for the saved model robust (fall back cleanly to ImageNet weights if the custom `.h5` is not present) while keeping the same architecture and argmax labeling. Finally, I ensure `submission.csv` is always written with the required `image_id,label` columns and aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS", None)

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except AttributeError as e:
    if "MessageFactory" in str(e) and "GetPrototype" in str(e):
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        import tensorflow as tf
        from tensorflow import keras
    else:
        raise

print("Tensorflow version " + tf.__version__)
tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep numerics stable; avoid compilation latency
except Exception:
    pass

_cpu = os.cpu_count() or 1
try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu // 2))
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/662799115.py in <cell line: 0>()
     11 
     12 try:
---> 13     import tensorflow as tf
     14     from tensorflow import keras
     15 except AttributeError as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64



## === cell 2
candidate_model_paths = [
    "../input/efficient-net-0115/efficientnet_0.h5",
    "/kaggle/input/efficient-net-0115/efficientnet_0.h5",
    "/kaggle/data/input/efficient-net-0115/efficientnet_0.h5",
]
model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)

if model_path is not None:
    print("Loading model:", model_path)
    modeleffb4_0 = keras.models.load_model(model_path, compile=False)
else:
    print("Model file not found; building EfficientNetB4 with ImageNet weights.")
    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    x = keras.layers.Dropout(0.2)(base.output)
    out = keras.layers.Dense(5, activation="softmax")(x)
    modeleffb4_0 = keras.Model(inputs=base.input, outputs=out)

mod_lst = [modeleffb4_0]



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/912453800.py in <cell line: 0>()
     11 else:
     12     print("Model file not found; building EfficientNetB4 with ImageNet weights.")
---> 13     base = keras.applications.EfficientNetB4(
     14         include_top=False,
     15         weights="imagenet",

NameError: name 'keras' is not defined

## === cell 3
test_dir = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/data/input/cassava-leaf-disease-classification/test_images"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv"
    )

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"

n_test = len(glob.glob(os.path.join(test_dir, "*.jpg")))
assert n_test > 0, f"No .jpg files found in test_dir: {test_dir}"
print("Found test images:", n_test)




## === cell 4
def _random_resized_crop(img, target_size=IMAGE_SIZE, scale=(0.75, 1.0)):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    s = tf.random.uniform([], scale[0], scale[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * s, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * s, tf.int32)
    new_h = tf.maximum(new_h, 1)
    new_w = tf.maximum(new_w, 1)
    offset_h = tf.random.uniform([], 0, tf.maximum(h - new_h + 1, 1), dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, tf.maximum(w - new_w + 1, 1), dtype=tf.int32)
    cropped = tf.image.crop_to_bounding_box(img, offset_h, offset_w, new_h, new_w)
    cropped = tf.image.resize(cropped, (target_size, target_size), method="bilinear")
    return cropped


def tta_augment(img_np):
    x = _random_resized_crop(img_np, target_size=IMAGE_SIZE, scale=(0.75, 1.0))
    if tf.random.uniform([]) < 0.5:
        x = tf.image.flip_left_right(x)
    if tf.random.uniform([]) < 0.5:
        x = tf.image.flip_up_down(x)
    return x.numpy()




## === cell 5
_PREPROCESS = tf.keras.applications.efficientnet.preprocess_input


@tf.function
def _decode_resize_preprocess(path, normalize):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    if normalize:
        img = _PREPROCESS(img)
    return img


@tf.function
def _make_tta_batch(img, aug_num, normalize):
    ta = tf.TensorArray(tf.float32, size=aug_num)
    i = tf.constant(0, dtype=tf.int32)

    def cond(i, ta):
        return i < aug_num

    def body(i, ta):
        x = _random_resized_crop(img, target_size=IMAGE_SIZE, scale=(0.75, 1.0))
        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_left_right(x)
        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_up_down(x)
        if normalize:
            x = _PREPROCESS(x)
        ta = ta.write(i, x)
        return i + 1, ta

    _, ta = tf.while_loop(cond, body, [i, ta], parallel_iterations=1)
    return ta.stack()  # [aug_num, IMAGE_SIZE, IMAGE_SIZE, 3]


def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    img_paths = sorted(tf.io.gfile.glob(os.path.join(image_dir, "*.jpg")))
    assert len(model_obj_list) >= 1

    model = model_obj_list[0]
    other_models = model_obj_list[1:]

    paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)

    options = tf.data.Options()
    options.deterministic = True
    try:
        options.threading.private_threadpool_size = max(1, (os.cpu_count() or 1))
    except Exception:
        pass
    paths_ds = paths_ds.with_options(options)

    if TTA:
        aug_num_t = tf.constant(int(aug_num), tf.int32)
        norm_t = tf.constant(bool(normalize))

        def _load_for_tta(p):
            img_bytes = tf.io.read_file(p)
            img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            tta = _make_tta_batch(img, aug_num_t, norm_t)
            image_id = tf.strings.split(p, os.sep)[-1]
            return image_id, tta

        ds = (
            paths_ds.map(_load_for_tta, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        preds = []
        out_ids = []

        @tf.function
        def _predict_flat(flat):
            pred = model(flat, training=False)
            if other_models:
                for m in other_models:
                    pred += m(flat, training=False)
                pred /= tf.cast((1 + len(other_models)), pred.dtype)
            return pred

        for id_batch, tta_batch in ds:
            b = tf.shape(tta_batch)[0]
            a = tf.shape(tta_batch)[1]
            flat = tf.reshape(tta_batch, [b * a, IMAGE_SIZE, IMAGE_SIZE, 3])

            pred = _predict_flat(flat)
            pred = tf.reshape(pred, [b, a, -1])  # [B, aug, 5]
            avg_pred = tf.reduce_mean(pred, axis=1)  # [B, 5]
            labels = tf.argmax(avg_pred, axis=-1)  # [B]

            preds.extend(labels.numpy().astype(np.int32).tolist())
            out_ids.extend([x.decode("utf-8") for x in id_batch.numpy()])

        return pd.DataFrame({"image_id": out_ids, "label": preds})

    else:
        norm_t = tf.constant(bool(normalize))

        def _load_one(p):
            image_id = tf.strings.split(p, os.sep)[-1]
            img = _decode_resize_preprocess(p, norm_t)
            return image_id, img

        ds = (
            paths_ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        @tf.function
        def _predict_batch(x_batch):
            pred = model(x_batch, training=False)
            if other_models:
                for m in other_models:
                    pred += m(x_batch, training=False)
                pred /= tf.cast((1 + len(other_models)), pred.dtype)
            return pred

        out_ids = []
        out_labels = []

        for id_batch, x_batch in ds:
            pred = _predict_batch(x_batch)
            labels = tf.argmax(pred, axis=-1)
            out_labels.extend(labels.numpy().astype(np.int32).tolist())
            out_ids.extend([x.decode("utf-8") for x in id_batch.numpy()])

        return pd.DataFrame(
            {"image_id": out_ids, "label": np.asarray(out_labels, dtype=np.int32)}
        )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3434089691.py in <cell line: 0>()
----> 1 _PREPROCESS = tf.keras.applications.efficientnet.preprocess_input
      2 
      3 
      4 @tf.function
      5 def _decode_resize_preprocess(path, normalize):

NameError: name 'tf' is not defined

## === cell 6
def get_preds(image_dir, model_obj, normalize=True):
    img_paths = sorted(tf.io.gfile.glob(os.path.join(image_dir, "*.jpg")))

    paths_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    options = tf.data.Options()
    options.deterministic = True
    paths_ds = paths_ds.with_options(options)

    norm_t = tf.constant(bool(normalize))

    def _load_one(p):
        image_id = tf.strings.split(p, os.sep)[-1]
        img = _decode_resize_preprocess(p, norm_t)
        return image_id, img

    ds = (
        paths_ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def _predict_batch(x_batch):
        return model_obj(x_batch, training=False)

    out_ids = []
    out_labels = []
    for id_batch, x_batch in ds:
        pred = _predict_batch(x_batch)
        labels = tf.argmax(pred, axis=-1)
        out_labels.extend(labels.numpy().astype(np.int32).tolist())
        out_ids.extend([x.decode("utf-8") for x in id_batch.numpy()])

    return pd.DataFrame(
        {"image_id": out_ids, "label": np.asarray(out_labels, dtype=np.int32)}
    )




## === cell 7
predict_df = get_preds_model_list(test_dir, mod_lst, TTA=False, normalize=True)

sub = pd.read_csv(sample_sub_path)
predict_df = sub[["image_id"]].merge(predict_df, on="image_id", how="left")

if predict_df["label"].isna().any():
    predict_df["label"] = predict_df["label"].fillna(0).astype(int)
else:
    predict_df["label"] = predict_df["label"].astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3759362258.py in <cell line: 0>()
----> 1 predict_df = get_preds_model_list(test_dir, mod_lst, TTA=False, normalize=True)
      2 
      3 sub = pd.read_csv(sample_sub_path)
      4 predict_df = sub[["image_id"]].merge(predict_df, on="image_id", how="left")
      5 

NameError: name 'get_preds_model_list' is not defined

## === cell 8
print(predict_df.describe(include="all"))
print("Unique labels:", sorted(predict_df["label"].unique().tolist()))
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1562641982.py in <cell line: 0>()
----> 1 print(predict_df.describe(include="all"))
      2 print("Unique labels:", sorted(predict_df["label"].unique().tolist()))
      3 print("submission.csv exists:", os.path.exists("submission.csv"))

NameError: name 'predict_df' is not defined
