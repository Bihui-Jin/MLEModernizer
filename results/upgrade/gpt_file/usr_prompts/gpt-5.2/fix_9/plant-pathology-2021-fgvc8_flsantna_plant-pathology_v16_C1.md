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

0.6601477377654648

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09848) has done: 'The timeout is dominated by per-image Python overhead and calling the model one image at a time, which prevents TensorFlow from efficiently pipelining CPU decode/resize and batching inference. I keep the exact same model loading/inference semantics and thresholding logic, but switch to a `tf.data` input pipeline that decodes/resizes in parallel, batches images, and prefetches to overlap I/O and compute. I also wrap inference in a `tf.function` (same computations, less eager overhead) and avoid repeated conversions by moving to batched NumPy post-processing. These changes are equivalent in results (same preprocessing ops, same thresholding/argmax fallback), but drastically reduce runtime.'
- What this solution (achieved 0.28474) has done: 'You’re hitting a TensorFlow/Protobuf incompatibility caused by forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which triggers the `MessageFactory.GetPrototype` error before the model can even load; removing those env overrides fixes runtime. Next, the current low score is consistent with label-index mismatch: you build `dataset_labels` from `train.csv`, but the external SavedModel likely uses a fixed class order (commonly the competition’s standard order), so predictions are being mapped to the wrong label names; I pin the label order to the known Plant Pathology 2021 class list to align outputs correctly. Finally, I keep your exact inference/thresholding semantics, but also ensure the submission rows are aligned to `sample_submission.csv` ordering (safer for Kaggle ingestion) and keep output as a valid `submission.csv`.'
- What this solution (achieved 0.23993) has done: 'We fix the immediate crash by ensuring no protobuf-breaking environment variables are set *before* importing TensorFlow, and by forcing the safe pure-Python protobuf implementation consistently. Then we keep your exact model/inference/threshold logic, but make the external SavedModel output selection more robust (prefer the common “predictions/probabilities” keys) to avoid silently grabbing the wrong tensor. Finally, we keep the submission aligned to `sample_submission.csv` order and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.22303) has done: 'I fix the immediate TensorFlow/Protobuf crash by removing the environment override that forces the pure-Python protobuf implementation (it’s incompatible with the protobuf version in this Kaggle image and triggers `MessageFactory.GetPrototype` errors). Then I keep your exact inference + threshold/argmax fallback logic, but make the SavedModel output tensor selection slightly safer by preferring 2D `(batch, classes)` outputs when multiple tensors exist (this is score-positive without changing the model). Finally, I ensure test image paths are correct, predictions align to `sample_submission.csv` order, and `submission.csv` is always written with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import tensorflow as tf
import numpy as np



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1486332207.py in <cell line: 0>()
      9 
     10 import pandas as pd
---> 11 import tensorflow as tf
     12 import numpy as np
     13 

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
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

model_dir = "../input/model-aug-epoch20/model_complete_with_augEpoch:20"

image_dims = (300, 300, 3)

dataset_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]


def _find_savedmodel_dir(preferred_path: str, fallback_root: str) -> str:
    """Return a directory that contains a TensorFlow SavedModel (saved_model.pb)."""
    preferred_path = os.path.abspath(preferred_path)
    fallback_root = os.path.abspath(fallback_root)

    def is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.exists(os.path.join(p, "saved_model.pb"))

    if is_savedmodel_dir(preferred_path):
        return preferred_path

    candidates = []
    candidates.append(preferred_path.replace(":", "_"))
    candidates.append(preferred_path.replace(":", ""))
    candidates.append(preferred_path.split(":")[0])
    for c in candidates:
        if is_savedmodel_dir(c):
            return c

    if os.path.isdir(fallback_root):
        for name in sorted(os.listdir(fallback_root)):
            p = os.path.join(fallback_root, name)
            if is_savedmodel_dir(p):
                return p

        for root, dirs, files in os.walk(fallback_root):
            if "saved_model.pb" in files:
                return root

    raise FileNotFoundError(
        "Could not locate a SavedModel directory. Tried preferred path and searched under: "
        f"{fallback_root}\nPreferred: {preferred_path}"
    )


try:
    model_dir = _find_savedmodel_dir(model_dir, "../input/model-aug-epoch20")
    has_external_model = True
except FileNotFoundError:
    has_external_model = False
    model_dir = None



## === cell 2
if __name__ == "__main__":
    if not os.path.isdir(test_dir):
        alt_test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
        if os.path.isdir(alt_test_dir):
            test_dir = alt_test_dir
        else:
            raise FileNotFoundError(f"test_dir not found: {test_dir}")

    sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if not os.path.exists(sample_sub_path):
        sample_sub_path = (
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
        )
    sample_df = pd.read_csv(sample_sub_path)
    sample_images = sample_df["image"].tolist()

    if has_external_model:
        loaded = tf.saved_model.load(model_dir)
        if (
            hasattr(loaded, "signatures")
            and isinstance(loaded.signatures, dict)
            and len(loaded.signatures) > 0
        ):
            if "serving_default" in loaded.signatures:
                serving_fn = loaded.signatures["serving_default"]
            else:
                serving_fn = list(loaded.signatures.values())[0]
        else:
            raise RuntimeError(f"SavedModel has no signatures dict at: {model_dir}")

        structured_inputs = serving_fn.structured_input_signature
        _, kw = structured_inputs
        input_key = None
        if isinstance(kw, dict) and len(kw) > 0:
            input_key = list(kw.keys())[0]

        preferred_out_keys = [
            "predictions",
            "prediction",
            "probabilities",
            "probs",
            "outputs",
            "output",
            "dense",
            "sigmoid",
        ]

        def _select_output_tensor(out_dict):
            if not isinstance(out_dict, dict):
                return out_dict
            for k in preferred_out_keys:
                if k in out_dict:
                    return out_dict[k]
            rank2 = []
            for k, v in out_dict.items():
                try:
                    if (
                        hasattr(v, "shape")
                        and v.shape is not None
                        and len(v.shape) == 2
                    ):
                        rank2.append(k)
                except Exception:
                    pass
            if len(rank2) > 0:
                return out_dict[sorted(rank2)[0]]
            return out_dict[sorted(out_dict.keys())[0]]

        @tf.function(reduce_retracing=True)
        def _infer_batch(batch_images):
            if input_key is None:
                out = serving_fn(batch_images)
            else:
                out = serving_fn(**{input_key: batch_images})
            return _select_output_tensor(out)

    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=image_dims,
            pooling="avg",
        )
        inp = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="input_image")
        x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
        x = base(x, training=False)
        out = tf.keras.layers.Dense(
            len(dataset_labels), activation="sigmoid", name="pred"
        )(x)
        infer_model = tf.keras.Model(inputs=inp, outputs=out)

        @tf.function(reduce_retracing=True)
        def _infer_batch(batch_images):
            return infer_model(batch_images, training=False)

    images_path_list = sample_images

    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # -> [0,1]
        img.set_shape([None, None, 3])
        img = tf.image.resize(
            img, [image_dims[0], image_dims[1]], method="bilinear", antialias=True
        )
        return img

    threshold = 0.6
    batch_size = 64

    file_paths = [os.path.join(test_dir, n) for n in images_path_list]
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    values = []
    offset = 0

    for batch_imgs in ds:
        batch_probs = _infer_batch(batch_imgs)
        batch_probs = tf.convert_to_tensor(batch_probs)

        if batch_probs.shape.rank is not None and batch_probs.shape.rank > 2:
            batch_probs = tf.reshape(batch_probs, [tf.shape(batch_probs)[0], -1])

        batch_probs_np = batch_probs.numpy().astype("float32")  # (B, C)

        bsz = batch_probs_np.shape[0]
        for i in range(bsz):
            probs_np = batch_probs_np[i]
            picked = [j for j, v in enumerate(probs_np) if v > threshold]
            if len(picked) == 0:
                picked = [int(probs_np.argmax())]
            classes_img = " ".join([dataset_labels[j] for j in picked]).strip()
            values.append([images_path_list[offset + i], classes_img])
        offset += bsz

    pred_df = pd.DataFrame(values, columns=["image", "labels"])

    sub_df = sample_df[["image"]].merge(pred_df, on="image", how="left")
    sub_df["labels"] = sub_df["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} with shape={sub_df.shape}")
    print(sub_df.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2413782399.py in <cell line: 0>()
     85 
     86     else:
---> 87         base = tf.keras.applications.EfficientNetB0(
     88             include_top=False,
     89             weights="imagenet",

NameError: name 'tf' is not defined
