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

0.7845060018467225

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the test image listing bug that incorrectly includes the nested `test_images/` directory, which causes `tf.io.read_file` to try to read a directory and crash. I make the file collection robust by filtering to actual image files and by auto-resolving the correct `test_images` path if the provided one is missing or points to an unexpected structure. I also keep the SavedModel search/wrapping logic intact but make it safe in Kaggle by avoiding the protobuf downgrade (which can break TF 2.18) and by ensuring inference input/output handling is stable. These changes are score-neutral except that they allow the notebook to run end-to-end and generate a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the immediate TensorFlow/protobuf runtime crash by removing the protobuf-import probe that triggers the `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. Then I keep your overall inference pipeline intact but add a minimal, metric-aligned improvement: tune the multi-label threshold on a small validation split using the same mean F1 logic (without changing the model), which should move the score substantially upward from the current overly-high fixed threshold. Finally, I make the preprocessing scale robust by trying both `[0..255]` and `[0..1]` input scaling (picked by validation F1), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash happening at import time by avoiding the incompatible protobuf message-factory path and by forcing TensorFlow to use the Python protobuf implementation (a common Kaggle TF2.18 workaround). Then I keep your inference logic intact but make it robust to different SavedModel output shapes by applying a sigmoid only when outputs look like logits (to improve mean-F1 thresholding without changing the model). Finally, I ensure the test/train directory resolution and submission writing stay the same, so the notebook runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment forcing of the Python protobuf implementation, which is incompatible with this TF 2.18 runtime and triggers the `MessageFactory.GetPrototype` error. Then I keep your SavedModel inference and threshold/scale tuning logic intact, only adding small robustness guards so the notebook always finds the correct data directories and always produces a valid `submission.csv`. These changes should both unblock execution and improve score substantially versus the current run (which is failing before inference), because the tuned threshold/scale and sigmoid-on-logits behavior actually be applied. Finally, I ensure the submission has the exact required columns and space-delimited labels.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment override so TF 2.18 can import cleanly in this Kaggle runtime. Then I renumber the cells and ensure TensorFlow is imported before any functions that reference `tf`, which also fixes the downstream `NameError` chain that prevented submission generation. Finally, I keep your SavedModel inference + threshold/scale tuning logic intact, only making the script run end-to-end and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2835453596.py in <cell line: 0>()
     11 import numpy as np
     12 import pandas as pd
---> 13 import tensorflow as tf
     14 
     15 print("TensorFlow:", tf.__version__)

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
model_dir = "../input/conv2de01/epoch-1"

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])




## === cell 2
def _is_savedmodel_dir(p: str) -> bool:
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "saved_model.pb"))
        or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
    )


def _find_any_savedmodel(start_path: str) -> str:
    """
    Search within ../input for a directory containing saved_model.pb when the provided model_dir
    is not valid in this Kaggle environment. Preserves the same "use a SavedModel" inference logic.
    """
    if start_path and _is_savedmodel_dir(start_path):
        return start_path

    candidates_roots = []
    try:
        parent = os.path.abspath(os.path.join(start_path, os.pardir))
        if os.path.isdir(parent):
            candidates_roots.append(parent)
    except Exception:
        pass
    if os.path.isdir("../input"):
        candidates_roots.append("../input")

    seen = set()
    for root in candidates_roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                return dirpath
    return ""


def _wrap_savedmodel_as_layer(savedmodel_path: str):
    return tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")


def _call_infer(infer_layer, images_batch: tf.Tensor) -> tf.Tensor:
    """
    Try common SavedModel input patterns (tensor first, then dict with common keys).
    """
    try:
        out = infer_layer(images_batch, training=False)
        return out
    except Exception:
        pass

    for key in ("inputs", "input_1", "image", "images", "x"):
        try:
            out = infer_layer({key: images_batch}, training=False)
            return out
        except Exception:
            continue

    raise RuntimeError(
        "Could not call inference layer with tensor or common dict keys."
    )


def _resolve_test_dir(p: str) -> str:
    """
    Resolve to a directory that actually contains image files (handles nested test_images/test_images).
    """
    candidates = []
    if p:
        candidates.append(p)
        candidates.append(os.path.join(p, "test_images"))
    candidates.extend(
        [
            "../input/plant-pathology-2021-fgvc8/test_images",
            "../input/plant-pathology-2021-fgvc8/test_images/test_images",
            "../input/test_images",
            "../input/test_images/test_images",
        ]
    )

    def has_img(dirpath: str) -> bool:
        if not os.path.isdir(dirpath):
            return False
        for fn in os.listdir(dirpath):
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
        return False

    for c in candidates:
        if has_img(c):
            return c

    return p


def _list_image_files(dirpath: str):
    """
    Filter out directories and non-image files.
    """
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"test_dir not found or not a directory: {dirpath}")

    files = []
    for fn in os.listdir(dirpath):
        full = os.path.join(dirpath, fn)
        if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(fn)
    files.sort()
    return files


def _resolve_train_dir() -> str:
    """
    Resolve train_images directory robustly across Kaggle path layouts.
    """
    candidates = [
        "../input/plant-pathology-2021-fgvc8/train_images",
        "../input/plant-pathology-2021-fgvc8/train_images/train_images",
        "../input/train_images",
        "../input/train_images/train_images",
    ]
    for c in candidates:
        if os.path.isdir(c):
            for fn in os.listdir(c):
                if fn.lower().endswith(".jpg") and os.path.isfile(os.path.join(c, fn)):
                    return c
    raise FileNotFoundError(
        "Could not resolve train_images directory from known candidates."
    )


def _extract_pred_tensor(pred_out):
    if isinstance(pred_out, dict):
        if "outputs" in pred_out:
            t = pred_out["outputs"]
        elif "predictions" in pred_out:
            t = pred_out["predictions"]
        else:
            t = next(iter(pred_out.values()))
    else:
        t = pred_out
    t = tf.convert_to_tensor(t)
    if len(t.shape) == 1:
        t = tf.expand_dims(t, axis=0)
    return t


def _maybe_apply_sigmoid(pred: tf.Tensor) -> tf.Tensor:
    """
    Metric-aligned improvement: apply sigmoid if outputs look like logits (outside [0,1]).
    """
    pred = tf.convert_to_tensor(pred)
    min_v = tf.reduce_min(pred)
    max_v = tf.reduce_max(pred)
    if (min_v < -1e-3) or (max_v > 1.0 + 1e-3):
        return tf.math.sigmoid(pred)
    return pred




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2064439495.py in <cell line: 0>()
     40 
     41 
---> 42 def _call_infer(infer_layer, images_batch: tf.Tensor) -> tf.Tensor:
     43     """
     44     Try common SavedModel input patterns (tensor first, then dict with common keys).

NameError: name 'tf' is not defined

## === cell 3
def _load_and_preprocess_image(img_path: str):
    input_img = tf.io.read_file(img_path)
    image = tf.io.decode_image(
        contents=input_img,
        channels=3,
        dtype=tf.dtypes.float32,
        expand_animations=False,
    )
    image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    return image  # float32 in [0,1]


def _mean_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean F1 over classes (macro F1).
    """
    eps = 1e-9
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def _tune_threshold_and_scale(
    infer_layer, train_df: pd.DataFrame, train_img_dir: str, n_val: int = 512
):
    """
    Select threshold and whether to scale by 255 using a small validation subset.
    Also ensures y_true columns match model output dimension to avoid shape bugs.
    """
    n_val = min(n_val, len(train_df))
    val_df = train_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)

    preds_01 = []
    preds_255 = []
    batch_size = 16

    for start in range(0, n_val, batch_size):
        batch = val_df.iloc[start : start + batch_size]
        imgs = []
        for fn in batch["image"].tolist():
            img_path = os.path.join(train_img_dir, fn)
            imgs.append(_load_and_preprocess_image(img_path))
        x = tf.stack(imgs, axis=0)  # [0,1]

        out_01 = _extract_pred_tensor(_call_infer(infer_layer, x))
        out_01 = _maybe_apply_sigmoid(out_01)
        out_255 = _extract_pred_tensor(_call_infer(infer_layer, x * 255.0))
        out_255 = _maybe_apply_sigmoid(out_255)

        preds_01.append(out_01.numpy())
        preds_255.append(out_255.numpy())

    p01 = np.concatenate(preds_01, axis=0)
    p255 = np.concatenate(preds_255, axis=0)

    n_out = int(p01.shape[1])
    if n_out != len(dataset_labels):
        print(
            f"Warning: model outputs {n_out} classes, but train.csv provides {len(dataset_labels)} labels. "
            "Aligning y_true to the first n_out columns for threshold tuning and submission mapping."
        )
        used_labels = dataset_labels[:n_out]
    else:
        used_labels = dataset_labels

    y_true = (
        val_df["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=used_labels, fill_value=0)
        .values.astype(np.int32)
    )

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    best = {"f1": -1.0, "thr": 0.5, "scale": "255", "labels": used_labels}
    for scale_name, p in (("01", p01), ("255", p255)):
        for thr in thresholds:
            y_pred = (p > thr).astype(np.int32)
            f1 = _mean_f1(y_true, y_pred)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "thr": float(thr),
                    "scale": scale_name,
                    "labels": used_labels,
                }

    print(
        f"Validation tuning on n={n_val}: best_f1={best['f1']:.5f} thr={best['thr']} scale={best['scale']}"
    )
    return best["thr"], best["scale"], best["labels"]




## === cell 4
if __name__ == "__main__":
    test_dir_resolved = _resolve_test_dir(test_dir)
    print("Resolved test_dir:", test_dir_resolved)

    resolved_model_dir = _find_any_savedmodel(model_dir)
    if resolved_model_dir:
        print("Using SavedModel at:", resolved_model_dir)
        infer_layer = _wrap_savedmodel_as_layer(resolved_model_dir)
    else:
        infer_layer = None
        print(
            "Warning: No SavedModel found under the expected paths. "
            "Will generate a valid submission with a safe baseline ('healthy')."
        )

    images_path_list = _list_image_files(test_dir_resolved)
    if len(images_path_list) == 0:
        raise RuntimeError(f"No image files found under: {test_dir_resolved}")

    if infer_layer is not None:
        train_img_dir = _resolve_train_dir()
        thr, scale_mode, used_labels = _tune_threshold_and_scale(
            infer_layer, data_set, train_img_dir, n_val=512
        )
    else:
        thr, scale_mode, used_labels = 0.7, "255", dataset_labels

    def test_on_sub(index):
        img_path = os.path.join(test_dir_resolved, images_path_list[index])
        image = _load_and_preprocess_image(img_path)
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(image, axis=0)

    values = []

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        if infer_layer is None:
            classes_img = "healthy"
            values.append([name, classes_img])
            continue

        images_in = images * 255.0 if scale_mode == "255" else images

        pred_out = _call_infer(infer_layer, images_in)
        test_values = _extract_pred_tensor(pred_out)
        test_values = _maybe_apply_sigmoid(test_values)

        n_out = int(test_values.shape[1])
        label_names = used_labels[:n_out]

        index_values = [
            i
            for i, v in enumerate(test_values[0].numpy().tolist()[: len(label_names)])
            if v > thr
        ]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(label_names[i]) for i in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape)
    print("Saved to:", out_path)
    print(csv_pd.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4280774227.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     test_dir_resolved = _resolve_test_dir(test_dir)
      3     print("Resolved test_dir:", test_dir_resolved)
      4 
      5     resolved_model_dir = _find_any_savedmodel(model_dir)

NameError: name '_resolve_test_dir' is not defined
