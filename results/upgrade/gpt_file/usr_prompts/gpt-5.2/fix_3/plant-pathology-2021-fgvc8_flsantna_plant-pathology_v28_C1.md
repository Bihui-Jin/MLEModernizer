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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_compatible_protobuf():
    """
    Keep as-is: original intent was to downgrade protobuf if needed.
    In this environment protobuf==6 is installed; TF 2.18 generally supports it,
    but we preserve the original behavior to avoid unexpected incompatibilities.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    if pb_ver is None or (_major(pb_ver) is not None and _major(pb_ver) >= 5):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            if "google.protobuf" in sys.modules:
                del sys.modules["google.protobuf"]
        except Exception as e:
            print(
                "Warning: Could not enforce protobuf<5. Proceeding with current protobuf. Error:",
                repr(e),
            )


_ensure_compatible_protobuf()

import pandas as pd
import tensorflow as tf



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
    Fix: original model_dir doesn't exist in this Kaggle environment.
    We search within ../input for a directory containing saved_model.pb.
    This preserves core inference behavior when a valid SavedModel is available.
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
    """
    Use Keras TFSMLayer when a SavedModel is available.
    """
    return tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")


def _call_infer(infer_layer, images_batch: tf.Tensor) -> tf.Tensor:
    """
    Fix: SavedModel signatures sometimes require a dict input with a specific key.
    We try common patterns while keeping the same model and outputs.
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




## === cell 3
if __name__ == "__main__":
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

    images_path_list = sorted(list(os.listdir(test_dir)))

    def test_on_sub(index):
        input_img = tf.io.read_file(os.path.join(test_dir, images_path_list[index]))
        image = tf.io.decode_image(
            contents=input_img,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    threshold = 0.7

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        images = images * 255.0

        if infer_layer is None:
            classes_img = "healthy"
            values.append([name, classes_img])
            continue

        pred_out = _call_infer(infer_layer, images)

        if isinstance(pred_out, dict):
            if "outputs" in pred_out:
                test_values = pred_out["outputs"]
            elif "predictions" in pred_out:
                test_values = pred_out["predictions"]
            else:
                test_values = next(iter(pred_out.values()))
        else:
            test_values = pred_out

        test_values = tf.convert_to_tensor(test_values)
        if len(test_values.shape) == 1:
            test_values = tf.expand_dims(test_values, axis=0)

        index_values = [
            i for i, v in enumerate(test_values[0].numpy().tolist()) if v > threshold
        ]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(dataset_labels[i]) for i in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape)
    print("Saved to:", out_path)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_55/236243832.py in <cell line: 0>()
     30 
     31     for idx in range(len(images_path_list)):
---> 32         name, images = test_on_sub(index=idx)
     33 
     34         # Keep original scaling (assumes model was trained on 0..255)

/tmp/ipykernel_55/236243832.py in test_on_sub(index)
     15 
     16     def test_on_sub(index):
---> 17         input_img = tf.io.read_file(os.path.join(test_dir, images_path_list[index]))
     18         image = tf.io.decode_image(
     19             contents=input_img,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    581       pass
    582     try:
--> 583       return read_file_eager_fallback(
    584           filename, name=name, ctx=_ctx)
    585     except _core._SymbolicException:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file_eager_fallback(filename, name, ctx)
    604   _inputs_flat = [filename]
    605   _attrs = None
--> 606   _result = _execute.execute(b"ReadFile", 1, inputs=_inputs_flat,
    607                              attrs=_attrs, ctx=ctx, name=name)
    608   if _execute.must_record_gradient():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     51   try:
     52     ctx.ensure_initialized()
---> 53     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     54                                         inputs, attrs, num_outputs)
     55   except core._NotOkStatusException as e:

FailedPreconditionError: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../input/plant-pathology-2021-fgvc8/test_images/test_images; Is a directory [Op:ReadFile]
