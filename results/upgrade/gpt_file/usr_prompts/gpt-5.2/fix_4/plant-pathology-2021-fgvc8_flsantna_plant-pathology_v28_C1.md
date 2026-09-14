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

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I fix the test image listing bug that incorrectly includes the nested `test_images/` directory, which causes `tf.io.read_file` to try to read a directory and crash. I make the file collection robust by filtering to actual image files and by auto-resolving the correct `test_images` path if the provided one is missing or points to an unexpected structure. I also keep the SavedModel search/wrapping logic intact but make it safe in Kaggle by avoiding the protobuf downgrade (which can break TF 2.18) and by ensuring inference input/output handling is stable. These changes are score-neutral except that they allow the notebook to run end-to-end and generate a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_compatible_protobuf():
    """
    Bugfix for Kaggle TF 2.18 + protobuf 6: do NOT attempt to downgrade protobuf.
    The original code tried to force protobuf<5, which is risky/slow and can break
    TensorFlow 2.18. Keeping current environment versions is the safest choice.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        print("protobuf version:", pb_ver)
    except Exception as e:
        print("Warning: protobuf import issue:", repr(e))


_ensure_compatible_protobuf()

import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    Fix: original model_dir may not exist in this Kaggle environment.
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


def _resolve_test_dir(p: str) -> str:
    """
    Bugfix: Kaggle dataset sometimes contains nested test_images/test_images.
    Also, os.listdir may include a 'test_images' subdirectory which is not a file.
    We resolve to a directory that actually contains .jpg files.
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

    def has_jpg(dirpath: str) -> bool:
        if not os.path.isdir(dirpath):
            return False
        for fn in os.listdir(dirpath):
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
        return False

    for c in candidates:
        if has_jpg(c):
            return c

    return p


def _list_image_files(dirpath: str):
    """
    Bugfix: filter out directories (e.g., nested 'test_images') and non-image files.
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




## === cell 3
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

    def test_on_sub(index):
        img_path = os.path.join(test_dir_resolved, images_path_list[index])
        input_img = tf.io.read_file(img_path)
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
    print(csv_pd.head())
