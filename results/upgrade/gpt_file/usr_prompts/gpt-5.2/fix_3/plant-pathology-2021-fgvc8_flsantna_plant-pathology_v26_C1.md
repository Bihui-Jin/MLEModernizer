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

0.7468698060941834

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import TFSMLayer

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_root = "../input/model-effb7e6"  # keep original source; we'll auto-resolve the actual SavedModel folder

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## === cell 2
def _find_saved_model_dir(root_dir: str) -> str:
    """
    BUGFIX: The provided model path may not directly be a SavedModel directory.
    TFSMLayer expects a directory containing saved_model.pb (or .pbtxt).
    We'll search root_dir recursively and pick the most likely candidate.
    """
    root_dir = os.path.abspath(root_dir)
    candidates = []

    for fn in ("saved_model.pb", "saved_model.pbtxt"):
        if os.path.exists(os.path.join(root_dir, fn)):
            return root_dir

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
            candidates.append(dirpath)

    if not candidates:
        raise FileNotFoundError(
            f"No TensorFlow SavedModel found under: {root_dir}. "
            f"Expected a directory containing saved_model.pb"
        )

    def score(p):
        rel = os.path.relpath(p, root_dir)
        depth = rel.count(os.sep)
        looks_epoch = 0 if ("epoch" in rel.lower() or "export" in rel.lower()) else 1
        return (depth, looks_epoch, rel)

    candidates = sorted(set(candidates), key=score)
    return candidates[0]


def _extract_tensor(out):
    if isinstance(out, dict):
        for k in (
            "outputs",
            "output_0",
            "predictions",
            "logits",
            "dense",
            "activation",
        ):
            if k in out:
                return out[k]
        first_key = sorted(out.keys())[0]
        return out[first_key]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _load_infer_layer(saved_model_dir: str) -> TFSMLayer:
    try:
        return TFSMLayer(saved_model_dir, call_endpoint="serving_default")
    except Exception as e:
        print("Failed with call_endpoint='serving_default' due to:", repr(e))
        return TFSMLayer(saved_model_dir, call_endpoint="serve")


def _read_and_preprocess_image(path: str):
    input_img = tf.io.read_file(path)
    image = tf.io.decode_image(contents=input_img, channels=3, dtype=tf.dtypes.float32)
    image.set_shape([None, None, 3])  # shape hint for resize
    tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    return tf.expand_dims(tensor_image, axis=0)




## === cell 3
if __name__ == "__main__":
    resolved_model_dir = _find_saved_model_dir(model_root)
    print("Resolved SavedModel dir:", resolved_model_dir)

    infer_layer = _load_infer_layer(resolved_model_dir)

    sub_template = pd.read_csv(
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    )
    images_path_list = sub_template["image"].astype(str).tolist()

    values = []
    missing_files = 0

    for name in images_path_list:
        img_path = os.path.join(test_dir, name)
        if not os.path.exists(img_path):
            missing_files += 1
            values.append([name, "healthy"])
            continue

        images = _read_and_preprocess_image(img_path)

        images = images * 255.0

        raw_out = infer_layer(images)
        pred = _extract_tensor(raw_out)
        pred = tf.convert_to_tensor(pred)
        pred_np = pred.numpy()[0]

        if pred_np.min() < -1e-3 or pred_np.max() > 1.0 + 1e-3:
            pred_np = tf.sigmoid(pred).numpy()[0]

        index_values = [i for i, v in enumerate(pred_np) if v > 0.7]
        if len(index_values) == 0:
            index_values = [int(pred_np.argmax())]

        classes_img = " ".join([str(dataset_labels[i]) for i in index_values]).strip()
        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(csv_path, index=False)

    print("Wrote:", csv_path, "rows:", len(csv_pd), "missing_files:", missing_files)
    print(csv_pd.head())
    print(
        "Unique label strings (sample):",
        csv_pd["labels"].value_counts().head(10).to_dict(),
    )

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2120802938.py in <cell line: 0>()
      1 if __name__ == "__main__":
      2     # Resolve model directory robustly
----> 3     resolved_model_dir = _find_saved_model_dir(model_root)
      4     print("Resolved SavedModel dir:", resolved_model_dir)
      5 

/tmp/ipykernel_55/99684404.py in _find_saved_model_dir(root_dir)
     19 
     20     if not candidates:
---> 21         raise FileNotFoundError(
     22             f"No TensorFlow SavedModel found under: {root_dir}. "
     23             f"Expected a directory containing saved_model.pb"

FileNotFoundError: No TensorFlow SavedModel found under: /kaggle/input/model-effb7e6. Expected a directory containing saved_model.pb
