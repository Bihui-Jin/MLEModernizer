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

0.5740535549399802

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
from pathlib import Path

import pandas as pd
import tensorflow as tf
import numpy as np



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = (
    "../input/model-30/epoch-30"  # may not exist; we will resolve robustly below
)

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()




## === cell 2
def _is_saved_model_dir(p: str) -> bool:
    p = Path(p)
    return p.is_dir() and (p / "saved_model.pb").exists()


def _find_any_saved_model_under_input() -> str:
    """
    Bugfix: the notebook hard-coded a model path that may not exist in this Kaggle environment.
    We search ../input for a folder containing saved_model.pb and use the first match.
    """
    input_root = Path("../input")
    if not input_root.exists():
        raise FileNotFoundError("Expected ../input to exist in Kaggle environment.")

    preferred = [
        input_root / "model-30" / "epoch-30",
        input_root / "model-30",
    ]
    for p in preferred:
        if _is_saved_model_dir(p):
            return str(p)

    for pb in input_root.rglob("saved_model.pb"):
        sm_dir = pb.parent
        return str(sm_dir)

    raise FileNotFoundError(
        "Could not find any TensorFlow SavedModel under ../input (no saved_model.pb found)."
    )


def _load_infer_layer(saved_model_path: str) -> tf.keras.layers.Layer:
    """
    Bugfix: robustly create a TFSMLayer. If 'serving_default' doesn't exist,
    fall back to the first available signature.
    """
    try:
        return tf.keras.layers.TFSMLayer(
            saved_model_path, call_endpoint="serving_default"
        )
    except Exception:
        loaded = tf.saved_model.load(saved_model_path)
        sigs = list(getattr(loaded, "signatures", {}).keys())
        if not sigs:
            raise RuntimeError(
                f"No signatures found in SavedModel at {saved_model_path}; cannot create TFSMLayer."
            )
        return tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint=sigs[0])




## === cell 3
if __name__ == "__main__":
    if not _is_saved_model_dir(model_dir):
        resolved = _find_any_saved_model_under_input()
        print(
            f"WARNING: model_dir not found/invalid: {model_dir}\nUsing SavedModel at: {resolved}"
        )
        model_dir = resolved
    else:
        print(f"Using SavedModel at: {model_dir}")

    infer_layer = _load_infer_layer(model_dir)

    images_path_list = sorted(list(os.listdir(test_dir)))
    if len(images_path_list) == 0:
        raise RuntimeError(f"No test images found in: {test_dir}")

    def test_on_sub(index: int):
        img_name = images_path_list[index]
        img_path = os.path.join(test_dir, img_name)

        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        return img_name, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)

        out = infer_layer(images, training=False)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        out = tf.convert_to_tensor(out)

        probs = out[0].numpy().astype(np.float32)
        index_values = [i for i, v in enumerate(probs) if v > 0.6]

        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join([str(classes[i]) for i in index_values])

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape)
    print(csv_pd.head())
    print("Saved to:", out_path)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3251890375.py in <cell line: 0>()
      2     # Resolve model path if the provided one doesn't exist (fixes the reported OSError).
      3     if not _is_saved_model_dir(model_dir):
----> 4         resolved = _find_any_saved_model_under_input()
      5         print(
      6             f"WARNING: model_dir not found/invalid: {model_dir}\nUsing SavedModel at: {resolved}"

/tmp/ipykernel_55/3593687759.py in _find_any_saved_model_under_input()
     27         return str(sm_dir)
     28 
---> 29     raise FileNotFoundError(
     30         "Could not find any TensorFlow SavedModel under ../input (no saved_model.pb found)."
     31     )

FileNotFoundError: Could not find any TensorFlow SavedModel under ../input (no saved_model.pb found).
