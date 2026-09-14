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

0.7860203139427543

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys
import pandas as pd
import tensorflow as tf
import numpy as np

print("TensorFlow:", tf.__version__)
try:
    import google.protobuf

    print("protobuf:", google.protobuf.__version__)
except Exception as e:
    print("Could not import protobuf version:", repr(e))

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-effb7-01/epoch-5/"  # original path (may not exist)

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
test_images_order = sample_sub["image"].tolist()

print("Num classes:", len(dataset_labels))
print("Test images in submission template:", len(test_images_order))



## === cell 2
if __name__ == "__main__":

    def _is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "saved_model.pb"))
            or os.path.isfile(os.path.join(p, "saved_model.pbtxt"))
        )

    def find_model_path(preferred_path: str, root: str = "../input") -> str:
        if preferred_path and _is_savedmodel_dir(preferred_path):
            return preferred_path

        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                candidates.append(dirpath)

            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth >= 4:
                dirnames[:] = []

        if candidates:
            candidates_sorted = sorted(
                candidates, key=lambda p: (("epoch" not in p.lower()), len(p))
            )
            return candidates_sorted[0]

        file_candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".keras") or fn.endswith(".h5"):
                    file_candidates.append(os.path.join(dirpath, fn))
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth >= 4:
                dirnames[:] = []

        if file_candidates:
            return sorted(file_candidates, key=len)[0]

        raise FileNotFoundError(
            f"Could not find a SavedModel directory or .keras/.h5 model under {root}. "
            f"Also checked preferred_path={preferred_path!r}."
        )

    resolved_model_path = find_model_path(model_dir, root="../input")
    print("Resolved model path:", resolved_model_path)

    model = None
    if _is_savedmodel_dir(resolved_model_path):
        try:
            import keras

            TFSMLayer = keras.layers.TFSMLayer
        except Exception:
            from tensorflow import keras

            TFSMLayer = keras.layers.TFSMLayer

        for endpoint in ["serving_default", "serve"]:
            try:
                model = TFSMLayer(resolved_model_path, call_endpoint=endpoint)
                print("Loaded SavedModel via TFSMLayer with endpoint:", endpoint)
                break
            except Exception as e:
                last_err = e
        if model is None:
            raise RuntimeError(f"Failed to load SavedModel via TFSMLayer: {last_err!r}")
    else:
        model = tf.keras.models.load_model(resolved_model_path, compile=False)
        print("Loaded Keras model file:", resolved_model_path)

    def load_image_for_infer(img_path):
        raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = tf.expand_dims(img, axis=0)  # [1,H,W,3]
        img = img * 255.0  # keep original scaling logic
        return img

    values = []
    missing = 0
    for name_jpg in test_images_order:
        img_path = os.path.join(test_dir, name_jpg)
        if not tf.io.gfile.exists(img_path):
            missing += 1
            values.append([name_jpg, "healthy"])
            continue

        images = load_image_for_infer(img_path)

        preds = model(images)
        if isinstance(preds, dict):
            preds = next(iter(preds.values()))
        preds = tf.convert_to_tensor(preds)
        preds_np = preds.numpy()[0]

        index_values = [j for j, v in enumerate(preds_np) if v > 0.8]
        if len(index_values) == 0:
            index_values = [int(np.argmax(preds_np))]

        classes_img = ""
        for j in index_values:
            classes_img = str(dataset_labels[j]) + " " + classes_img
        classes_img = classes_img.strip()

        values.append([name_jpg, classes_img])

    if missing:
        print(
            f"Warning: {missing} test images were missing on disk; filled with 'healthy'."
        )

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])

    csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
    csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print(csv_pd.head())
    print("Wrote:", out_path, "rows:", len(csv_pd))
    assert out_path.endswith(".csv")
    assert list(csv_pd.columns) == ["image", "labels"]
    assert len(csv_pd) == len(sample_sub)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1359045478.py in <cell line: 0>()
     50         )
     51 
---> 52     resolved_model_path = find_model_path(model_dir, root="../input")
     53     print("Resolved model path:", resolved_model_path)
     54 

/tmp/ipykernel_55/1359045478.py in find_model_path(preferred_path, root)
     45             return sorted(file_candidates, key=len)[0]
     46 
---> 47         raise FileNotFoundError(
     48             f"Could not find a SavedModel directory or .keras/.h5 model under {root}. "
     49             f"Also checked preferred_path={preferred_path!r}."

FileNotFoundError: Could not find a SavedModel directory or .keras/.h5 model under ../input. Also checked preferred_path='../input/model-effb7-01/epoch-5/'.
