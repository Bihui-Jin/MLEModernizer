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

0.6627331486611256

# 6. Current score

0.35916

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I make the script robust to missing or misplaced test image paths and simplify the fallback prediction to a single most‑common label (reducing false positives and moving the F1 score toward the target). The core model loading and inference logic stay unchanged, and the script now always writes a valid `submission.csv` file.'
- What this solution (achieved 0.35916) has done: 'I prevent the TensorFlow import error by lazily importing TensorFlow inside a safe try/except block and by adjusting the model‑loading logic to fall back to `tf.keras.models.load_model` if the TF‑SMLayer cannot be used. This removes the immediate crash while keeping the original fallback‑prediction behavior, allowing the script to run end‑to‑end and produce a valid `submission.csv`. The core logic and evaluation semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path

possible_test_dirs = [
    Path("../input/plant-pathology-2021-fgvc8/test_images"),
    Path("/kaggle/input/plant-pathology-2021-fgvc8/test_images"),
    Path("test_images"),
]
test_dir = next((p for p in possible_test_dirs if p.is_dir()), None)
if test_dir is None:
    raise FileNotFoundError("Test images directory not found in expected locations.")
test_dir = test_dir.resolve()

output_dir = "./"
image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.tolist()  # ordered list of class names

label_counts = one_hot.sum()
top_labels = label_counts.sort_values(ascending=False).head(2).index.tolist()
most_common_label = top_labels[0]
second_common_label = top_labels[1] if len(top_labels) > 1 else ""  # safety



## === cell 1
if __name__ == "__main__":
    try:
        import tensorflow as tf
    except Exception as e:
        tf = None
        print(f"Warning: TensorFlow import failed ({e}). Using fallback predictions.")

    model = None
    model_path = "../input/model-to-load/model_complete"

    if tf is not None and os.path.isdir(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded via tf.keras.models.load_model.")
        except Exception as e_keras:
            try:
                model = tf.keras.layers.TFSMLayer(
                    model_path, call_endpoint="serving_default"
                )
                print("Model loaded via TFSMLayer.")
            except Exception as e_smlayer:
                print(
                    f"Warning: could not load model ({e_keras}); "
                    f"TFSMLayer also failed ({e_smlayer}). Using fallback."
                )
                model = None

    images_path_list = sorted(
        [
            p.name
            for p in test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    values = []
    prob_threshold = 0.2

    if model is not None:
        def test_on_sub(index):
            img_path = test_dir / images_path_list[index]
            input_img = tf.io.read_file(str(img_path))
            image = tf.io.decode_image(
                contents=input_img, channels=3, dtype=tf.dtypes.float32
            )
            tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
            name_jpg = images_path_list[index]  # filename only
            return name_jpg, tf.expand_dims(tensor_image, axis=0)

        for i in range(len(images_path_list)):
            name, images = test_on_sub(index=i)

            preds = model(images)
            if isinstance(preds, dict):
                preds = list(preds.values())[0]

            pred_vals = preds.numpy()[0]
            index_values = [
                idx for idx, v in enumerate(pred_vals) if v > prob_threshold
            ]

            classes_img = " ".join(
                dataset_labels[idx] for idx in reversed(index_values)
            ).strip()
            values.append([name, classes_img])
    else:
        fallback_pred = most_common_label
        if second_common_label:
            fallback_pred = f"{most_common_label} {second_common_label}"
        for name in images_path_list:
            values.append([name, fallback_pred])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
