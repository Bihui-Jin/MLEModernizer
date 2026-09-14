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

0.7802770083102512

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I guard the TensorFlow import and the model loading with try/except blocks, falling back to a simple constant‑label prediction when TensorFlow cannot be used or the saved model is missing. This removes the protobuf and missing‑model errors and guarantees that a `submission.csv` file with the correct columns is always written, enabling the notebook to run end‑to‑end.'
- What this solution (achieved 0.28656) has done: 'Implemented a fix for the TensorFlow import error by disabling the protobuf C‑extension before loading TensorFlow. This resolves the `MessageFactory` AttributeError, allowing the saved model to be loaded and predictions to be generated, which should improve the F1‑score toward the target while still retaining the fallback mechanism.'
- What this solution (achieved 0.38173) has done: 'The fix keeps the original workflow but improves the fallback prediction: instead of always using a single most‑common label, we now predict the top 3 most frequent labels (space‑delimited). This modest change preserves all core logic while giving the model more chances to match true multi‑label targets, which should raise the mean F1‑score toward the target value. The rest of the script remains unchanged and still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.3327) has done: 'I increase the label coverage in the fallback by using the top 5 most common labels instead of 3 and lower the confidence threshold for model predictions from 0.7 to 0.5. These small adjustments keep the core workflow unchanged while giving each image more chances to match true multi‑label targets, which should raise the mean F1‑score toward the target.'
- What this solution (achieved 0.30565) has done: 'The changes increase label coverage by expanding the fallback to the top 10 most common classes and lower the prediction confidence threshold to 0.3, giving the model more chances to match true multi‑label targets while keeping the original workflow intact. Minor variable additions replace the hard‑coded values, and comments explain the purpose of each tweak.'
- What this solution (achieved 0.30565) has done: 'The fix reorders imports so the protobuf environment variables are set *before* any library (including pandas) can load protobuf, allowing TensorFlow to import successfully and the model to be used. The rest of the logic is unchanged, preserving the original workflow while ensuring a proper `submission.csv` is written.'
- What this solution (achieved 0.30565) has done: 'Implemented robust environment handling to prevent protobuf C‑extension errors and ensured TensorFlow is imported cleanly. Added a safe reload of any pre‑loaded protobuf modules before importing TensorFlow. Streamlined model loading (preferring `tf.saved_model.load`) and retained the fallback mechanism. The script now reliably writes a correctly formatted `submission.csv` and, when the model loads successfully, uses its predictions to improve the mean F1‑Score toward the target.'
- What this solution (achieved 0.28656) has done: 'I narrow the fallback to the single most frequent label (instead of the top 10) and keep the rest of the workflow unchanged. This reduces false‑positive predictions when the model cannot be loaded, which should raise the mean F1‑Score toward the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_C_EXTENSION"] = "1"

for mod in list(sys.modules.keys()):
    if mod.startswith("google.protobuf"):
        del sys.modules[mod]

try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow import failed:", e)
    tf = None

import pandas as pd



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/modeleffbe62/epoch-6"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]

TOP_K = 1
most_common_labels = (
    df_labels.str.split().explode().value_counts().index.tolist()[:TOP_K]
)
fallback_label_str = " ".join(most_common_labels)

PREDICTION_THRESHOLD = 0.3

one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()



## === cell 2
if __name__ == "__main__":
    model = None
    use_fallback = False

    if tf is not None:
        try:
            model = tf.saved_model.load(model_dir)
        except Exception as e:
            print("SavedModel load failed:", e)
            use_fallback = True
    else:
        use_fallback = True

    images_path_list = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if use_fallback:
        print(f"Using fallback label: {fallback_label_str}")

    values = []
    for idx, img_name in enumerate(images_path_list):
        if not use_fallback:
            img_path = os.path.join(test_dir, img_name)
            try:
                raw = tf.io.read_file(img_path)
                img = tf.io.decode_image(
                    contents=raw, channels=3, dtype=tf.dtypes.float32
                )
                img = tf.image.resize(img, [image_dims[0], image_dims[1]])
                img_tensor = tf.expand_dims(img, axis=0) * 255.0

                preds = model(img_tensor)
                if isinstance(preds, dict):
                    preds = list(preds.values())[0]
                pred_vals = preds[0].numpy()
                index_values = [
                    i for i, v in enumerate(pred_vals) if v > PREDICTION_THRESHOLD
                ]
                classes_img = " ".join(
                    [dataset_labels[i] for i in index_values]
                ).strip()
                if not classes_img:
                    classes_img = fallback_label_str
            except Exception as e_pred:
                print(f"Prediction failed for {img_name}: {e_pred}")
                classes_img = fallback_label_str
        else:
            classes_img = fallback_label_str

        values.append([img_name, classes_img])

    submission_df = pd.DataFrame(values, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
