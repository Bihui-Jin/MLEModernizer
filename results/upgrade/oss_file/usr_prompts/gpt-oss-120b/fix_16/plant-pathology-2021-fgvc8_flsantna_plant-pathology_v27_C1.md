# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

try:
    from PIL import Image
except Exception:
    Image = None


## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/modeleffbe62/epoch-6"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]

TOP_K = 3
most_common_labels = (
    df_labels.str.split().explode().value_counts().index.tolist()[:TOP_K]
)
fallback_common_str = " ".join(most_common_labels)

PREDICTION_THRESHOLD = 0.05

one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()


## === cell 2
if __name__ == "__main__":
    model = None
    use_fallback = False

    if tf is not None:
        try:
            model = tf.saved_model.load(model_dir)
        except Exception as e_saved:
            print("tf.saved_model.load failed:", e_saved)
            try:
                model = tf.keras.models.load_model(model_dir)
                print("Loaded model via tf.keras.models.load_model")
            except Exception as e_keras:
                print("tf.keras.models.load_model failed:", e_keras)
                use_fallback = True
        if model is None:
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
        print(f"Using fallback labels based on common classes: {fallback_common_str}")

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
                    classes_img = fallback_common_str
            except Exception as e_pred:
                print(f"Prediction failed for {img_name}: {e_pred}")
                classes_img = fallback_common_str
        else:
            heuristic_labels = list(most_common_labels)  # start with common labels
            if Image is not None:
                try:
                    with Image.open(os.path.join(test_dir, img_name)) as im:
                        im = im.convert("RGB")
                        r, g, b = im.split()
                        green_mean = sum(g.getdata()) / (im.width * im.height)
                        if green_mean > 100:  # likely a healthy leaf
                            heuristic_labels.append("healthy")
                        else:
                            heuristic_labels.append("complex")
                except Exception as e_img:
                    pass
            else:
                pass
            classes_img = " ".join(sorted(set(heuristic_labels)))

        values.append([img_name, classes_img])

    submission_df = pd.DataFrame(values, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
