# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7735364727608504

# 6. Current score

0.38173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I adjust the fallback baseline to use only the single most frequent label (instead of the top three) which usually matches the dominant “healthy” class and is likely to raise the mean F1‑Score toward the target. I also filter the test directory to keep only jpg files to avoid read errors. The rest of the pipeline and model loading logic remain unchanged.'
- What this solution (achieved 0.38173) has done: 'I increase the fallback baseline to the three most frequent labels (instead of just one) and lower the prediction threshold to 0.5 so that the model, when available, outputs more labels. If the model predicts no label for an image, it fall back to the three‑label baseline for that image. These minimal adjustments should raise the mean F1‑Score toward the target.'
- What this solution (achieved 0.3327) has done: 'I increase the baseline label set from the top 3 to the top 5 most frequent classes and, when a model is available, lower the prediction threshold to 0.3 so more relevant labels are emitted. These modest adjustments should raise recall without drastically hurting precision, moving the mean F1 toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.38173) has done: 'I lower the fallback baseline to the three most frequent labels (which performed better in earlier experiments) and, when a model is available, reduce the prediction threshold to 0.2 so more relevant labels are emitted. This keeps the original pipeline intact while nudging the mean F1‑Score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

base_input = "/kaggle/input/plant-pathology-2021-fgvc8"
output_dir = "./"

test_dir = os.path.join(base_input, "test_images")
model_dir = "/kaggle/input/dense-e1/epoch-1"  # may not exist in this environment
image_dims = (300, 300, 3)

train_path = os.path.join(base_input, "train.csv")
data_set = pd.read_csv(train_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.tolist()



## === cell 1
label_counts = one_hot.sum().sort_values(ascending=False)

top_n = 3
baseline_labels = label_counts.head(top_n).index.tolist()
baseline_str = " ".join(baseline_labels)



## === cell 2
use_model = os.path.isdir(model_dir)

if use_model:
    try:
        import tensorflow as tf

        model_layer = tf.keras.layers.TFSMLayer(
            model_dir, call_endpoint="serving_default"
        )
        images_path_list = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
        )
        values = []
        for img_name in images_path_list:
            img_path = os.path.join(test_dir, img_name)
            raw_img = tf.io.read_file(img_path)
            img = tf.io.decode_image(
                contents=raw_img, channels=3, dtype=tf.dtypes.float32
            )
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.expand_dims(img, axis=0)  # batch dimension
            img = img * 255.0  # match training scale if needed

            preds = model_layer(img)  # (1, num_classes)
            preds = preds.numpy()[0]  # NumPy array

            thresh = 0.2
            index_values = [i for i, v in enumerate(preds) if v > thresh]

            if index_values:
                labels_str = " ".join(dataset_labels[i] for i in index_values)
            else:
                labels_str = baseline_str

            values.append([img_name, labels_str])
    except Exception as e:
        print(f"Model loading failed ({e}); falling back to baseline predictions.")
        use_model = False

if not use_model:
    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    values = [[img_name, baseline_str] for img_name in images_path_list]

submission_df = pd.DataFrame(values, columns=["image", "labels"])
submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
