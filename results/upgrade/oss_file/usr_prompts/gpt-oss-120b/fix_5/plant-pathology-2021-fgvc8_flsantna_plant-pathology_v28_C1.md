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

0.35916

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.35916) has done: 'I fix the runtime error by safely handling the TensorFlow model load (using a fallback when it fails) and ensure the submission file contains exactly the same rows as the official sample submission. To improve the score, I replace the constant “healthy” prediction with a simple heuristic that predicts the most frequent disease(s) from the training data, which is a minimal yet effective change that respects the original workflow.'
- What this solution (achieved 0.35916) has done: 'I fixed the TensorFlow model loading error by replacing the problematic TFSMLayer with a standard tf.keras.models.load_model fallback (and a secondary tf.saved_model.load fallback). The inference code is now wrapped in a try/except so any runtime protobuf issues fall back to the default most‑frequent‑labels prediction. I also lowered the probability threshold to 0.5 to capture more relevant classes, which should raise the F1 score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
output_dir = "./"
test_dir = os.path.join(BASE_INPUT, "test_images")
sample_submission_path = os.path.join(BASE_INPUT, "sample_submission.csv")
model_dir = "/kaggle/input/conv2de01/epoch-1"  # keep original intent

image_dims = (300, 300, 3)

data_set = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

label_counts = one_hot.sum().sort_values(ascending=False)
top_labels = label_counts.head(2).index.tolist()
default_label_str = " ".join(top_labels)


## === cell 2
if __name__ == "__main__":
    import tensorflow as tf

    model = None
    use_tf = False
    try:
        model = tf.keras.models.load_model(model_dir, compile=False)
        use_tf = True
    except Exception as e_keras:
        try:
            model = tf.saved_model.load(model_dir)
            if hasattr(model, "signatures") and "serving_default" in model.signatures:
                model = model.signatures["serving_default"]
            use_tf = True
        except Exception as e_saved:
            print(
                f"Warning: TensorFlow model could not be loaded ({e_keras}; {e_saved}). Using fallback predictions."
            )
            model = None
            use_tf = False

    sample_sub = pd.read_csv(sample_submission_path)
    images_list = sample_sub["image"].tolist()

    predictions = []

    prob_threshold = 0.5

    for img_name in images_list:
        label_str = default_label_str  # default in case anything fails
        if use_tf:
            try:
                img_path = os.path.join(test_dir, img_name)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"{img_path} not found")
                img_bytes = tf.io.read_file(img_path)
                img = tf.io.decode_image(img_bytes, channels=3, dtype=tf.dtypes.float32)
                img_resized = tf.image.resize(img, [image_dims[0], image_dims[1]])
                img_tensor = tf.expand_dims(img_resized, axis=0) * 255.0

                if isinstance(model, tf.keras.Model):
                    logits = model(img_tensor, training=False)
                else:
                    out = model(img_tensor)
                    logits = list(out.values())[0]

                probs = tf.squeeze(logits).numpy()
                selected_idx = [i for i, p in enumerate(probs) if p > prob_threshold]
                label_str = " ".join(dataset_labels[i] for i in selected_idx).strip()
                if not label_str:
                    label_str = default_label_str
            except Exception as infer_err:
                print(f"Inference error for {img_name}: {infer_err}")
                label_str = default_label_str

        predictions.append([img_name, label_str])

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
