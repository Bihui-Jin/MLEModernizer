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

0.3327

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.35916) has done: 'I fix the runtime error by safely handling the TensorFlow model load (using a fallback when it fails) and ensure the submission file contains exactly the same rows as the official sample submission. To improve the score, I replace the constant “healthy” prediction with a simple heuristic that predicts the most frequent disease(s) from the training data, which is a minimal yet effective change that respects the original workflow.'
- What this solution (achieved 0.35916) has done: 'I fixed the TensorFlow model loading error by replacing the problematic TFSMLayer with a standard tf.keras.models.load_model fallback (and a secondary tf.saved_model.load fallback). The inference code is now wrapped in a try/except so any runtime protobuf issues fall back to the default most‑frequent‑labels prediction. I also lowered the probability threshold to 0.5 to capture more relevant classes, which should raise the F1 score toward the target.'
- What this solution (achieved 0.3327) has done: 'Implemented two key fixes:  
1. Removed the forced pure‑Python protobuf implementation which caused TensorFlow model loading to fail (`MessageFactory` error). This allows the pretrained model to be loaded correctly, enabling real predictions.  
2. Improved the fallback default prediction by selecting all disease labels that appear in at least 10 % of the training set (instead of only the top 2). This yields a more sensible baseline when inference cannot run.  

The script now loads the model reliably and writes a proper `submission.csv` aligned with the sample submission format.'
- What this solution (achieved 0.3327) has done: 'Implemented a protobuf compatibility fix by setting the environment variable before any TensorFlow import, and wrapped the TensorFlow import in a safe try/except block. This prevents the “MessageFactory” AttributeError, allowing the pretrained model to load correctly and improve predictions while preserving the original workflow. The rest of the pipeline remains unchanged, ensuring a proper submission CSV is generated.'
- What this solution (achieved 0.3327) has done: 'Implemented a robust TensorFlow workflow that avoids the protobuf loading issue and adds a lightweight fine‑tuned MobileNetV2 model trained on a small random subset of the training images. This model provides image‑based predictions (instead of the previous constant heuristic), improving the F1‑score toward the target while preserving the original submission format. All other logic and paths remain unchanged.'
- What this solution (achieved 0.3327) has done: 'I added a more robust protobuf setup before any TensorFlow import, increased the MobileNetV2 fine‑tuning epochs to give the model more learning opportunity, and lowered the probability threshold to 0.3 so more relevant classes are kept. These changes keep the original workflow intact while improving prediction quality, moving the F1 score toward the target. The script still falls back to the frequency‑based heuristic if TensorFlow cannot be loaded.'
- What this solution (achieved 0.3327) has done: 'Implemented a protobuf compatibility shim before any TensorFlow imports by aliasing the missing `GetPrototype` method to `GetMessageClass`. This resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error, allowing the MobileNetV2 fine‑tuning to run and produce better predictions. No other logic changes were made, preserving the original workflow while enabling a higher F1 score.'
- What this solution (achieved 0.3327) has done: 'Implemented a robust fix for the protobuf import issue by setting the required environment variables **before** any protobuf or TensorFlow imports, eliminating the faulty `MessageFactory` patch. Adjusted the training regime to run longer (12 epochs) and lowered the probability threshold to 0.2, allowing the fine‑tuned MobileNetV2 model to generate richer predictions while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
output_dir = "./"
test_dir = os.path.join(BASE_INPUT, "test_images")
sample_submission_path = os.path.join(BASE_INPUT, "sample_submission.csv")

image_dims = (300, 300, 3)

data_set = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

label_counts = one_hot.sum()
freq_threshold = 0.10  # fallback frequency threshold
freq_labels = (
    (label_counts / label_counts.sum())
    .loc[lambda s: s >= freq_threshold]
    .index.tolist()
)
default_label_str = " ".join(freq_labels) if freq_labels else "healthy"



## === cell 1
if __name__ == "__main__":
    try:
        import tensorflow as tf

        tf_available = True
    except Exception as import_err:
        print(
            f"TensorFlow import failed ({import_err}); falling back to heuristic predictions."
        )
        tf = None
        tf_available = False

    use_tf = False
    model = None

    if tf_available:
        try:
            subset_df = data_set.sample(n=2000, random_state=42).reset_index(drop=True)
            subset_paths = (
                subset_df["image"]
                .apply(lambda x: os.path.join(BASE_INPUT, "train_images", x))
                .tolist()
            )
            subset_labels = one_hot.loc[subset_df["image"]].values.astype(np.float32)

            AUTOTUNE = tf.data.AUTOTUNE

            def _load_image(path):
                img_bytes = tf.io.read_file(path)
                img = tf.io.decode_jpeg(img_bytes, channels=3)
                img = tf.image.resize(img, [image_dims[0], image_dims[1]])
                img = tf.cast(img, tf.float32) / 255.0
                return img

            train_ds = tf.data.Dataset.from_tensor_slices((subset_paths, subset_labels))
            train_ds = train_ds.map(
                lambda p, l: (_load_image(p), l), num_parallel_calls=AUTOTUNE
            )
            train_ds = train_ds.shuffle(1024).batch(32).prefetch(AUTOTUNE)

            base = tf.keras.applications.MobileNetV2(
                input_shape=image_dims, weights="imagenet", include_top=False
            )
            base.trainable = False

            inputs = tf.keras.Input(shape=image_dims)
            x = base(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(len(dataset_labels), activation="sigmoid")(
                x
            )
            model = tf.keras.Model(inputs, outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
            )

            model.fit(train_ds, epochs=12, verbose=0)
            use_tf = True
            print("Fine‑tuned MobileNetV2 model ready for inference.")
        except Exception as train_err:
            print(
                f"Model training/loading failed ({train_err}); falling back to heuristic predictions."
            )
            model = None
            use_tf = False

    sample_sub = pd.read_csv(sample_submission_path)
    images_list = sample_sub["image"].tolist()

    prob_threshold = 0.2  # lower threshold to capture more relevant classes
    predictions = []

    if use_tf:

        def _load_test_image(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32) / 255.0
            return img

        test_paths = [os.path.join(test_dir, img_name) for img_name in images_list]
        test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        test_ds = test_ds.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(32)

        all_probs = model.predict(test_ds, verbose=0)
        for img_name, probs in zip(images_list, all_probs):
            selected_idx = [i for i, p in enumerate(probs) if p > prob_threshold]
            label_str = " ".join(dataset_labels[i] for i in selected_idx).strip()
            if not label_str:
                label_str = default_label_str
            predictions.append([img_name, label_str])
    else:
        for img_name in images_list:
            predictions.append([img_name, default_label_str])

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
