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

No external packages required in the script and installed.

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

0.8004801477377672

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd

tf = None




## === cell 1
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (
        Dense,
        BatchNormalization,
        Dropout,
        GlobalMaxPool2D,
        Conv2D,
        InputLayer,
    )
    from tensorflow import concat

    class MultiLabel(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone_input_shape = None
            self.backbone = None
            self.model = Sequential()

        def build(self, input_shape):
            self.backbone_input_shape = input_shape[1:]
            backbone = EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=self.backbone_input_shape,
            )
            self.model.add(InputLayer(input_shape=self.backbone_input_shape))
            self.model.add(backbone)
            self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

            def branch():
                seq = Sequential()
                seq.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
                seq.add(BatchNormalization())
                seq.add(Conv2D(filters=512, kernel_size=(1, 1), padding="same"))
                seq.add(GlobalMaxPool2D())
                seq.add(Dense(units=1, activation="sigmoid"))
                return seq

            self.model_pred1 = branch()
            self.model_pred2 = branch()
            self.model_pred3 = branch()
            self.model_pred4 = branch()
            self.model_pred5 = branch()
            self.model_pred6 = branch()
            super().build(input_shape)

        def call(self, x, **kwargs):
            y = self.model(x)

            pred1 = y[:, :, :, 0:400]
            pred2 = y[:, :, :, 400:800]
            pred3 = y[:, :, :, 800:1200]
            pred4 = y[:, :, :, 1200:1600]
            pred5 = y[:, :, :, 1600:2000]
            pred6 = y[:, :, :, 2000:2400]

            pred1 = self.model_pred1(pred1)
            pred2 = self.model_pred2(pred2)
            pred3 = self.model_pred3(pred3)
            pred4 = self.model_pred4(pred4)
            pred5 = self.model_pred5(pred5)
            pred6 = self.model_pred6(pred6)

            return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

        def create_model(self):
            return self.model

else:

    class MultiLabel:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, input_shape):
            pass

        def __call__(self, x):
            if tf is not None:
                batch = tf.shape(x)[0]
                return tf.zeros((batch, 6), dtype=tf.float32)
            else:
                return np.zeros((1, 6), dtype=np.float32)




## === cell 2
if __name__ == "__main__":
    image_dims = (300, 300, 3)

    possible_base_dirs = [
        os.path.abspath(os.path.join(".", "data", "plant-pathology-2021-fgvc8")),
        os.path.abspath(os.path.join(".", "input", "plant-pathology-2021-fgvc8")),
        os.path.abspath(os.path.join(".", "working", "plant-pathology-2021-fgvc8")),
        "/kaggle/input/plant-pathology-2021-fgvc8",  # <-- new entry
    ]
    base_dir = next((p for p in possible_base_dirs if os.path.isdir(p)), None)
    if base_dir is None:
        raise FileNotFoundError(
            "Could not locate the dataset directory. Checked: "
            + ", ".join(possible_base_dirs)
        )

    test_dir = os.path.join(base_dir, "test_images")
    output_dir = os.path.join(".", "submission")
    os.makedirs(output_dir, exist_ok=True)

    if tf is not None:
        model = MultiLabel()
        model.build(input_shape=[None, *image_dims])
        model_dir = os.path.join("..", "input", "conve01", "eff7-e9", "epoch-9")
        try:
            model.load_weights(model_dir)
        except Exception as e:
            print("Could not load weights:", e)
    else:

        class DummyModel:
            def __call__(self, x):
                return np.zeros((1, 6), dtype=np.float32)

        model = DummyModel()

    images_path_list = sorted(
        [
            os.path.basename(p)
            for p in glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
        ]
    )
    if not images_path_list:
        raise RuntimeError(f"No test images found in {test_dir}")

    def load_image(idx):
        img_path = os.path.join(test_dir, images_path_list[idx])
        if tf is not None:
            raw = tf.io.read_file(img_path)
            img = tf.io.decode_image(contents=raw, channels=3, dtype=tf.dtypes.float32)
            img_resized = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img_tensor = tf.expand_dims(img_resized, axis=0) * 255.0
        else:
            img_tensor = np.expand_dims(
                np.zeros((image_dims[0], image_dims[1], 3)), axis=0
            )
        name = images_path_list[idx]
        return name, img_tensor

    train_csv_path = os.path.join(base_dir, "train.csv")
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)

        label_counts = {}
        for lbls in train_df["labels"].astype(str):
            for lbl in lbls.split():
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total_images = len(train_df)

        freq_threshold = 0.05
        frequent_labels = [
            lbl
            for lbl, cnt in label_counts.items()
            if cnt / total_images >= freq_threshold
        ]

        if not frequent_labels:
            frequent_labels = sorted(
                label_counts.keys(),
                key=lambda x: label_counts[x],
                reverse=True,
            )[:3]

        baseline_pred_str = " ".join(frequent_labels)
    else:
        baseline_pred_str = "healthy"

    values = []
    for i in range(len(images_path_list)):
        name, img_tensor = load_image(i)

        if tf is not None:
            preds = model(img_tensor).numpy().flatten()
            pred_idx = int(np.argmax(preds))
            if "label_list" in locals():
                classes_img = (
                    label_list[pred_idx]
                    if pred_idx < len(label_list)
                    else baseline_pred_str
                )
            else:
                classes_img = baseline_pred_str
        else:
            classes_img = baseline_pred_str

        values.append([name, classes_img.strip()])

    submission_df = pd.DataFrame(values, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
