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

# 5. Code solution

## === cell 0
import os
import glob
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf



## === cell 1
output_dir = "./"

_CANDIDATE_ROOTS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
]
DATA_ROOT = next((p for p in _CANDIDATE_ROOTS if os.path.isdir(p)), None)
if DATA_ROOT is None:
    matches = glob.glob("../input/**/plant-pathology-2021-fgvc8", recursive=True)
    DATA_ROOT = matches[0] if matches else "../input/plant-pathology-2021-fgvc8"

test_dir = os.path.join(DATA_ROOT, "test_images")

model_dir = "../input/conve01/eff7-e11/epoch-11"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)



## === cell 2
from keras import Sequential, Model
from keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)
from tensorflow import concat


def _find_savedmodel_dir(path: str) -> str:
    """
    Find a directory containing saved_model.pb/pbtxt.
    Returns None if not found (we will fallback to tf.keras.applications).
    """

    def is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "saved_model.pb"))
            or os.path.isfile(os.path.join(p, "saved_model.pbtxt"))
        )

    if path and is_savedmodel_dir(path):
        return path

    if path and os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                return root

    input_roots = ["../input", "/kaggle/input"]
    candidates = []
    for input_root in input_roots:
        if not os.path.isdir(input_root):
            continue
        for root, dirs, files in os.walk(input_root):
            rel = os.path.relpath(root, input_root)
            if rel.count(os.sep) > 6:
                dirs[:] = []
                continue
            if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                low = root.lower()
                score = 0
                if "efficient" in low:
                    score += 3
                if "eff" in low:
                    score += 1
                if "b7" in low:
                    score += 3
                candidates.append((score, root))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[-1][1]

    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        savedmodel_dir = _find_savedmodel_dir(efficientB7)
        if savedmodel_dir is not None:
            from keras.layers import TFSMLayer

            self.model_backbone = TFSMLayer(
                savedmodel_dir, call_endpoint="serving_default"
            )
            backbone_out_is_dict = True
        else:
            base = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            self.model_backbone = base
            backbone_out_is_dict = False

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)

        if backbone_out_is_dict:
            self.model.add(tf.keras.layers.Lambda(lambda d: d[list(d.keys())[0]]))

        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, x, **kwargs):
        y = self.model(x)

        pred1 = y[:, :, :, :400]
        pred2 = y[:, :, :, 400:800]
        pred3 = y[:, :, :, 800:1200]
        pred4 = y[:, :, :, 1200:1600]
        pred5 = y[:, :, :, 1600:2000]
        pred6 = y[:, :, :, 2000:2400]

        pred1 = self.model_pred1_2(pred1)
        pred1 = self.model_pred1_3(pred1)
        pred1 = self.model_pred1_4(pred1)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2)
        pred2 = self.model_pred2_3(pred2)
        pred2 = self.model_pred2_4(pred2)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3)
        pred3 = self.model_pred3_3(pred3)
        pred3 = self.model_pred3_4(pred3)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4)
        pred4 = self.model_pred4_3(pred4)
        pred4 = self.model_pred4_4(pred4)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5)
        pred5 = self.model_pred5_3(pred5)
        pred5 = self.model_pred5_4(pred5)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6)
        pred6 = self.model_pred6_3(pred6)
        pred6 = self.model_pred6_4(pred6)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model




## === cell 3
def _find_weights_path(model_dir_path: str) -> str:
    if os.path.isfile(model_dir_path):
        return model_dir_path

    if not os.path.isdir(model_dir_path):
        base = os.path.basename(model_dir_path.rstrip("/"))
        search_roots = ["../input", "/kaggle/input"]
        found = None
        for sr in search_roots:
            matches = glob.glob(os.path.join(sr, "**", base), recursive=True)
            matches = [m for m in matches if os.path.isdir(m)]
            if matches:
                found = matches[0]
                break
        if found is not None:
            model_dir_path = found
        else:
            raise FileNotFoundError(f"model_dir not found: {model_dir_path}")

    candidates = []
    for fn in os.listdir(model_dir_path):
        full = os.path.join(model_dir_path, fn)
        if os.path.isfile(full):
            candidates.append(fn)

    index_files = sorted([fn for fn in candidates if fn.endswith(".index")])
    if index_files:
        prefix = index_files[-1].replace(".index", "")
        return os.path.join(model_dir_path, prefix)

    h5_files = sorted(
        [fn for fn in candidates if fn.endswith(".weights.h5") or fn.endswith(".h5")]
    )
    if h5_files:
        return os.path.join(model_dir_path, h5_files[-1])

    raise FileNotFoundError(
        f"No recognizable weights found in {model_dir_path}. Contents: {sorted(candidates)[:50]}"
    )


if __name__ == "__main__":
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    model = MultiLabel()

    _ = model(
        tf.zeros([1, image_dims[0], image_dims[1], image_dims[2]], dtype=tf.float32)
    )

    try:
        weights_path = _find_weights_path(model_dir)
        model.load_weights(weights_path)
    except FileNotFoundError as e:
        print(f"WARNING: {e}\nProceeding without external weights.")

    images_path_list = sample_sub["image"].tolist()

    def test_on_sub(index):
        input_img = tf.io.read_file(os.path.join(test_dir, images_path_list[index]))
        image = tf.io.decode_image(
            contents=input_img,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)
        images = images * 255.0  # keep original scaling behavior

        test_values = model.call(images)
        index_values = [
            j for j, v in enumerate(test_values[0].numpy().tolist()) if v > 0.7
        ]

        classes_img_list = [dataset_labels[j] for j in index_values]
        classes_img = " ".join(classes_img_list).strip()

        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print(
        f"Wrote submission to: {out_path}  rows={len(csv_pd)}  cols={list(csv_pd.columns)}"
    )
