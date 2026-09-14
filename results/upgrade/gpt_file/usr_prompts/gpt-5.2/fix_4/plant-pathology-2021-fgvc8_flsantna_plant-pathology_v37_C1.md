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

0.8117451523545736

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e6/epoch-6"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

train_csv_candidates = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train.csv",
]
train_csv_path = None
for p in train_csv_candidates:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(
        f"Could not locate train.csv. Tried: {train_csv_candidates}"
    )

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()


def _find_savedmodel_dir(base_path: str) -> str:
    """Locate a SavedModel directory for TFSMLayer, searching nested subdirs if needed."""
    if base_path is None:
        raise FileNotFoundError("SavedModel base_path is None")

    if tf.io.gfile.exists(
        os.path.join(base_path, "saved_model.pb")
    ) or tf.io.gfile.exists(os.path.join(base_path, "saved_model.pbtxt")):
        return base_path

    if tf.io.gfile.exists(base_path) and tf.io.gfile.isdir(base_path):
        for ch in tf.io.gfile.listdir(base_path):
            cand = os.path.join(base_path, ch)
            if tf.io.gfile.isdir(cand) and (
                tf.io.gfile.exists(os.path.join(cand, "saved_model.pb"))
                or tf.io.gfile.exists(os.path.join(cand, "saved_model.pbtxt"))
            ):
                return cand

    local_glob = glob.glob(
        os.path.join(base_path, "**", "saved_model.pb"), recursive=True
    )
    if local_glob:
        return os.path.dirname(local_glob[0])

    raise FileNotFoundError(
        f"Could not find SavedModel under: {base_path}. Expected saved_model.pb/pbtxt."
    )


def _resolve_weights_path(path: str) -> str:
    """Resolve a usable path for model.load_weights()."""
    if path is None:
        raise FileNotFoundError("Weights path is None")

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidates = sorted(
            glob.glob(os.path.join(path, "*.weights.h5"))
            + glob.glob(os.path.join(path, "*.h5"))
        )
        if candidates:
            return candidates[-1]

        ckpt_state = os.path.join(path, "checkpoint")
        if os.path.exists(ckpt_state):
            try:
                with open(ckpt_state, "r") as f:
                    for line in f:
                        if "model_checkpoint_path" in line:
                            prefix = line.split('"')[1]
                            return os.path.join(path, prefix)
            except Exception:
                pass

        return path

    return path


def _find_any_savedmodel_under_inputs(preferred_base: str) -> str:
    """
    Fix: if the external backbone dataset isn't attached, try to find ANY SavedModel
    under /kaggle/input (or ../input) to keep the pipeline runnable.
    """
    if preferred_base and os.path.exists(preferred_base):
        try:
            return _find_savedmodel_dir(preferred_base)
        except FileNotFoundError:
            pass

    search_roots = ["../input", "/kaggle/input"]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        dataset_dirs = sorted(glob.glob(os.path.join(root, "*")))
        for d in dataset_dirs:
            if not os.path.isdir(d):
                continue
            for depth_pat in [
                os.path.join(d, "saved_model.pb"),
                os.path.join(d, "*", "saved_model.pb"),
                os.path.join(d, "*", "*", "saved_model.pb"),
            ]:
                hits = glob.glob(depth_pat)
                if hits:
                    return os.path.dirname(hits[0])

    raise FileNotFoundError(
        "Could not locate any SavedModel under Kaggle inputs. "
        "Attach the EfficientNetB7 SavedModel dataset used by this notebook."
    )


efficientB7_sm = _find_any_savedmodel_under_inputs(efficientB7)
model_weights_path = _resolve_weights_path(model_dir)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1658397052.py in <cell line: 0>()
    132 
    133 # Fix: robustly locate backbone SavedModel even if the originally referenced dataset isn't attached.
--> 134 efficientB7_sm = _find_any_savedmodel_under_inputs(efficientB7)
    135 model_weights_path = _resolve_weights_path(model_dir)
    136 

/tmp/ipykernel_11/1658397052.py in _find_any_savedmodel_under_inputs(preferred_base)
    125                     return os.path.dirname(hits[0])
    126 
--> 127     raise FileNotFoundError(
    128         "Could not locate any SavedModel under Kaggle inputs. "
    129         "Attach the EfficientNetB7 SavedModel dataset used by this notebook."

FileNotFoundError: Could not locate any SavedModel under Kaggle inputs. Attach the EfficientNetB7 SavedModel dataset used by this notebook.

## === cell 2
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

        self.model_backbone = tf.keras.layers.TFSMLayer(
            efficientB7_sm,
            call_endpoint="serving_default",
        )

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)

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
if __name__ == "__main__":
    test_dir_candidates = [
        test_dir,
        "../input/test_images/",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images/",
    ]
    for td in test_dir_candidates:
        if os.path.isdir(td):
            test_dir = td
            break
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Could not locate test_images/. Tried: {test_dir_candidates}"
        )

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    model.load_weights(model_weights_path)

    images_path_list = sorted(list(os.listdir(test_dir)))

    def test_on_sub(index):
        img_path = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_image(
            contents=input_img, channels=3, dtype=tf.dtypes.float32
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)
        images = images * 255.0
        test_values = model.call(images)

        n_out = int(test_values.shape[1])
        n_cls = len(classes)
        n_use = min(n_out, n_cls)

        index_values = [
            j for j, v in enumerate(test_values[0, :n_use]) if float(v) > 0.7
        ]

        classes_img = ""
        for j in index_values:
            classes_img = str(classes[j]) + " " + classes_img

        classes_img = classes_img.strip()
        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2163952074.py in <cell line: 0>()
     15         )
     16 
---> 17     model = MultiLabel()
     18     model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])
     19 

/tmp/ipykernel_11/573392781.py in __init__(self, *args, **kwargs)
     17         # Core logic preserved: TFSMLayer backbone + conv heads.
     18         self.model_backbone = tf.keras.layers.TFSMLayer(
---> 19             efficientB7_sm,
     20             call_endpoint="serving_default",
     21         )

NameError: name 'efficientB7_sm' is not defined
