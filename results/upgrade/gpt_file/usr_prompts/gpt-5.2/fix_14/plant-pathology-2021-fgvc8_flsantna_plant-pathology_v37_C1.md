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
import sys
import warnings

import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

TF_AVAILABLE = False
tf_import_error = None
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
    tf.random.set_seed(42)
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = e
    warnings.warn(
        f"TensorFlow import failed; will run a non-TF fallback to still produce submission.csv.\n"
        f"Import error: {repr(e)}"
    )




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e6/epoch-6"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

train_csv_candidates = [
    "../input/plant-pathology-2021-fgvcvc8/train.csv",  # typo kept as candidate
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

if "healthy" not in dataset_labels:
    dataset_labels = ["healthy"] + dataset_labels

efficientB7_sm = None
model_weights_path = None

if TF_AVAILABLE:

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
        """If the external backbone dataset isn't attached, try to find ANY SavedModel under inputs."""
        if preferred_base and os.path.exists(preferred_base):
            return _find_savedmodel_dir(preferred_base)

        common_roots = [
            "../input/efficientb7",
            "../input",
            "/kaggle/input/efficientb7",
            "/kaggle/input",
        ]
        for root in common_roots:
            if not os.path.exists(root):
                continue
            hits = glob.glob(os.path.join(root, "**", "saved_model.pb"), recursive=True)
            if hits:
                return os.path.dirname(hits[0])

        raise FileNotFoundError("Could not locate any SavedModel under Kaggle inputs.")

    try:
        efficientB7_sm = _find_any_savedmodel_under_inputs(efficientB7)
    except FileNotFoundError:
        efficientB7_sm = None

    try:
        if model_dir and os.path.exists(model_dir):
            model_weights_path = _resolve_weights_path(model_dir)
    except Exception:
        model_weights_path = None




## === cell 2
if TF_AVAILABLE:
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

            if efficientB7_sm is not None:
                self.model_backbone = tf.keras.layers.TFSMLayer(
                    efficientB7_sm,
                    call_endpoint="serving_default",
                )
                self._backbone_is_tfsm = True
            else:
                self.model_backbone = tf.keras.applications.EfficientNetB7(
                    include_top=False,
                    weights="imagenet",
                    input_shape=image_dims,
                )
                self._backbone_is_tfsm = False

            self.model = Sequential()
            self.model.add(InputLayer(input_shape=image_dims))
            self.model.add(self.model_backbone)

            self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
            self.model.add(BatchNormalization(momentum=0.7))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
            self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

            self.model_pred1_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred1_3 = BatchNormalization()
            self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred1_5 = GlobalMaxPool2D()
            self.model_pred1_6 = Dense(units=1, activation="sigmoid")

            self.model_pred2_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred2_3 = BatchNormalization()
            self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred2_5 = GlobalMaxPool2D()
            self.model_pred2_6 = Dense(units=1, activation="sigmoid")

            self.model_pred3_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred3_3 = BatchNormalization()
            self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred3_5 = GlobalMaxPool2D()
            self.model_pred3_6 = Dense(units=1, activation="sigmoid")

            self.model_pred4_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred4_3 = BatchNormalization()
            self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred4_5 = GlobalMaxPool2D()
            self.model_pred4_6 = Dense(units=1, activation="sigmoid")

            self.model_pred5_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred5_3 = BatchNormalization()
            self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred5_5 = GlobalMaxPool2D()
            self.model_pred5_6 = Dense(units=1, activation="sigmoid")

            self.model_pred6_2 = Conv2D(
                filters=1024, kernel_size=(1, 1), padding="same"
            )
            self.model_pred6_3 = BatchNormalization()
            self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
            self.model_pred6_5 = GlobalMaxPool2D()
            self.model_pred6_6 = Dense(units=1, activation="sigmoid")

        def call(self, x, **kwargs):
            y = self.model(x)

            if isinstance(y, dict):
                y = next(iter(y.values()))

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
        "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    ]
    for td in test_dir_candidates:
        if os.path.isdir(td):
            test_dir = td
            break
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Could not locate test_images/. Tried: {test_dir_candidates}"
        )

    sample_sub_candidates = [
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/sample_submission.csv",
        "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
    ]
    sample_sub_path = None
    for p in sample_sub_candidates:
        if os.path.exists(p):
            sample_sub_path = p
            break

    if sample_sub_path is not None:
        sub_df = pd.read_csv(sample_sub_path)
        images_path_list = sub_df["image"].astype(str).tolist()
    else:
        images_path_list = sorted(
            [fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")]
        )

    if not TF_AVAILABLE:
        label_counts = one_hot.sum(axis=0).sort_values(ascending=False)
        top_label = str(label_counts.index[0]) if len(label_counts) else "healthy"

        out_df = pd.DataFrame(
            {"image": images_path_list, "labels": [top_label] * len(images_path_list)}
        )
        out_path = os.path.join(output_dir, "submission.csv")
        out_df.to_csv(out_path, index=False)
        print(
            f"Wrote fallback submission to {out_path} (TF import failed: {repr(tf_import_error)})"
        )
        sys.exit(0)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    full_paths = [os.path.join(test_dir, fn) for fn in images_path_list]

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    if model_weights_path is not None:
        try:
            model.load_weights(model_weights_path)
        except Exception as e:
            print(f"Warning: failed to load weights from {model_weights_path}: {e}")

    classes = dataset_labels
    n_cls = len(classes)

    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(path, fname):
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(img_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        image = image * 255.0
        image.set_shape(image_dims)
        return image, fname

    options = tf.data.Options()
    options.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(
        (full_paths, images_path_list)
    ).with_options(options)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(32, drop_remainder=False).prefetch(AUTOTUNE)

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                shape=[None, image_dims[0], image_dims[1], image_dims[2]],
                dtype=tf.float32,
            )
        ],
    )
    def _predict_step(batch_images):
        return model.call(batch_images, training=False)

    threshold = 0.7
    import numpy as np

    class_arr = np.asarray(list(classes), dtype=object)

    values = []
    values_append = (
        values.append
    )  # local binding to reduce Python overhead in tight loop

    for batch_images, batch_fnames in ds:
        preds = _predict_step(batch_images)  # [B, n_out]
        n_out = tf.shape(preds)[1]

        preds2 = tf.cond(
            n_out < n_cls,
            lambda: tf.pad(
                preds, paddings=[[0, 0], [0, n_cls - n_out]], constant_values=0.0
            ),
            lambda: preds[:, :n_cls],
        )

        mask_np = (preds2 > threshold).numpy()  # [B, n_cls] bool
        fnames_np = batch_fnames.numpy().astype(str)

        for i in range(mask_np.shape[0]):
            idxs = np.flatnonzero(mask_np[i])
            if idxs.size == 0:
                lbl = "healthy"
            else:
                lbl = " ".join(class_arr[idxs].tolist())
            values_append((fnames_np[i], lbl))

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
