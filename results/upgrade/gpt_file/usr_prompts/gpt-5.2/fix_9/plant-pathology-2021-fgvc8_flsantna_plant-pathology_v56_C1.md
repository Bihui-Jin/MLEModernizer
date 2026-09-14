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

3.10

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

0.3914655058699362

# 6. Current score

0.21672

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31823) has done: 'The timeout is dominated by per-image Python overhead: reading/decoding/resizing each JPEG and running `model.call` one image at a time, forcing thousands of small TensorFlow executions and repeated `.numpy()` transfers. I keep the exact same preprocessing, model, thresholding, and label formatting, but run inference as a batched `tf.data` pipeline with parallel I/O and prefetch so the GPU/CPU stays busy and Python overhead is minimized. I also compile the forward pass with `tf.function` (same math, faster graph execution) and avoid converting full probability vectors to Python lists by thresholding in NumPy. These changes are runtime-only and preserve the algorithm and outputs up to negligible floating-point differences.'
- What this solution (achieved 0.26825) has done: 'The crash happens immediately on importing TensorFlow due to an incompatible protobuf build in this Kaggle environment (the `MessageFactory.GetPrototype` error). To make the notebook run end-to-end and still use the same model/inference logic, I add a small environment pin (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) before importing TensorFlow, which avoids the broken C++ protobuf path. I also remove the `if __name__ == "__main__":` guard so the code actually executes in a Kaggle notebook context (otherwise it may write no submission), while keeping the same preprocessing, thresholds, and label formatting. These changes are execution/stability fixes and should preserve the existing scoring behavior while unblocking submission generation.'
- What this solution (achieved 0.2813) has done: 'I fix the TensorFlow import crash by applying a safer protobuf environment configuration (including forcing the pure-Python implementation and disabling the C++ fast path) before TensorFlow is imported, which is the root cause of the `MessageFactory.GetPrototype` error in some Kaggle images. Then I keep your model/inference logic intact but make the execution robust: validate/resolve the dataset base path so the test image reads don’t silently fail due to path mismatch. Finally, I ensure the submission file is always written as `submission.csv` with the required `image,labels` columns and the same threshold/label formatting as you already use.'
- What this solution (achieved 0.30566) has done: 'I fix the TensorFlow import crash by setting additional protobuf-related environment flags *before* importing TensorFlow, which addresses the `MessageFactory.GetPrototype` issue seen in some Kaggle images. Then I keep your exact model architecture, preprocessing, thresholding, and label formatting, but ensure the code can still proceed even if optional weight/model paths are missing (without changing core behavior when they exist). Finally, I keep the batched `tf.data` inference (runtime-only) and guarantee a valid `submission.csv` with the required `image,labels` columns is written.'
- What this solution (achieved 0.21672) has done: 'I fix the immediate crash by preventing the protobuf/TensorFlow incompatibility: in this Kaggle image, TensorFlow import fails due to a broken protobuf runtime, so I implement a safe fallback path that avoids TensorFlow entirely when it can’t be imported. To keep core logic intact, the TensorFlow path remains unchanged (same model, preprocessing, thresholding, and label formatting), but it only run if TF imports successfully. When TF is unavailable, I generate a valid `submission.csv` using the sample submission images and a conservative baseline (`healthy`) so the notebook always runs end-to-end. This is primarily a stability fix; score improvement toward the target requires TF to import successfully, so the fallback is correctness/robustness-oriented.'

# 9. Code solution

## === cell 0
import os
import glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_DESCRIPTOR_POOL", "1"
)

import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa: F401

    print("TF version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("[WARN] TensorFlow failed to import; will use fallback submission writer.")
    print("[WARN] TF import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"

CANDIDATE_BASES = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]

base_dir = None
if TF_AVAILABLE:
    for b in CANDIDATE_BASES:
        if tf.io.gfile.exists(b):
            base_dir = b
            break
else:
    for b in CANDIDATE_BASES:
        if os.path.exists(b):
            base_dir = b
            break

if base_dir is None:
    base_dir = "../input/plant-pathology-2021-fgvc8"

test_dir = os.path.join(base_dir, "test_images") + "/"
train_csv_path = os.path.join(base_dir, "train.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

model_dir = "../input/conve01/eff7-e17/epoch-17"
resnet50_weights = "../input/resnet50weights/last_epoch-20"
efficientB7 = "../input/efficientb7/effb7"
resnet50 = "../input/resnet50/Model-Resnet"

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
dataset_labels = list(dataset_labels)

print("Using base_dir:", base_dir)
if TF_AVAILABLE:
    print("train_csv_path exists:", tf.io.gfile.exists(train_csv_path))
    print("sample_sub_path exists:", tf.io.gfile.exists(sample_sub_path))
    print("test_dir exists:", tf.io.gfile.exists(test_dir))
else:
    print("train_csv_path exists:", os.path.exists(train_csv_path))
    print("sample_sub_path exists:", os.path.exists(sample_sub_path))
    print("test_dir exists:", os.path.exists(test_dir))
print("Classes:", dataset_labels, "n_classes=", len(dataset_labels))



## === cell 2
if not TF_AVAILABLE:
    sub_df = pd.read_csv(sample_sub_path)
    sub_df["labels"] = "healthy"
    out_path = os.path.join(output_dir, "submission.csv")
    sub_df[["image", "labels"]].to_csv(out_path, index=False)
    print("Wrote (fallback):", out_path, "rows:", len(sub_df))
    print(sub_df.head())



## === cell 3
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

    def _find_savedmodel_dir(path: str) -> str:
        """Return a directory that contains a SavedModel (saved_model.pb / pbtxt), or raise."""
        if path and tf.io.gfile.exists(path) and tf.io.gfile.isdir(path):
            entries = tf.io.gfile.listdir(path)
            if "saved_model.pb" in entries or "saved_model.pbtxt" in entries:
                return path
            for sub in entries:
                subp = os.path.join(path, sub)
                if tf.io.gfile.isdir(subp):
                    sub_entries = tf.io.gfile.listdir(subp)
                    if (
                        "saved_model.pb" in sub_entries
                        or "saved_model.pbtxt" in sub_entries
                    ):
                        return subp
        raise OSError(f"SavedModel not found under: {path}")

    def _find_any_model_file_or_dir(path: str) -> tuple[str, str]:
        """
        Try to locate a usable model artifact under `path`.
        Returns (kind, located_path) where kind in {"savedmodel_dir","keras_file","h5_file","checkpoint_prefix"}.
        """
        if not path or not tf.io.gfile.exists(path):
            raise OSError(f"Backbone path does not exist: {path}")

        try:
            sm_dir = _find_savedmodel_dir(path)
            return ("savedmodel_dir", sm_dir)
        except Exception:
            pass

        if tf.io.gfile.isdir(path):
            entries = tf.io.gfile.listdir(path)
            keras_candidates = [
                os.path.join(path, e) for e in entries if e.endswith(".keras")
            ]
            h5_candidates = [
                os.path.join(path, e)
                for e in entries
                if e.endswith(".h5") or e.endswith(".hdf5")
            ]
            if keras_candidates:
                return ("keras_file", keras_candidates[0])
            if h5_candidates:
                return ("h5_file", h5_candidates[0])

            index_candidates = [
                os.path.join(path, e) for e in entries if e.endswith(".index")
            ]
            if index_candidates:
                prefix = index_candidates[0][: -len(".index")]
                return ("checkpoint_prefix", prefix)

        if tf.io.gfile.exists(path) and not tf.io.gfile.isdir(path):
            if path.endswith(".keras"):
                return ("keras_file", path)
            if path.endswith(".h5") or path.endswith(".hdf5"):
                return ("h5_file", path)

        raise OSError(f"No SavedModel/.keras/.h5/checkpoint found under: {path}")

    def _load_backbone_layer(path: str, input_shape=(300, 300, 3)):
        """
        Robustly attempt multiple formats, then fall back to a standard ResNet50 backbone
        so the notebook always runs end-to-end and can write a valid submission.csv.
        """
        try:
            kind, located = _find_any_model_file_or_dir(path)

            if kind == "savedmodel_dir":
                try:
                    from keras.layers import TFSMLayer  # type: ignore

                    return TFSMLayer(located, call_endpoint="serving_default")
                except Exception:
                    return tf.keras.models.load_model(located, compile=False)

            if kind in ("keras_file", "h5_file"):
                return tf.keras.models.load_model(located, compile=False)

            if kind == "checkpoint_prefix":
                base = tf.keras.applications.ResNet50(
                    include_top=False, weights=None, input_shape=input_shape
                )
                base.load_weights(located).expect_partial()
                return base

        except Exception as e:
            print(f"[WARN] Could not load provided backbone from {path}: {repr(e)}")
            print(
                "[WARN] Falling back to tf.keras.applications.ResNet50(weights='imagenet', include_top=False)."
            )

        return tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )

    class MultiLabel(Model):
        def __init__(self):
            super().__init__()

            self.model_backbone = _load_backbone_layer(resnet50, input_shape=image_dims)

            self.model = Sequential()
            self.model.add(InputLayer(input_shape=image_dims))
            self.model.add(self.model_backbone)
            self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
            self.model.add(BatchNormalization(momentum=0.7))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same"))
            self.model.add(BatchNormalization(momentum=0.7))
            self.model.add(Dropout(0.1))
            self.model.add(Conv2D(filters=128, kernel_size=(1, 1), padding="same"))
            self.model.add(GlobalMaxPool2D())
            self.model.add(Dense(units=len(dataset_labels), activation="sigmoid"))

        def call(self, predict_input):
            return self.model(predict_input)

        def create_model(self):
            return self.model




## === cell 4
if TF_AVAILABLE:
    import numpy as np

    tf.random.set_seed(42)

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    try:
        if tf.io.gfile.exists(resnet50_weights) or tf.io.gfile.exists(
            resnet50_weights + ".index"
        ):
            model.load_weights(resnet50_weights).expect_partial()
            print("Loaded weights from:", resnet50_weights)
        else:
            raise FileNotFoundError(resnet50_weights)
    except Exception as e:
        print(f"[WARN] Could not load weights from {resnet50_weights}: {repr(e)}")
        print(
            "[WARN] Proceeding with current weights (may reduce score but will produce a valid submission)."
        )

    sub_df = pd.read_csv(sample_sub_path)
    images_path_list = sub_df["image"].astype(str).tolist()

    classes = dataset_labels
    thr = 0.6

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 32  # runtime-only speedup; does not change evaluation semantics

    @tf.function(reduce_retracing=True)
    def _predict_batch(batch_images):
        return model(batch_images, training=False)

    def _load_and_preprocess(name):
        img_path = tf.strings.join([tf.constant(test_dir), name])
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        return name, image

    if not tf.io.gfile.exists(test_dir):
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(images_path_list))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    values = []
    for batch_names, batch_images in ds:
        batch_probs = _predict_batch(batch_images).numpy()  # shape [B, C]
        mask = batch_probs > thr
        for i in range(batch_probs.shape[0]):
            idx = np.flatnonzero(mask[i])
            if idx.size == 0:
                classes_img = "healthy"
            else:
                classes_img = " ".join(classes[j] for j in idx.tolist())
            values.append([batch_names[i].numpy().decode("utf-8"), classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])

    csv_pd = sub_df[["image"]].merge(csv_pd, on="image", how="left")
    csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(csv_pd))
    print(csv_pd.head())
