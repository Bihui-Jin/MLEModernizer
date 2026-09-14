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

0.6584302862419203

# 6. Current score

0.11004

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11004) has done: 'I fix the protobuf/TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the test image reading failure by filtering `test_images` to include only actual image files (your current `os.listdir` is picking up a nested `test_images/` directory and passing it to `tf.io.read_file`). Finally, I keep your model/inference logic intact but ensure deterministic, correctly ordered submission rows and always write a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.11004) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow-related import happens*, and by also setting a safe protobuf runtime version flag that prevents the `MessageFactory.GetPrototype` path from being used. Then we keep your inference/training logic intact, but correct a label-ordering bug in submission generation (you were reversing class names, which hurts F1 significantly). Finally, we make the submission formatting strictly match the competition spec (`image,labels` exactly) and ensure deterministic ordering and full coverage of the provided sample test set files.'
- What this solution (achieved 0.11004) has done: 'We fix the immediate crash in cell 1 by moving the protobuf/TensorFlow compatibility environment variables to the very top of the script (before any import that can pull in `google.protobuf`), and we add a small safety fallback to force the pure-Python protobuf implementation at runtime if the C++ one is already loaded. Then, to improve your score toward the target while preserving core inference logic, we stop thresholding with a hard-coded `0.6` and instead derive per-class thresholds from the training label prevalence (a standard calibration step for multilabel mean-F1) while keeping the same sigmoid outputs and prediction flow. Finally, we ensure the submission is aligned to `sample_submission.csv` image order (so no missing/extra rows) and always writes a valid `submission.csv` with exactly `image,labels`.'
- What this solution (achieved 0.11004) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top (before any import that could load `google.protobuf`) and adding a safe runtime fallback in case protobuf was already imported. Then I keep your model and inference logic the same, but make the SavedModel output extraction more robust to avoid silent shape/key mismatches that can tank F1. Finally, I keep the submission generation aligned to `sample_submission.csv` order and ensure the output is always written as `submission.csv` with exactly `image,labels` columns.'
- What this solution (achieved 0.11004) has done: 'We fix the TensorFlow/protobuf import crash by forcing a protobuf version that is compatible with TF 2.18 in this environment, and we do it before importing anything that can load protobuf/TensorFlow. This is a pure runtime-stability fix (score-neutral) that unblocks the rest of your pipeline. Then we keep your model/inference logic intact, but also force the submission to always exactly follow `sample_submission.csv` image order to avoid any hidden-test ordering/coverage issues. No architectural/training/loop changes are introduced beyond what’s required to run end-to-end and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")


def _ensure_tf_compatible_protobuf():
    import importlib
    import subprocess
    import sys

    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(ver):
        try:
            return int(str(ver).split(".", 1)[0])
        except Exception:
            return None

    if pb_ver is None or (_major(pb_ver) is not None and _major(pb_ver) >= 5):
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "protobuf<5",
        ]
        subprocess.check_call(cmd)
        importlib.invalidate_caches()


_ensure_tf_compatible_protobuf()

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Python:", sys.version.split()[0])
print("TF:", tf.__version__)



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-complete/model_complete_with_augEpoch:20"

image_dims = (300, 300, 3)


def _resolve_competition_root() -> str:
    candidates = [
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    for base in ["../input", "/kaggle/input"]:
        basep = Path(base)
        if basep.exists():
            for p in basep.rglob("train.csv"):
                if p.name == "train.csv" and "plant-pathology-2021-fgvc8" in str(
                    p.parent
                ):
                    return str(p.parent)
    raise FileNotFoundError(
        "Could not find competition train.csv under ../input or /kaggle/input"
    )


COMP_ROOT = _resolve_competition_root()
train_csv_path = os.path.join(COMP_ROOT, "train.csv")
sample_sub_path = os.path.join(COMP_ROOT, "sample_submission.csv")
train_images_dir = os.path.join(COMP_ROOT, "train_images")
test_dir = os.path.join(COMP_ROOT, "test_images")

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("COMP_ROOT:", COMP_ROOT)
print("num_classes:", len(dataset_labels))
print(
    "train rows:", len(data_set), "test images dir entries:", len(os.listdir(test_dir))
)




## === cell 2
def _find_saved_model_dir(path: str) -> str:
    """
    The provided model_dir may not exist in this Kaggle environment.
    If it exists, find the directory containing saved_model.pb/.pbtxt.
    """
    path = os.path.expanduser(path)

    def is_saved_model_dir(d: str) -> bool:
        return os.path.isfile(os.path.join(d, "saved_model.pb")) or os.path.isfile(
            os.path.join(d, "saved_model.pbtxt")
        )

    if os.path.isdir(path) and is_saved_model_dir(path):
        return path

    if os.path.isfile(path):
        parent = os.path.dirname(path)
        if is_saved_model_dir(parent):
            return parent

    candidates = []
    search_roots = []
    if os.path.isdir(path):
        search_roots.append(path)
    parent = os.path.dirname(path)
    if parent and os.path.isdir(parent):
        search_roots.append(parent)

    if os.path.isdir("../input/model-complete"):
        search_roots.append("../input/model-complete")

    seen = set()
    for root in search_roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root):
            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                candidates.append(dirpath)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a SavedModel directory starting from: {path} "
            f"(also searched parent and ../input/model-complete)."
        )

    candidates = sorted(candidates, key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0]


def _extract_pred_tensor(pred):
    """Robust extraction of output tensor from TFSMLayer output (dict/nested structures)."""
    if isinstance(pred, dict):
        for k in ("outputs", "predictions", "probs", "probabilities", "logits"):
            if k in pred:
                pred = pred[k]
                break
        else:
            pred = next(iter(pred.values()))
        if isinstance(pred, dict):
            pred = next(iter(pred.values()))
        return pred
    return pred




## === cell 3
def _make_model(num_classes: int):
    inputs = tf.keras.Input(shape=image_dims, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.cast(img, tf.float32)
    return img


def _build_train_dataset(df: pd.DataFrame, label_matrix: np.ndarray, batch_size=16):
    paths = df["image"].apply(lambda x: os.path.join(train_images_dir, x)).values
    y = label_matrix.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    def _map(p, y):
        img = _load_image(p)
        return img, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _get_inference_callable():
    try:
        resolved_model_dir = _find_saved_model_dir(model_dir)
        try:
            layer = tf.keras.layers.TFSMLayer(
                resolved_model_dir, call_endpoint="serving_default"
            )
        except Exception:
            layer = tf.keras.layers.TFSMLayer(resolved_model_dir, call_endpoint="call")

        @tf.function(reduce_retracing=True)
        def _predict_fn(batch_images):
            pred = layer(batch_images)
            pred = _extract_pred_tensor(pred)
            pred = tf.convert_to_tensor(pred)
            pred = tf.reshape(pred, [tf.shape(pred)[0], -1])
            return pred

        print("Using SavedModel from:", resolved_model_dir)
        return _predict_fn
    except FileNotFoundError:
        print(
            "SavedModel not found; training fallback model on train_images for inference..."
        )

        label_matrix = one_hot.values
        ds_train = _build_train_dataset(data_set, label_matrix, batch_size=16)

        model = _make_model(num_classes=len(dataset_labels))
        model.fit(ds_train, epochs=2, verbose=2)

        @tf.function(reduce_retracing=True)
        def _predict_fn(batch_images):
            pred = model(batch_images, training=False)
            pred = tf.reshape(pred, [tf.shape(pred)[0], -1])
            return pred

        return _predict_fn




## === cell 4
def _compute_class_thresholds_from_prevalence(one_hot_df: pd.DataFrame) -> np.ndarray:
    """
    Calibration consistent with mean-F1 for multilabel:
    per-class threshold based on train prevalence (keeps sigmoid outputs/inference same).
    """
    prev = one_hot_df.mean(axis=0).values.astype(np.float32)
    thr = 0.75 - 0.9 * prev
    thr = np.clip(thr, 0.20, 0.75)
    return thr


if __name__ == "__main__":
    predict_fn = _get_inference_callable()

    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        images_path_list = sample_df["image"].astype(str).tolist()
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        images_path_list = sorted(
            [
                fn
                for fn in os.listdir(test_dir)
                if os.path.isfile(os.path.join(test_dir, fn))
                and fn.lower().endswith(exts)
            ]
        )

    if len(images_path_list) == 0:
        raise RuntimeError(f"No image files found under test_dir={test_dir}")

    test_paths = [os.path.join(test_dir, fn) for fn in images_path_list]

    BATCH_SIZE = 64

    def _load_test_image(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(
            img_bytes, channels=3, dtype=tf.dtypes.float32, expand_animations=False
        )
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        return img

    ds_test = tf.data.Dataset.from_tensor_slices((images_path_list, test_paths))
    ds_test = ds_test.map(
        lambda name, p: (name, _load_test_image(p)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    labels_arr = np.array(dataset_labels, dtype=object)
    class_thresholds = _compute_class_thresholds_from_prevalence(one_hot).reshape(1, -1)

    values = []
    for batch_names, batch_images in ds_test:
        preds = predict_fn(batch_images)
        preds_np = tf.convert_to_tensor(preds).numpy()  # [B, C]

        if preds_np.shape[1] != len(dataset_labels):
            raise RuntimeError(
                f"Predictions have C={preds_np.shape[1]} but expected {len(dataset_labels)} classes."
            )

        mask = preds_np > class_thresholds  # [B,C] > [1,C]

        for i in range(mask.shape[0]):
            idxs = np.flatnonzero(mask[i])
            if idxs.size:
                classes_img = " ".join(labels_arr[idxs].tolist())
            else:
                classes_img = ""
            values.append([batch_names[i].numpy().decode("utf-8"), classes_img])

    pred_df = pd.DataFrame(values, columns=["image", "labels"])

    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        sub_df = sample_df[["image"]].merge(pred_df, on="image", how="left")
        sub_df["labels"] = sub_df["labels"].fillna("")
        sub_df = sub_df[["image", "labels"]]
    else:
        sub_df = pred_df[["image", "labels"]]

    csv_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(csv_path, index=False)
    print(f"Wrote submission to: {csv_path} (rows={len(sub_df)})")
    print(sub_df.head())
