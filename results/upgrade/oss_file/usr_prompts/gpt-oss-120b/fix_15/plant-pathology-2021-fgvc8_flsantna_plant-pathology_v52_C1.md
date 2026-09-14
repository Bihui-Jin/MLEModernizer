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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from PIL import Image, ImageStat

import concurrent.futures

try:
    import tensorflow as tf

    tf.config.optimizer.set_jit(True)  # enable XLA
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
except Exception as e:
    tf = None
    print(
        f"Warning: TensorFlow could not be imported ({e}). Using fallback predictions."
    )

_test_mean_array = None  # shape (N_test, 3)
_test_fnames = []  # ordered list matching _test_mean_array
_TEST_MEAN_CACHE_PATH = os.path.join(output_dir, "test_means.npz")


def _process_image_mean_generic(dir_path, fname):
    """Read an image, resize to 32×32, and return its mean RGB."""
    try:
        img_path = os.path.join(dir_path, fname)
        img = Image.open(img_path).convert("RGB").resize((32, 32))
        stat = ImageStat.Stat(img)
        mean_rgb = np.array(stat.mean, dtype=np.float32)  # length‑3
        return fname, mean_rgb
    except Exception:
        return None  # signal failure


def _build_test_mean_data():
    """Pre‑compute mean RGB vectors for all test images (cached)."""
    global _test_mean_array, _test_fnames
    if os.path.exists(_TEST_MEAN_CACHE_PATH):
        try:
            cache = np.load(_TEST_MEAN_CACHE_PATH, allow_pickle=True)
            _test_fnames = cache["filenames"].tolist()
            _test_mean_array = cache["means"]
            print(
                f"Loaded cached mean‑RGB vectors for {_test_mean_array.shape[0]} test images."
            )
            return
        except Exception as e:
            print(f"Warning: could not load test cache ({e}). Recomputing.")

    candidates = [
        entry.name
        for entry in os.scandir(test_dir)
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(
            executor.map(
                lambda fn: _process_image_mean_generic(test_dir, fn), candidates
            )
        )

    filtered = [r for r in results if r is not None]
    _test_fnames = [fname for fname, _ in filtered]
    _test_mean_array = np.stack([mean for _, mean in filtered])

    try:
        np.savez_compressed(
            _TEST_MEAN_CACHE_PATH,
            filenames=np.array(_test_fnames, dtype=object),
            means=_test_mean_array,
        )
        print(f"Cached test mean‑RGB vectors to {_TEST_MEAN_CACHE_PATH}.")
    except Exception as e:
        print(f"Warning: could not write test cache ({e}).")




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
model_dir = "../input/conve01/eff7-e16/epoch-16"

image_dims = (300, 300, 3)

train_meta = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = train_meta["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

all_labels = df_labels.str.split(" ").explode()
most_common_label = all_labels.value_counts().idxmax()
print(f"Fallback will predict the most common label: {most_common_label}")

_fallback_built = False
_train_mean_dict = {}
_train_mean_array = None  # NumPy array of shape (N_train, 3)
_train_fnames = []  # Ordered list matching _train_mean_array
_train_label_dict = {}

_MEAN_CACHE_PATH = os.path.join(output_dir, "train_means.npz")


def _process_image_mean(fname):
    """Read an image, resize to 32×32, and return its mean RGB using Pillow."""
    try:
        img_path = os.path.join(train_dir, fname)
        img = Image.open(img_path).convert("RGB").resize((32, 32))
        stat = ImageStat.Stat(img)
        mean_rgb = np.array(stat.mean, dtype=np.float32)  # length‑3
        return fname, mean_rgb
    except Exception:
        return None  # signal failure


def _build_fallback_data():
    """Pre‑compute mean RGB vectors for all training images using parallel I/O.
    Results are cached on disk to avoid re‑reading images on subsequent runs."""
    global _fallback_built, _train_mean_dict, _train_mean_array, _train_fnames, _train_label_dict
    if _fallback_built:
        return

    if os.path.exists(_MEAN_CACHE_PATH):
        try:
            cache = np.load(_MEAN_CACHE_PATH, allow_pickle=True)
            _train_fnames = cache["filenames"].tolist()
            _train_mean_array = cache["means"]
            _train_label_dict = dict(zip(train_meta["image"], train_meta["labels"]))
            _fallback_built = True
            print(
                f"Loaded cached mean‑RGB vectors for {_train_mean_array.shape[0]} images."
            )
            return
        except Exception as e:
            print(f"Warning: could not load cache ({e}). Recomputing.")

    print("Building mean‑RGB lookup for training images (fallback mode)...")
    _train_label_dict = dict(zip(train_meta["image"], train_meta["labels"]))

    candidates = [
        entry.name
        for entry in os.scandir(train_dir)
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_process_image_mean, candidates))

    filtered = [r for r in results if r is not None]
    _train_fnames = [fname for fname, _ in filtered]
    _train_mean_array = np.stack([mean for _, mean in filtered])

    try:
        np.savez_compressed(
            _MEAN_CACHE_PATH,
            filenames=np.array(_train_fnames, dtype=object),
            means=_train_mean_array,
        )
        print(f"Cached mean‑RGB vectors to {_MEAN_CACHE_PATH}.")
    except Exception as e:
        print(f"Warning: could not write cache ({e}).")

    print(f"Computed means for {len(_train_fnames)} training images.")
    _fallback_built = True


def nearest_neighbor_label(test_fname):
    """Return the label of the closest training image by mean RGB (vectorized)."""
    _build_fallback_data()
    try:
        img_path = os.path.join(test_dir, test_fname)
        img = Image.open(img_path).convert("RGB").resize((32, 32))
        arr = np.asarray(img, dtype=np.float32)
        test_mean = arr.mean(axis=(0, 1))  # (3,)
    except Exception:
        return most_common_label

    diffs = _train_mean_array - test_mean  # (N_train, 3)
    dists = np.linalg.norm(diffs, axis=1)  # (N_train,)
    best_idx = np.argmin(dists)
    best_fname = _train_fnames[best_idx]
    return _train_label_dict.get(best_fname, most_common_label)




## === cell 2
if __name__ == "__main__":
    records = []

    if tf is not None:
        try:

            class MultiLabel(tf.keras.Model):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.model_backbone = tf.keras.applications.EfficientNetB7(
                        include_top=False, weights="imagenet", input_shape=image_dims
                    )
                    self.model = tf.keras.Sequential()
                    self.model.add(tf.keras.layers.InputLayer(input_shape=image_dims))
                    self.model.add(self.model_backbone)
                    self.model.add(
                        tf.keras.layers.Conv2D(
                            filters=1024, kernel_size=(1, 1), padding="same"
                        )
                    )
                    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
                    self.model.add(tf.keras.layers.Dropout(0.2))
                    self.model.add(
                        tf.keras.layers.Conv2D(
                            filters=1024, kernel_size=(1, 1), padding="same"
                        )
                    )
                    self.model.add(
                        tf.keras.layers.Conv2D(
                            filters=2400, kernel_size=(1, 1), padding="same"
                        )
                    )
                    self.heads = []
                    for _ in range(6):
                        head = tf.keras.Sequential(
                            [
                                tf.keras.layers.Conv2D(
                                    filters=1024, kernel_size=(1, 1), padding="same"
                                ),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Conv2D(
                                    filters=512, kernel_size=(1, 1), padding="same"
                                ),
                                tf.keras.layers.GlobalMaxPool2D(),
                                tf.keras.layers.Dense(units=1, activation="sigmoid"),
                            ]
                        )
                        self.heads.append(head)

                def call(self, x, **kwargs):
                    y = self.model(x)  # (B, H, W, 2400)
                    splits = tf.split(
                        y, num_or_size_splits=6, axis=-1
                    )  # each (B, H, W, 400)
                    outputs = []
                    for split, head in zip(splits, self.heads):
                        out = head(split)  # (B, 1)
                        outputs.append(out)
                    return tf.concat(outputs, axis=1)

            model = MultiLabel()
            model.build(input_shape=[None, *image_dims])

            try:
                model.load_weights(model_dir)
            except Exception as e:
                print(f"Warning: could not load weights from {model_dir}: {e}")

            image_files = sorted(
                [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
            )
            image_paths = [os.path.join(test_dir, fname) for fname in image_files]

            def _load_and_preprocess(path):
                img_raw = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img_raw, channels=3)
                img = tf.image.resize(img, [image_dims[0], image_dims[1]])
                img = tf.cast(img, tf.float32) * 255.0
                return img

            ds = tf.data.Dataset.from_tensor_slices(image_paths)
            ds = ds.map(
                lambda p: (
                    tf.strings.split(tf.strings.regex_replace(p, r".*/", ""), ".")[0],
                    _load_and_preprocess(p),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            BATCH_SIZE = 128
            ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            for name_batch, img_batch in ds:
                preds_np = model(img_batch, training=False).numpy()
                filenames = [n.decode("utf-8") + ".jpg" for n in name_batch.numpy()]
                for fname, preds_vec in zip(filenames, preds_np):
                    selected = [idx for idx, p in enumerate(preds_vec) if p > 0.7]
                    label_str = " ".join([dataset_labels[idx] for idx in selected])
                    records.append([fname, label_str])

        except Exception as e:
            print(f"Error during TensorFlow inference: {e}")
            print("Falling back to nearest‑neighbor predictions.")
            tf = None  # trigger fallback path

    if tf is None:
        _build_fallback_data()
        _build_test_mean_data()

        diffs = _train_mean_array[:, np.newaxis, :] - _test_mean_array[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (N_train, N_test)

        best_train_idx = np.argmin(dists, axis=0)  # (N_test,)

        for test_fname, train_idx in zip(_test_fnames, best_train_idx):
            train_fname = _train_fnames[train_idx]
            label_str = _train_label_dict.get(train_fname, most_common_label)
            records.append([test_fname, label_str])

    submission = pd.DataFrame(records, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
