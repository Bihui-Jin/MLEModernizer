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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.02688

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I remove the dependency on `imgaug` because it’s crashing at import time due to an incompatible protobuf/TensorFlow environment, and replace it with an equivalent augmentation pipeline using TensorFlow ops (same intent: rotations and flips) so `processed_train` is created and training can run. I also fix the zip extraction paths so they reliably extract and list the correct `train/`, `train_cleaned/`, and `test/` folders in `/kaggle/working/` without double slashes. Finally, I keep your autoencoder architecture and training loop intact, but streamline inference to call `autoencoder(test_batch)` directly (same computation) and ensure the submission is written with the required `id,value` columns and `.csv` suffix.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend before importing TensorFlow (this is a common Kaggle env compatibility issue and is score-neutral). Then I fix the augmentation bug: `tf.image.rot90` swaps height/width, causing concatenation to fail; I rotate and then resize back to the configured `(420,540)` so shapes always match while keeping the same augmentation intent. Finally, I keep your model/training loop and submission logic intact, only unblocking the pipeline so it trains, predicts, and writes a valid `submission.csv` with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I fix the immediate runtime crash in the first cell by removing the protobuf-backend environment forcing that’s incompatible with TF 2.18/protobuf 6 in this Kaggle image, so TensorFlow imports cleanly. Then I fix a key logic issue that severely hurts score: the train and train_cleaned images must be paired by identical image id (filenames), but currently they’re loaded by independent sorted directory listings, which can misalign inputs/targets and train the model to map an image to the wrong “clean” label. Finally, I keep your model, training loop, and submission format the same, only ensuring the pairing is correct and the pipeline runs end-to-end to write `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this unblocks execution and is score-neutral). Then I keep your architecture and training loop intact, but correct the augmentation pipeline so each augmentation is applied sequentially (instead of repeatedly from the original), which restores the intended diversity and should significantly reduce RMSE toward the target. Finally, I keep the submission generation logic the same while ensuring numeric stability (float32) and that `submission.csv` is always written with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I first fix the TensorFlow/protobuf crash by removing the incompatible protobuf backend forcing; TF 2.18 with protobuf 6 in this environment should be imported with default settings. Then I fix a core logic bug that is severely hurting your score: the augmentation pipeline must apply the same random-free transforms in the same order to both `train` and `train_cleaned`, but your current sequential augmentation applies different transforms to the cleaned targets (because it reuses the evolving `x`), breaking input/target alignment. I keep the exact same augmentation “intent” (rot90/180/270 + flips and shape-stability resize), but generate each augmented variant from the original tensor so that noisy and clean stay perfectly paired. Finally, I keep model/training/submission logic the same and ensure the submission CSV is written correctly.'
- What this solution (achieved 0.28616) has done: 'We fix the crash in the first cell caused by an incompatibility between TensorFlow 2.18 and the protobuf runtime by forcing a compatible protobuf implementation *before* importing TensorFlow (this is required just to run end-to-end). Then we keep your model/training/inference logic intact, but make one score-critical correction: ensure `train` and `train_cleaned` are paired by filename (already present) and ensure the augmentation variants are generated from the same base tensor for both noisy and clean (your current cell 7 intent is correct, we just make it fully shape-safe and deterministic without altering the augmentation set). Finally, we keep the submission formatting the same while making the pixel-id generation faster and less error-prone (still produces identical `id,value` semantics) and guarantee `submission.csv` is written.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import crash by removing the incompatible protobuf environment forcing in the first cell (TF 2.18 + protobuf 6 works with defaults here). Then I make one score-critical but minimal correction to the model I/O: your autoencoder downsamples by MaxPool but doesn’t upsample back to the original size because `UpSampling2D` is placed too late in the decoder; I move `UpSampling2D` to the start of the decoder so the network outputs exactly `(420, 540, 1)` to match the clean targets (same layers, same counts, same core architecture intent). Finally, I keep your data loading/augmentation/training loop intact and only add a small safety check to ensure predictions and submission lengths match the required format before writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    cv2.setNumThreads(min(8, (os.cpu_count() or 4)))
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

need_extract = False
for folder in ["train", "test", "train_cleaned"]:
    if not os.path.isdir(os.path.join(path, folder)) and not os.path.isdir(
        os.path.join(path, "denoising-dirty-documents", folder)
    ):
        need_extract = True
        break

if need_extract:
    for zname in [
        "train.zip",
        "test.zip",
        "train_cleaned.zip",
        "sampleSubmission.csv.zip",
    ]:
        zpath = os.path.join(path_zip, zname)
        with zipfile.ZipFile(zpath, "r") as zip_ref:
            zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

if not (
    os.path.isdir(train_dir)
    and os.path.isdir(train_cleaned_dir)
    and os.path.isdir(test_dir)
):
    nested_base = os.path.join(path, "denoising-dirty-documents")
    train_dir = os.path.join(nested_base, "train")
    train_cleaned_dir = os.path.join(nested_base, "train_cleaned")
    test_dir = os.path.join(nested_base, "test")


def _list_pngs(d):
    files = []
    for fn in os.listdir(d):
        full = os.path.join(d, fn)
        if os.path.isfile(full) and fn.lower().endswith(".png"):
            files.append(fn)
    return sorted(files)


train_img = _list_pngs(train_dir)
train_cleaned_img = _list_pngs(train_cleaned_dir)
test_img = _list_pngs(test_dir)

print(
    "Found:",
    len(train_img),
    "train,",
    len(train_cleaned_img),
    "train_cleaned,",
    len(test_img),
    "test images",
)

if len(test_img) == 0:
    raise RuntimeError(f"No test .png images found in: {test_dir}")




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(train_dir, f)) for f in _list_pngs(train_dir)[:10]]
imgs = [im for im in imgs if im is not None]
print(
    "Median Dimensions:",
    np.median([img.shape[0] for img in imgs]),
    np.median([img.shape[1] for img in imgs]),
)
del imgs



## === cell 3
from concurrent.futures import ThreadPoolExecutor


def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img


def load_images_parallel(paths, max_workers):
    n = len(paths)
    out = np.empty((n, *config.IMG_SIZE, 1), dtype=np.float32)

    def _load_one(i_p):
        i, p = i_p
        out[i] = process_image(p)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_load_one, enumerate(paths)))
    return out




## === cell 4
train_files = set(_list_pngs(train_dir))
clean_files = set(_list_pngs(train_cleaned_dir))
paired_files = sorted(train_files.intersection(clean_files))

if len(paired_files) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned folders."
    )

missing_in_clean = sorted(train_files - clean_files)
missing_in_train = sorted(clean_files - train_files)
if missing_in_clean:
    print(
        f"Warning: {len(missing_in_clean)} files exist in train but not in train_cleaned. They will be ignored."
    )
if missing_in_train:
    print(
        f"Warning: {len(missing_in_train)} files exist in train_cleaned but not in train. They will be ignored."
    )

train_paths = [os.path.join(train_dir, f) for f in paired_files]
clean_paths = [os.path.join(train_cleaned_dir, f) for f in paired_files]
test_img = _list_pngs(test_dir)
test_paths = [os.path.join(test_dir, f) for f in test_img]

max_workers = min(8, (os.cpu_count() or 4))

train = load_images_parallel(train_paths, max_workers=max_workers)
train_cleaned = load_images_parallel(clean_paths, max_workers=max_workers)
test = load_images_parallel(test_paths, max_workers=max_workers)

train_img = paired_files
train_cleaned_img = paired_files

print("Loaded paired train set:", train.shape, train_cleaned.shape)
print("Loaded test set:", test.shape)



## === cell 5
train.shape, train_cleaned.shape, test.shape



## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)
plt.show()




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    """
    Preserved intent:
    - Noisy and clean must receive identical deterministic transforms.
    - Variants: [original, rot90, rot180, rot270, hflip, vflip].
    - Keep shape stable for rot90/rot270 by resizing back to (H, W).
    """
    tf.random.set_seed(seed)
    x0 = tf.convert_to_tensor(images, dtype=tf.float32)
    target_hw = config.IMG_SIZE  # (H, W)

    out = [x0]

    for step in pipeline:
        x = x0
        if step == "rot90":
            x = tf.image.rot90(x, k=1)
            x = tf.image.resize(x, target_hw, method="bilinear")
        elif step == "rot180":
            x = tf.image.rot90(x, k=2)
        elif step == "rot270":
            x = tf.image.rot90(x, k=3)
            x = tf.image.resize(x, target_hw, method="bilinear")
        elif step == "hflip":
            x = tf.image.flip_left_right(x)
        elif step == "vflip":
            x = tf.image.flip_up_down(x)
        else:
            raise ValueError(f"Unknown augmentation step: {step}")

        x = tf.cast(x, tf.float32)
        out.append(x)

    return tf.concat(out, axis=0)




## === cell 8
pipeline = ["rot90", "rot180", "rot270", "hflip", "vflip"]



## === cell 9
AUG_NAMES = ["orig"] + pipeline


def make_augmented_dataset(x_np, y_np, batch_size, seed=19):
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    x = tf.convert_to_tensor(x_np, dtype=tf.float32)
    y = tf.convert_to_tensor(y_np, dtype=tf.float32)

    base = tf.data.Dataset.from_tensor_slices((x, y)).with_options(opts).cache()

    target_hw = config.IMG_SIZE  # (H, W)

    def apply_step(xi, yi, step):
        if step == "orig":
            return xi, yi
        if step == "rot90":
            xo = tf.image.rot90(xi, k=1)
            yo = tf.image.rot90(yi, k=1)
            xo = tf.image.resize(xo, target_hw, method="bilinear")
            yo = tf.image.resize(yo, target_hw, method="bilinear")
            return tf.cast(xo, tf.float32), tf.cast(yo, tf.float32)
        if step == "rot180":
            xo = tf.image.rot90(xi, k=2)
            yo = tf.image.rot90(yi, k=2)
            return tf.cast(xo, tf.float32), tf.cast(yo, tf.float32)
        if step == "rot270":
            xo = tf.image.rot90(xi, k=3)
            yo = tf.image.rot90(yi, k=3)
            xo = tf.image.resize(xo, target_hw, method="bilinear")
            yo = tf.image.resize(yo, target_hw, method="bilinear")
            return tf.cast(xo, tf.float32), tf.cast(yo, tf.float32)
        if step == "hflip":
            return (
                tf.cast(tf.image.flip_left_right(xi), tf.float32),
                tf.cast(tf.image.flip_left_right(yi), tf.float32),
            )
        if step == "vflip":
            return (
                tf.cast(tf.image.flip_up_down(xi), tf.float32),
                tf.cast(tf.image.flip_up_down(yi), tf.float32),
            )
        raise ValueError(f"Unknown augmentation step: {step}")

    steps = ["orig"] + pipeline

    def expand_one(xi, yi):
        step_ds = tf.data.Dataset.from_tensor_slices(tf.constant(steps))
        return step_ds.map(
            lambda s: apply_step(xi, yi, s), num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = base.flat_map(expand_one)

    shuffle_buf = min(int(x_np.shape[0] * len(steps)), 256)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


processed_train = None
processed_train_cleaned = None




## === cell 10
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 11
y_hat_shape = autoencoder(train[:1], training=False).shape
if tuple(y_hat_shape[1:]) != tuple(train_cleaned[:1].shape[1:]):
    raise RuntimeError(
        f"Model output shape {y_hat_shape} does not match target shape {train_cleaned[:1].shape}."
    )
print("Output shape matches target:", y_hat_shape)



## === cell 12
BATCH_SIZE = 12

train_ds = make_augmented_dataset(train, train_cleaned, batch_size=BATCH_SIZE, seed=19)

es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train_ds,
    callbacks=[es],
    epochs=500,
    verbose=2,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/1597533517.py in <cell line: 0>()
      1 BATCH_SIZE = 12
      2 
----> 3 train_ds = make_augmented_dataset(train, train_cleaned, batch_size=BATCH_SIZE, seed=19)
      4 
      5 es = callbacks.EarlyStopping(

/tmp/ipykernel_11/3693717194.py in make_augmented_dataset(x_np, y_np, batch_size, seed)
     58         )
     59 
---> 60     ds = base.flat_map(expand_one)
     61 
     62     # Preserve shuffling semantics (per-epoch reshuffle, deterministic seed).

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in flat_map(self, map_func, name)
   2387     # pylint: disable=g-import-not-at-top,protected-access
   2388     from tensorflow.python.data.ops import flat_map_op
-> 2389     return flat_map_op._flat_map(self, map_func, name=name)
   2390     # pylint: enable=g-import-not-at-top,protected-access
   2391 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/flat_map_op.py in _flat_map(input_dataset, map_func, name)
     22 def _flat_map(input_dataset, map_func, name=None):  # pylint: disable=unused-private-name
     23   """See `Dataset.flat_map()` for details."""
---> 24   return _FlatMapDataset(input_dataset, map_func, name)
     25 
     26 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/flat_map_op.py in __init__(self, input_dataset, map_func, name)
     31 
     32     self._input_dataset = input_dataset
---> 33     self._map_func = structured_function.StructuredFunctionWrapper(
     34         map_func, self._transformation_name(), dataset=input_dataset)
     35     if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

OperatorNotAllowedInGraphError: in user code:

    File "/tmp/ipykernel_11/3693717194.py", line 57, in expand_one  *
        lambda s: apply_step(xi, yi, s), num_parallel_calls=tf.data.AUTOTUNE
    File "/tmp/ipykernel_11/3693717194.py", line 20, in apply_step  **
        if step == "orig":

    OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.


## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1420696232.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).plot(ax=ax)
      3 plt.show()
      4 del history
      5 

NameError: name 'history' is not defined

## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 15
decoded_imgs = autoencoder(train[:4], training=False).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)
plt.show()

del decoded_imgs



## === cell 16
import csv

test_ds = tf.data.Dataset.from_tensor_slices(test).batch(8).prefetch(tf.data.AUTOTUNE)
pred_test = autoencoder.predict(test_ds, verbose=0).astype(np.float32)

out_path = "submission.csv"

sample_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv"
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        sample_n = sum(1 for _ in f) - 1
    print("Sample submission rows:", sample_n)

test_hw = []
for fname in test_img:
    img0 = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"cv2.imread failed for test image: {fname}")
    test_hw.append(img0.shape)


def _write_image_rows(fh, imgid, preds_2d):
    h, w = preds_2d.shape
    r = np.repeat(np.arange(1, h + 1, dtype=np.int32), w)
    c = np.tile(np.arange(1, w + 1, dtype=np.int32), h)
    vals = preds_2d.reshape(-1)

    chunk = 250_000
    for start in range(0, vals.size, chunk):
        end = min(start + chunk, vals.size)
        ids = (
            imgid.astype(str)
            + "_"
            + r[start:end].astype(str)
            + "_"
            + c[start:end].astype(str)
        )
        lines = "\n".join(f"{i},{v:.10g}" for i, v in zip(ids, vals[start:end]))
        fh.write(lines + "\n")


with open(out_path, "w", newline="") as f:
    f.write("id,value\n")
    for i, fname in tqdm(list(enumerate(test_img)), total=len(test_img)):
        imgid_int = int(fname[:-4])
        imgid = np.array(imgid_int)  # for vectorized string ops
        h, w0 = test_hw[i]

        decoded_img = np.squeeze(pred_test[i]).astype(np.float32)
        preds = cv2.resize(decoded_img, (w0, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        preds = np.clip(preds, 0.0, 1.0)

        _write_image_rows(f, imgid, preds)

print(f"Results saved to {out_path}!")
print(pd.read_csv(out_path, nrows=5))
if os.path.exists(sample_path):
    with open(out_path, "rb") as f:
        sub_rows = sum(1 for _ in f) - 1
    print("Submission rows:", sub_rows, "Expected:", sample_n)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2269314940.py in <cell line: 0>()
     63         preds = np.clip(preds, 0.0, 1.0)
     64 
---> 65         _write_image_rows(f, imgid, preds)
     66 
     67 print(f"Results saved to {out_path}!")

/tmp/ipykernel_11/2269314940.py in _write_image_rows(fh, imgid, preds_2d)
     39         end = min(start + chunk, vals.size)
     40         ids = (
---> 41             imgid.astype(str)
     42             + "_"
     43             + r[start:end].astype(str)

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U21'), dtype('<U1')) -> None

## --- ERROR in outputing the csv:
Invalid submission: Expected the submission to have 5789880 rows, but got 0.
