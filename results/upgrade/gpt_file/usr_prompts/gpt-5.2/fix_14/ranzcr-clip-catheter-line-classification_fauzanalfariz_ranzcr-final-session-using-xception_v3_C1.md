# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import cv2
except Exception:
    cv2 = None

print("TF:", tf.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("cv2:", "ok" if cv2 is not None else "missing")




## === cell 1
WORK_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

test_dir = os.path.join(WORK_DIR, "test")
train_dir = os.path.join(WORK_DIR, "train")
assert os.path.isdir(test_dir), f"Missing test dir: {test_dir}"
assert os.path.isdir(train_dir), f"Missing train dir: {train_dir}"

test_images = (test_dir + "/" + ss["StudyInstanceUID"].astype(str) + ".jpg").values

full_label_cols = [
    c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]
]

missing_in_ss = [c for c in full_label_cols if c not in ss.columns]
for c in missing_in_ss:
    ss[c] = 0.0

final_label_cols = full_label_cols

print("Sample submission label cols:", ss.columns[1:].tolist())
print("Train label cols:", full_label_cols)
print("Added missing cols to ss:", missing_in_ss)
print("Final training targets:", len(final_label_cols))




## === cell 2
"""
TPU/GPU detection: return appropriate distribution strategy
"""
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 3
from threading import Lock

_NEEDLE_CACHE = {
    "folder": None,
    "imgs": None,  # np.ndarray (N,H,W,3) uint8 or list of uint8 HWC
    "n": 0,
    "initialized": False,
    "available": False,  # folder exists + cv2 available + at least 1 image loaded
}
_NEEDLE_LOCK = Lock()


def _init_needle_cache(needle_folder):
    with _NEEDLE_LOCK:
        if _NEEDLE_CACHE["initialized"] and _NEEDLE_CACHE["folder"] == needle_folder:
            return

        _NEEDLE_CACHE["folder"] = needle_folder
        _NEEDLE_CACHE["initialized"] = True
        _NEEDLE_CACHE["available"] = False
        _NEEDLE_CACHE["imgs"] = np.empty((0, 1, 1, 3), dtype=np.uint8)
        _NEEDLE_CACHE["n"] = 0

        if cv2 is None or needle_folder is None or (not os.path.isdir(needle_folder)):
            return

        files = [im for im in os.listdir(needle_folder) if im.lower().endswith(".png")]
        if not files:
            return

        imgs_list = []
        shapes = []
        for fn in files:
            p = os.path.join(needle_folder, fn)
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs_list.append(im)
            shapes.append(im.shape[:2])

        if not imgs_list:
            return

        imgs = imgs_list
        h0, w0 = imgs_list[0].shape[:2]
        if all((h == h0 and w == w0) for (h, w) in shapes):
            imgs = np.stack(imgs_list, axis=0)

        _NEEDLE_CACHE["imgs"] = imgs
        _NEEDLE_CACHE["n"] = len(imgs_list)
        _NEEDLE_CACHE["available"] = _NEEDLE_CACHE["n"] > 0


def _random_choice_needle():
    n = _NEEDLE_CACHE["n"]
    if n <= 0:
        return None
    idx = random.randrange(n)
    imgs = _NEEDLE_CACHE["imgs"]
    if isinstance(imgs, np.ndarray):
        return imgs[idx].copy()
    return imgs[idx].copy()


def NeedleAugmentation(
    image,
    n_needles=2,
    dark_needles=False,
    p=0.5,
    needle_folder="/kaggle/input/xray-needle-augmentation",
):
    """
    OpenCV-based custom augmentation. Used only on training pipeline.
    If cv2 or needle folder is unavailable, returns the image unchanged.
    """
    if random.random() >= p:
        return image

    _init_needle_cache(needle_folder)
    if not _NEEDLE_CACHE["available"]:
        return image

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]

    for _ in range(1, n_needles):
        needle = _random_choice_needle()
        if needle is None:
            continue

        needle = cv2.flip(needle, random.choice([-1, 0, 1]))
        needle = cv2.rotate(needle, random.choice([0, 1, 2]))

        h_height, h_width = needle.shape[:2]
        if h_height >= height or h_width >= width:
            continue

        roi_ho = random.randint(0, height - h_height)
        roi_wo = random.randint(0, width - h_width)
        roi = image[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

        img2gray = cv2.cvtColor(needle, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        if dark_needles:
            needle_fg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
        else:
            needle_fg = cv2.bitwise_and(needle, needle, mask=mask)

        dst = cv2.add(img_bg, needle_fg)
        image[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

    return image.astype(np.float32) / 255.0




## === cell 4
BATCH_SIZE = 8
EPOCHS = 3  # unchanged
TARGET_SIZE = 384  # unchanged


def _steps(n, bsz):
    return int(np.ceil(n / bsz))


from concurrent.futures import ThreadPoolExecutor

_AUG_POOL = ThreadPoolExecutor(max_workers=max(1, min((os.cpu_count() or 4), 8)))


class NeedleAugLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=None):
        training = tf.convert_to_tensor(False) if training is None else training

        def _aug_batch_np(batch):
            batch = np.asarray(batch, dtype=np.float32)
            out = np.empty_like(batch, dtype=np.float32)

            def _one(img):
                return NeedleAugmentation(
                    img, n_needles=2, dark_needles=False, p=0.5
                ).astype(np.float32)

            bs = batch.shape[0]
            if bs == 1:
                out[0] = _one(batch[0])
                return out

            for i, v in zip(range(bs), _AUG_POOL.map(_one, batch)):
                out[i] = v
            return out

        def _do_aug():
            y = tf.numpy_function(_aug_batch_np, [x], Tout=tf.float32)
            y.set_shape(x.shape)
            return y

        return tf.cond(tf.cast(training, tf.bool), _do_aug, lambda: tf.identity(x))




## === cell 5
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, target_size, method="bilinear")
        return img

    def decode_with_labels(path, label):
        return decode(path), tf.cast(label, tf.float32)

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.image.random_flip_up_down(img, seed=SEED)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=False,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_slack = True
    try:
        options.threading.private_threadpool_size = max(
            8, min(32, (os.cpu_count() or 8))
        )
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        dset = dset.cache(cache_dir) if cache_dir else dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 6
rng = np.random.RandomState(42)
unique_patients = train["PatientID"].unique()
rng.shuffle(unique_patients)
cut = int(0.8 * len(unique_patients))
train_pats = set(unique_patients[:cut])
val_pats = set(unique_patients[cut:])

train_df = train[train["PatientID"].isin(train_pats)].reset_index(drop=True)
val_df = train[train["PatientID"].isin(val_pats)].reset_index(drop=True)

train_paths = (
    train_dir + "/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
).values
val_paths = (train_dir + "/" + val_df["StudyInstanceUID"].astype(str) + ".jpg").values

y_train = train_df[final_label_cols].values.astype(np.float32)
y_val = val_df[final_label_cols].values.astype(np.float32)

print("Train size:", len(train_df), "Val size:", len(val_df))
print("Targets:", len(final_label_cols))

_init_needle_cache("/kaggle/input/xray-needle-augmentation")

CACHE_ROOT = "/kaggle/working/tfdata_cache_ranzcr"
os.makedirs(CACHE_ROOT, exist_ok=True)
train_cache = os.path.join(CACHE_ROOT, f"train_{TARGET_SIZE}_{TARGET_SIZE}.cache")
val_cache = os.path.join(CACHE_ROOT, f"val_{TARGET_SIZE}_{TARGET_SIZE}.cache")
test_cache = os.path.join(CACHE_ROOT, f"test_{TARGET_SIZE}_{TARGET_SIZE}.cache")

train_ds = build_dataset(
    train_paths,
    y_train,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache,
    drop_remainder=True,
)
val_ds = build_dataset(
    val_paths,
    y_val,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=val_cache,
    drop_remainder=False,
)
test_ds = build_dataset(
    test_images,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=test_cache,
    drop_remainder=False,
)

steps_per_epoch = _steps(len(train_df), BATCH_SIZE)
val_steps = _steps(len(val_df), BATCH_SIZE)
test_steps = _steps(len(ss), BATCH_SIZE)

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
    "test_steps:",
    test_steps,
)




## === cell 7
with strategy.scope():
    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        pooling=None,
    )
    inputs = tf.keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

    x = NeedleAugLayer(name="needle_aug")(inputs)

    x = tf.keras.applications.xception.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(final_label_cols), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        jit_compile=False,
    )

model.summary()




## === cell 8
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 9
pred = model.predict(test_ds, verbose=1)
pred = np.asarray(pred, dtype=np.float32)

if pred.shape[0] != len(ss):
    pred = pred[: len(ss)]

out = ss[["StudyInstanceUID"]].copy()
for i, c in enumerate(final_label_cols):
    out[c] = pred[:, i]

official_cols = ["StudyInstanceUID"] + ss.columns[1:].tolist()
for c in final_label_cols:
    if c not in official_cols:
        official_cols.append(c)

for c in official_cols[1:]:
    if c not in out.columns:
        out[c] = 0.0

out = out[official_cols]
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print("Columns:", out.columns.tolist())
print(out.head())
