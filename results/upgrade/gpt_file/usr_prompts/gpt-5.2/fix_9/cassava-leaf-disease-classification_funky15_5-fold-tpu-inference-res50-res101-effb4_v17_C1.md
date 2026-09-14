# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, glob, math, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

print("Tensorflow version " + tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

tf.config.experimental.enable_op_determinism()

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"



## === cell 2
MODEL_DIRS_TO_TRY = [
    "../input/resnet50-5fold-0",
    "../input/tpus-resnet50-with-5-fold",
    "../input/resnet50-5fold",
]


def try_load_fold_models():
    models = []

    search_dirs = [d for d in MODEL_DIRS_TO_TRY if os.path.isdir(d)]

    if os.path.isdir("../input"):
        search_dirs.append("../input")

    patterns = (
        "resnet50_*.h5",
        "model_*.h5",
        "fold*.h5",
        "*.h5",
    )

    seen_paths = set()
    candidate_paths = []

    for d in search_dirs:
        for pat in patterns:
            recursive = d == "../input"
            for p in glob.glob(
                os.path.join(d, "**", pat) if recursive else os.path.join(d, pat),
                recursive=recursive,
            ):
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                candidate_paths.append(p)

    def _score(path):
        bn = os.path.basename(path).lower()
        s = 0
        if "resnet50" in bn:
            s -= 5
        if "fold" in bn:
            s -= 3
        if re.search(r"(_|-)0\.h5$", bn):
            s -= 1
        return (s, len(path))

    candidate_paths.sort(key=_score)

    for p in candidate_paths:
        try:
            m = keras.models.load_model(p, compile=False)
            models.append(m)
            if len(models) >= 5:
                break
        except Exception:
            continue

    if len(models) > 0:
        print(f"Loaded {len(models)} models (pretrained) from ../input search.")
    return models


mod_lst = try_load_fold_models()
print("Number of pretrained models loaded:", len(mod_lst))




## === cell 3
def tta_augment(img_np, k):
    if k % 4 == 0:
        return img_np
    if k % 4 == 1:
        return np.fliplr(img_np)
    if k % 4 == 2:
        return np.flipud(img_np)
    return np.flipud(np.fliplr(img_np))


def load_image_rgb_float(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
num_classes = int(train_df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)


def df_to_dataset(df, img_dir, training):
    paths = df["image_id"].apply(lambda x: os.path.join(img_dir, x)).values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        label = tf.cast(label, tf.int32)
        label.set_shape([])
        return img, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    cache_path = None
    if training:
        cache_path = os.path.join("/tmp", "cassava_train_cache")
    else:
        cache_path = os.path.join("/tmp", "cassava_val_cache")
    ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        def _stateless_flip(img, label):
            s = tf.random.experimental.stateless_fold_in(
                tf.constant([SEED, 0], tf.int64), tf.cast(label, tf.int64)
            )
            img = tf.image.stateless_random_flip_left_right(img, seed=s)
            return img, label

        ds = ds.map(_stateless_flip, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = df_to_dataset(tr_df, TRAIN_IMG_DIR, training=True)
val_ds = df_to_dataset(va_df, TRAIN_IMG_DIR, training=False)


def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = layers.Dropout(0.2)(base.output)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


if len(mod_lst) == 0:
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    EPOCHS = 3
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    mod_lst = [model]
    print("Trained fallback model for inference.")
else:
    print("Using loaded pretrained models for inference.")



## === cell 5
test_dir = TEST_IMG_DIR


def _build_test_dataset(files):
    files = tf.constant(files)
    ds = tf.data.Dataset.from_tensor_slices(files)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache(os.path.join("/tmp", "cassava_test_cache"))

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def get_preds_model_list(image_dir, model_obj_list, TTA=True, aug_num=5):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]
    n = len(files)

    base_test_ds = _build_test_dataset(files)

    @tf.function(reduce_retracing=True)
    def _predict_batch_tta(m, batch, tta_on, aug_num_tf):
        if not tta_on:
            return m(batch, training=False)

        img0 = batch
        img1 = tf.image.flip_left_right(batch)
        img2 = tf.image.flip_up_down(batch)
        img3 = tf.image.flip_up_down(tf.image.flip_left_right(batch))
        variants = (img0, img1, img2, img3)

        p_sum = tf.zeros([tf.shape(batch)[0], num_classes], dtype=tf.float32)
        for k in tf.range(aug_num_tf):
            p_sum += m(variants[tf.math.mod(k, 4)], training=False)
        return p_sum / tf.cast(aug_num_tf, tf.float32)

    tta_on = tf.constant(bool(TTA))
    aug_num_tf = tf.constant(int(aug_num), dtype=tf.int32)

    probs_ens = np.zeros((n, num_classes), dtype=np.float32)

    for mod in model_obj_list:
        offset = 0
        for batch in base_test_ds:
            p = _predict_batch_tta(mod, batch, tta_on, aug_num_tf).numpy()
            bs = p.shape[0]
            probs_ens[offset : offset + bs] += p
            offset += bs

    probs_ens /= float(len(model_obj_list))
    labels = np.argmax(probs_ens, axis=1).astype(int)

    return pd.DataFrame({"image_id": img_ids, "label": labels.tolist()})




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
predict_df = get_preds_model_list(test_dir, mod_lst, TTA=True, aug_num=5)

pred_map = dict(zip(predict_df["image_id"], predict_df["label"]))
final_labels = sample_sub["image_id"].map(pred_map)

if final_labels.isna().any():
    fallback_label = int(train_df["label"].value_counts().idxmax())
    final_labels = final_labels.fillna(fallback_label).astype(int)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": final_labels.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 7
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(pd.read_csv(SAMPLE_SUB))
assert sub_check["label"].between(0, 4).all()
sub_check.head()
