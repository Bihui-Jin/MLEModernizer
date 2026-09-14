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
import os, glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64




## === cell 2
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    test_dir = "../input/cassava-leaf-disease-classification/test_images"

print("Using test_dir:", test_dir)
print("Num test images found:", len(glob.glob(os.path.join(test_dir, "*.jpg"))))




## === cell 3
def _find_any_h5_model(search_root="/kaggle/input"):
    if not os.path.isdir(search_root):
        return None
    candidates = glob.glob(os.path.join(search_root, "**", "*.h5"), recursive=True)
    return candidates[0] if len(candidates) else None


h5_path = _find_any_h5_model("/kaggle/input")
fallback_is_efficientnet = False

if h5_path is not None:
    print("Found .h5 model, loading:", h5_path)
    try:
        loaded_model = keras.models.load_model(h5_path, compile=False)
        mod_lst = [loaded_model]
    except Exception as e:
        print(
            "Failed to load found .h5 model; falling back to a small model. Error:",
            repr(e),
        )
        h5_path = None

if h5_path is None:
    print(
        "No usable pretrained .h5 model found; using EfficientNetB0 (ImageNet) fallback model."
    )
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False  # inference-only in this script

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    fallback_model = keras.Model(inputs, outputs)
    mod_lst = [fallback_model]
    fallback_is_efficientnet = True

print("Number of models in mod_lst:", len(mod_lst))
print("fallback_is_efficientnet:", fallback_is_efficientnet)




## === cell 4
@tf.function(jit_compile=True)
def _seq_tf(image):
    x = tf.convert_to_tensor(image)
    if x.dtype not in (tf.uint8, tf.float32, tf.float64):
        x = tf.cast(x, tf.uint8)

    crop_px = tf.random.uniform([], minval=0, maxval=129, dtype=tf.int32)
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    new_h = tf.maximum(1, h - 2 * crop_px)
    new_w = tf.maximum(1, w - 2 * crop_px)
    x = tf.image.random_crop(x, size=[new_h, new_w, 3])
    x = tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return tf.cast(x, tf.float32)


@tf.function(jit_compile=True)
def _seq_tf_batch(batch):
    return tf.map_fn(
        _seq_tf, batch, fn_output_signature=tf.float32, parallel_iterations=32
    )


def seq(image: np.ndarray) -> np.ndarray:
    """
    image: HxWxC uint8/float ndarray
    returns: augmented image ndarray (float32) resized to IMAGE_SIZE

    Note: kept for API compatibility; internally uses the compiled TF implementation.
    """
    return _seq_tf(image).numpy()




## === cell 5
def get_preds_model_list_norm_inds(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize_indices=None
):
    """
    normalize_indices: list of 0/1 flags; for each augmentation we also run each normalization option.
    This keeps the original function behavior.
    """
    if normalize_indices is None:
        normalize_indices = [1]

    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]
    if len(files) == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    def _load(fp):
        im = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        im = tf.image.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        return tf.cast(im, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(files).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    num_classes = int(model_obj_list[0].output_shape[-1])
    probs_sum = np.zeros((len(files), num_classes), dtype=np.float32)
    total_passes = 0

    offset = 0
    for batch in ds:
        bsz = int(batch.shape[0])

        for _ in range(aug_num):
            aug = _seq_tf_batch(batch) if TTA else batch  # [B,512,512,3] float32

            for g in normalize_indices:
                scale = (1.0 - float(g)) + float(g) / 255.0
                x = aug * scale

                for mod in model_obj_list:
                    p = mod(x, training=False)
                    probs_sum[offset : offset + bsz] += tf.cast(p, tf.float32).numpy()
                    total_passes += 1

        offset += bsz

    probs = probs_sum / float(total_passes)
    labels = np.argmax(probs, axis=1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 6
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]
    if len(files) == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    def _load(fp):
        im = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        im = tf.image.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        return tf.cast(im, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(files).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    num_classes = int(model_obj_list[0].output_shape[-1])
    probs_sum = np.zeros((len(files), num_classes), dtype=np.float32)
    total_passes = 0

    offset = 0
    for batch in ds:
        bsz = int(batch.shape[0])

        if not TTA:
            x = (batch / 255.0) if normalize else batch
            for mod in model_obj_list:
                p = mod(x, training=False)
                probs_sum[offset : offset + bsz] += tf.cast(p, tf.float32).numpy()
                total_passes += 1
            offset += bsz
            continue

        for _ in range(aug_num):
            aug = _seq_tf_batch(batch)  # [B,512,512,3]
            x = (aug / 255.0) if normalize else aug
            for mod in model_obj_list:
                p = mod(x, training=False)
                probs_sum[offset : offset + bsz] += tf.cast(p, tf.float32).numpy()
                total_passes += 1

        offset += bsz

    probs = probs_sum / float(total_passes)
    labels = np.argmax(probs, axis=1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 7
def get_preds(image_dir, model_obj, normalize=True):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]
    if len(files) == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    def _load(fp):
        im = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        im = tf.image.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = tf.cast(im, tf.float32)
        if normalize:
            im = im / 255.0
        return im

    ds = tf.data.Dataset.from_tensor_slices(files).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    all_probs = []
    for batch in ds:
        p = model_obj(batch, training=False)
        all_probs.append(tf.cast(p, tf.float32).numpy())
    probs = np.concatenate(all_probs, axis=0)

    labels = np.argmax(probs, axis=1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 8
predict_df = get_preds_model_list(
    test_dir,
    mod_lst,
    normalize=False,  # keep raw [0..255] float32; EfficientNet model does preprocessing internally
    aug_num=5,
)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

sample = pd.read_csv(sample_path)
sub = sample[["image_id"]].merge(predict_df, on="image_id", how="left")

sub["label"] = sub["label"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())




## === cell 9
print(predict_df.shape, predict_df.head())
print(pd.read_csv("submission.csv").head())
