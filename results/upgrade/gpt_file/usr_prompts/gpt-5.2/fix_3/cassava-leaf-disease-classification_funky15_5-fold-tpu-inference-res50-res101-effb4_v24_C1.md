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
def seq(image: np.ndarray) -> np.ndarray:
    """
    image: HxWxC uint8/float ndarray
    returns: augmented image ndarray (float32) resized to IMAGE_SIZE
    """
    x = tf.convert_to_tensor(image)
    if x.dtype != tf.uint8 and x.dtype != tf.float32 and x.dtype != tf.float64:
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

    return tf.cast(x, tf.float32).numpy()




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

    preds = []
    img_ids = []

    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    for fp in files:
        image = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).numpy()

        aug_imgs = [
            seq(image=image) * ((1 - g) + g / 255.0)
            for _ in range(aug_num)
            for g in normalize_indices
        ]
        avg_pred = np.concatenate(
            [
                mod.predict(np.expand_dims(aug_imgs[k], axis=0), verbose=0)
                for k in range(len(aug_imgs))
                for mod in model_obj_list
            ],
            axis=0,
        ).mean(0)

        preds.append(int(np.argmax(avg_pred)))
        img_ids.append(os.path.basename(fp))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 6
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]

    imgs = []
    for fp in files:
        im = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        im = tf.image.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        imgs.append(im)
    imgs = tf.stack(imgs, axis=0)  # [N,H,W,3]
    imgs = tf.cast(imgs, tf.float32)

    if not TTA:
        x = imgs / 255.0 if normalize else imgs
        probs_sum = None
        for mod in model_obj_list:
            p = mod.predict(x, batch_size=BATCH_SIZE, verbose=0)
            probs_sum = p if probs_sum is None else (probs_sum + p)
        probs = probs_sum / float(len(model_obj_list))
        labels = np.argmax(probs, axis=1).astype(int)
        return pd.DataFrame({"image_id": img_ids, "label": labels})

    probs_sum = None
    total_passes = 0
    base_imgs_np = imgs.numpy()  # seq expects numpy
    for _ in range(aug_num):
        aug_np = np.stack(
            [seq(image=base_imgs_np[i]) for i in range(base_imgs_np.shape[0])], axis=0
        ).astype(np.float32)
        x = aug_np / 255.0 if normalize else aug_np

        for mod in model_obj_list:
            p = mod.predict(x, batch_size=BATCH_SIZE, verbose=0)
            probs_sum = p if probs_sum is None else (probs_sum + p)
            total_passes += 1

    probs = probs_sum / float(total_passes)
    labels = np.argmax(probs, axis=1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 7
def get_preds(image_dir, model_obj, normalize=True):
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(fp) for fp in files]

    imgs = []
    for fp in files:
        im = tf.io.decode_jpeg(tf.io.read_file(fp), channels=3)
        im = tf.image.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        imgs.append(im)
    x = tf.cast(tf.stack(imgs, axis=0), tf.float32)
    if normalize:
        x = x / 255.0

    probs = model_obj.predict(x, batch_size=BATCH_SIZE, verbose=0)
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
