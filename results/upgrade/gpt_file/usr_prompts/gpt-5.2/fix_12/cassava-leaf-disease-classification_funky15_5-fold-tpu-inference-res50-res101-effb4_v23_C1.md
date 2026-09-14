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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf") or k.startswith("tensorflow"):
        del sys.modules[k]

import google.protobuf  # noqa: F401

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = os.path.join(BASE_INPUT, "test_images")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"
assert os.path.isfile(train_csv_path), f"train.csv not found: {train_csv_path}"

tf.random.set_seed(0)
np.random.seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64

AUG_IS_NOOP = True
try:
    from imgaug import augmenters as iaa  # may not be installed

    seq = iaa.Sequential(
        [
            iaa.Crop(px=(0, 128)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ]
    )

    def apply_aug(img_np):
        return seq(image=img_np)

    AUG_IS_NOOP = False
    print("Using imgaug for TTA.")
except Exception as e:
    print(
        f"imgaug not available or failed to import ({type(e).__name__}: {e}). Using no-op augmentation."
    )

    def apply_aug(img_np):
        return img_np

    AUG_IS_NOOP = True



## === cell 2
NUM_CLASSES = 5


def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    x = base.output
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


modeleffb4_0 = build_model()
mod_lst = [modeleffb4_0]


def _try_load_weights(model):
    input_root = "/kaggle/input"
    exts = (".h5", ".keras", ".ckpt")
    candidates = []
    for ext in exts:
        candidates.extend(
            glob.glob(os.path.join(input_root, "**", f"*{ext}"), recursive=True)
        )

    preferred = []
    for p in candidates:
        lp = os.path.basename(p).lower()
        if ("cassava" in lp or "eff" in lp or "b4" in lp or "efficientnet" in lp) and (
            "weight" in lp or "model" in lp
        ):
            preferred.append(p)

    search_list = preferred + candidates

    tried = 0
    for wpath in search_list:
        tried += 1
        if tried > 200:
            break
        try:
            model.load_weights(wpath)
            print(f"Loaded model weights from: {wpath}")
            return True
        except Exception:
            continue

    print(
        "No compatible external weights found in /kaggle/input; proceeding with ImageNet base + random head."
    )
    return False


loaded_external = _try_load_weights(modeleffb4_0)




## === cell 3
@tf.function(reduce_retracing=True)
def _tf_load_image_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear", antialias=False
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


def _load_image_as_array(path, target_size):
    img = _tf_load_image_uint8(tf.constant(path))
    return img.numpy()


def get_preds_model_list_norm_inds(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize_indices=None
):
    """
    normalize_indices: list of 0/1 indicators corresponding to model_obj_list indicating
    whether to scale inputs for each model.

    Kept for compatibility; main path uses get_preds_model_list().
    """
    if normalize_indices is None:
        normalize_indices = [1] * len(model_obj_list)
    if len(normalize_indices) != len(model_obj_list):
        raise ValueError("normalize_indices length must match model_obj_list length")

    preds = []
    img_ids = []

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    for p in img_paths:
        img_np = _load_image_as_array(p, IMAGE_SIZE)

        if TTA:
            aug_imgs = [apply_aug(img_np) for _ in range(aug_num)]
        else:
            aug_imgs = [img_np]

        all_preds = []
        for mod, norm_flag in zip(model_obj_list, normalize_indices):
            for a in aug_imgs:
                x = a.astype(np.float32)
                if norm_flag:
                    x = tf.keras.applications.efficientnet.preprocess_input(x)
                x = np.expand_dims(x, axis=0)
                all_preds.append(mod.predict(x, verbose=0))
        avg_pred = np.mean(np.concatenate(all_preds, axis=0), axis=0, keepdims=True)

        preds.append(int(np.argmax(avg_pred, axis=1)[0]))
        img_ids.append(os.path.basename(p))

    return pd.DataFrame({"image_id": img_ids, "label": preds})




## === cell 4
def _make_train_val_datasets(train_df, batch_size=BATCH_SIZE, val_split=0.1):
    df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n = len(df)
    n_val = int(round(n * val_split))
    val_df = df.iloc[:n_val].copy()
    tr_df = df.iloc[n_val:].copy()

    tr_paths = tf.constant(
        [os.path.join(BASE_INPUT, "train_images", x) for x in tr_df["image_id"].values]
    )
    tr_labels = tf.constant(tr_df["label"].values, dtype=tf.int64)

    val_paths = tf.constant(
        [os.path.join(BASE_INPUT, "train_images", x) for x in val_df["image_id"].values]
    )
    val_labels = tf.constant(val_df["label"].values, dtype=tf.int64)

    def _load_and_norm(path, label):
        img_u8 = _tf_load_image_uint8(path)
        img = tf.cast(img_u8, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    tr_ds = (
        tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
        .map(_load_and_norm, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_load_and_norm, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tr_ds, val_ds


if not loaded_external:
    train_df = pd.read_csv(train_csv_path)
    assert set(train_df.columns) >= {"image_id", "label"}
    train_df["label"] = train_df["label"].astype(int)

    base_model = modeleffb4_0.get_layer(
        index=1
    )  # EfficientNetB4 is the 2nd layer after Input
    try:
        base_model.trainable = False
    except Exception:
        for lyr in modeleffb4_0.layers:
            lyr.trainable = lyr.name == "pred"

    tr_ds, val_ds = _make_train_val_datasets(
        train_df, batch_size=BATCH_SIZE, val_split=0.1
    )

    modeleffb4_0.fit(tr_ds, validation_data=val_ds, epochs=3, verbose=2)




## === cell 5
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(p) for p in img_paths]
    n = len(img_paths)
    if n == 0:
        return pd.DataFrame({"image_id": [], "label": []})

    paths_tf = tf.constant(img_paths)

    def _load_and_maybe_norm(path):
        img_u8 = _tf_load_image_uint8(path)  # uint8 [H,W,3]
        img = tf.cast(img_u8, tf.float32)
        if normalize:
            img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    ds = (
        tf.data.Dataset.from_tensor_slices(paths_tf)
        .map(
            _load_and_maybe_norm,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    infer_fns = []
    for m in model_obj_list:

        @tf.function(reduce_retracing=True)
        def _infer(x, model=m):
            return model(x, training=False)

        infer_fns.append(_infer)

    out_labels = np.empty((n,), dtype=np.int64)
    write_pos = 0

    def _tta_make_aug_batch_float32(batch_imgs_f32, aug_num_local):
        B = batch_imgs_f32.shape[0]
        out = np.empty((B * aug_num_local, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        k = 0
        for i in range(B):
            img = batch_imgs_f32[i]
            for _ in range(aug_num_local):
                out[k] = apply_aug(img)
                k += 1
        return out

    effective_tta = bool(TTA and aug_num > 1 and (not AUG_IS_NOOP))

    for batch_imgs in ds:
        B = int(batch_imgs.shape[0])

        if effective_tta:
            batch_np = batch_imgs.numpy()  # float32 already preprocessed
            aug_batch = _tta_make_aug_batch_float32(batch_np, aug_num)  # (B*aug, H,W,3)

            denom = float(aug_num * len(model_obj_list))
            prob_sum = np.zeros((B, NUM_CLASSES), dtype=np.float64)

            x_tf = tf.convert_to_tensor(aug_batch, dtype=tf.float32)
            for infer in infer_fns:
                pr = infer(x_tf).numpy()  # (B*aug, C)
                pr = pr.reshape(B, aug_num, NUM_CLASSES).sum(axis=1, dtype=np.float64)
                prob_sum += pr
            avg_pred = prob_sum / denom
        else:
            denom = float(len(model_obj_list))
            prob_sum = np.zeros((B, NUM_CLASSES), dtype=np.float64)

            for infer in infer_fns:
                pr = infer(batch_imgs).numpy()  # (B,C)
                prob_sum += pr.astype(np.float64, copy=False)
            avg_pred = prob_sum / denom

        out_labels[write_pos : write_pos + B] = np.argmax(avg_pred, axis=1).astype(
            np.int64
        )
        write_pos += B

    return pd.DataFrame({"image_id": img_ids, "label": out_labels.tolist()})




## === cell 6
predict_df = get_preds_model_list(test_dir, mod_lst, normalize=True, aug_num=9)

sample_sub = pd.read_csv(sample_sub_path)
predict_df = sample_sub[["image_id"]].merge(predict_df, on="image_id", how="left")
predict_df["label"] = predict_df["label"].fillna(0).astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())



## === cell 7
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub.columns.tolist()}"
assert (
    sub.shape[0] == pd.read_csv(sample_sub_path).shape[0]
), "Row count mismatch vs sample_submission"
assert sub["label"].between(0, 4).all(), "Labels out of range [0,4]"
sub.head()
