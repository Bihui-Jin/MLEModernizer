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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
tf.config.experimental.enable_tensor_float_32_execution(False)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64

NUM_CLASSES = 5

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"




## === cell 2
def build_model(image_size=512, num_classes=5):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    base = keras.applications.EfficientNetB4(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


modeleffb4_0 = build_model(IMAGE_SIZE, NUM_CLASSES)
mod_lst = [modeleffb4_0]

for _m in mod_lst:
    try:
        _m.compile(run_eagerly=False)
    except Exception:
        pass




## === cell 3
@tf.function
def _tta_augment_tf(img_uint8, crop_max=128, p_fliplr=0.5, p_flipud=0.5):
    """
    img_uint8: HxWx3 uint8 tensor
    returns: IMAGE_SIZExIMAGE_SIZEx3 uint8 tensor
    """
    img = img_uint8
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    if crop_max > 0:
        h_lim = tf.minimum(tf.cast(crop_max, tf.int32), h // 4)
        w_lim = tf.minimum(tf.cast(crop_max, tf.int32), w // 4)
        top = tf.random.uniform([], 0, h_lim + 1, dtype=tf.int32)
        bottom = tf.random.uniform([], 0, h_lim + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, w_lim + 1, dtype=tf.int32)
        right = tf.random.uniform([], 0, w_lim + 1, dtype=tf.int32)

        end_h = tf.cond((h - bottom) > top, lambda: h - bottom, lambda: h)
        end_w = tf.cond((w - right) > left, lambda: w - right, lambda: w)
        img = img[top:end_h, left:end_w, :]

    img_f = tf.cast(img, tf.float32)
    img_f = tf.image.resize(
        img_f, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear", antialias=False
    )
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)

    r1 = tf.random.uniform([], 0.0, 1.0)
    img_u8 = tf.cond(
        r1 < p_fliplr, lambda: tf.image.flip_left_right(img_u8), lambda: img_u8
    )
    r2 = tf.random.uniform([], 0.0, 1.0)
    img_u8 = tf.cond(
        r2 < p_flipud, lambda: tf.image.flip_up_down(img_u8), lambda: img_u8
    )

    return img_u8


@tf.function
def _load_image_uint8_tf(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_jpeg(b, channels=3)  # uint8
    img = tf.image.resize(
        tf.cast(img, tf.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
        antialias=False,
    )
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    return img




## === cell 4
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    try:
        image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        )
    except FileNotFoundError:
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")

    img_ids = [os.path.basename(p) for p in image_paths]
    n = len(image_paths)

    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    def _load_u8(path):
        return _load_image_uint8_tf(path)

    ds = paths_ds.map(_load_u8, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if TTA:
        def _augment_batch(imgs_u8):
            return tf.map_fn(
                _tta_augment_tf,
                imgs_u8,
                fn_output_signature=tf.TensorSpec(
                    shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.uint8
                ),
                parallel_iterations=16,
            )

        ds = ds.flat_map(lambda b: tf.data.Dataset.from_tensors(b).repeat(aug_num))
        ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)

    def _to_model_input(imgs_u8):
        x = tf.cast(imgs_u8, tf.float32)
        if normalize:
            x = x / 255.0
        return x

    ds = ds.map(_to_model_input, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    preds = np.empty((n,), dtype=np.int64)

    bsz = BATCH_SIZE
    if TTA:
        num_orig_batches = (n + bsz - 1) // bsz
        it = iter(ds)
        out_pos = 0
        for b in range(num_orig_batches):
            curr_bsz = min(bsz, n - b * bsz)

            sum_over_aug = np.zeros((curr_bsz, NUM_CLASSES), dtype=np.float32)
            for _ in range(aug_num):
                x = next(it)  # (curr_bsz, H, W, 3)
                sum_over_models = None
                for mod in model_obj_list:
                    p = mod.predict_on_batch(x)  # (curr_bsz, C)
                    if sum_over_models is None:
                        sum_over_models = p
                    else:
                        sum_over_models = sum_over_models + p
                sum_over_models = sum_over_models / float(len(model_obj_list))
                sum_over_aug += sum_over_models

            avg_pred = sum_over_aug / float(aug_num)
            preds[out_pos : out_pos + curr_bsz] = np.argmax(avg_pred, axis=1).astype(
                np.int64
            )
            out_pos += curr_bsz
    else:
        out_pos = 0
        for x in ds:
            curr_bsz = int(x.shape[0])
            sum_over_models = None
            for mod in model_obj_list:
                p = mod.predict_on_batch(x)  # (curr_bsz, C)
                if sum_over_models is None:
                    sum_over_models = p
                else:
                    sum_over_models = sum_over_models + p
            avg_pred = sum_over_models / float(len(model_obj_list))
            preds[out_pos : out_pos + curr_bsz] = np.argmax(avg_pred, axis=1).astype(
                np.int64
            )
            out_pos += curr_bsz

    return pd.DataFrame({"image_id": img_ids, "label": preds.tolist()})


predict_df = get_preds_model_list(
    test_dir, mod_lst, normalize=True, aug_num=4, TTA=True
)

sample_sub = pd.read_csv(sample_sub_path)
submission = sample_sub[["image_id"]].merge(predict_df, on="image_id", how="left")

submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)




## === cell 5
try:
    from IPython.display import display

    display(submission.head(10))
except Exception as e:
    print("Display not available:", e)
