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

# 5. Target score

0.8824418253248716

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def _decode_resize_uint8_tf(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_jpeg(b, channels=3)  # uint8, original size
    img_f = tf.image.resize(
        tf.cast(img, tf.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
        antialias=False,
    )
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)
    return img_u8


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


_load_image_uint8_tf = _decode_resize_uint8_tf




## === cell 4
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    image_paths = tf.io.gfile.glob(os.path.join(image_dir, "*.jpg"))
    image_paths = sorted(image_paths)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")

    img_ids = [os.path.basename(p) for p in image_paths]
    n = len(image_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True

    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    @tf.function(reduce_retracing=True)
    def _infer_step(x):
        sum_over = None
        for m in model_obj_list:
            p = m(x, training=False)
            sum_over = p if sum_over is None else (sum_over + p)
        return sum_over / tf.cast(len(model_obj_list), tf.float32)

    if TTA:
        base_ds = paths_ds.map(
            _load_image_uint8_tf, num_parallel_calls=AUTOTUNE
        ).cache()

        ds = base_ds.repeat(aug_num)

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

        @tf.function(reduce_retracing=True)
        def _aug_and_norm_batch(imgs_u8):
            imgs_u8 = tf.vectorized_map(_tta_augment_tf, imgs_u8)
            x = tf.cast(imgs_u8, tf.float32)
            if normalize:
                x = x / 255.0
            return x

        ds = ds.map(_aug_and_norm_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        ds = ds.with_options(options)

        sum_logits = np.zeros((n, NUM_CLASSES), dtype=np.float32)

        offset = 0
        for x in ds:
            p = _infer_step(x)  # tensor [bs, C]
            p_np = p.numpy().astype(np.float32, copy=False)
            bs = p_np.shape[0]

            idx = np.arange(offset, offset + bs, dtype=np.int64) % n
            np.add.at(sum_logits, idx, p_np)
            offset += bs

        avg_preds = sum_logits / float(aug_num)
        preds = np.argmax(avg_preds, axis=1).astype(np.int64)
        return pd.DataFrame({"image_id": img_ids, "label": preds.tolist()})

    else:
        ds = paths_ds.map(_load_image_uint8_tf, num_parallel_calls=AUTOTUNE)

        @tf.function(reduce_retracing=True)
        def _to_model_input(imgs_u8):
            x = tf.cast(imgs_u8, tf.float32)
            if normalize:
                x = x / 255.0
            return x

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.map(_to_model_input, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        ds = ds.with_options(options)

        preds_all = np.empty((n,), dtype=np.int64)
        out = 0
        for x in ds:
            p = _infer_step(x).numpy()
            bs = p.shape[0]
            preds_all[out : out + bs] = np.argmax(p, axis=1).astype(np.int64)
            out += bs

        return pd.DataFrame({"image_id": img_ids, "label": preds_all.tolist()})


predict_df = get_preds_model_list(
    test_dir, mod_lst, normalize=True, aug_num=4, TTA=True
)

sample_sub = pd.read_csv(sample_sub_path)
submission = sample_sub[["image_id"]].merge(predict_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2882066387.py in <cell line: 0>()
     98 
     99 
--> 100 predict_df = get_preds_model_list(
    101     test_dir, mod_lst, normalize=True, aug_num=4, TTA=True
    102 )

/tmp/ipykernel_11/2882066387.py in get_preds_model_list(image_dir, model_obj_list, TTA, aug_num, normalize)
     57         # For batch k, we can map each row to its original image index by global offset.
     58         offset = 0
---> 59         for x in ds:
     60             p = _infer_step(x)  # tensor [bs, C]
     61             p_np = p.numpy().astype(np.float32, copy=False)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:5 transformation with iterator: Iterator::Root::Prefetch::ParallelMapV2: PartialTensorShape: Incompatible shapes during merge: [396,401,3] vs. [434,365,3]
	 [[{{function_node __inference_f_6214}}{{node strided_slice_2/pfor/TensorArrayV2Stack/TensorListStack}}]] [Op:IteratorGetNext] name: 

## === cell 5
try:
    from IPython.display import display

    display(submission.head(10))
except Exception as e:
    print("Display not available:", e)
