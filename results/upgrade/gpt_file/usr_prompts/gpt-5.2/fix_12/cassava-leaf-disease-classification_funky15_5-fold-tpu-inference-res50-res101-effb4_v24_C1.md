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

0.8895436687821094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I remove the problematic/unused imports that trigger the `MessageFactory.GetPrototype` error, and eliminate the dependency on unavailable external model files by building a small fallback Keras model only when no pretrained `.h5` can be found. I also remove `imgaug` (not installed) and replace the test-time augmentation with simple TensorFlow image augmentations so `seq` is always defined. Finally, I ensure prediction runs on the full test set efficiently (batching instead of per-image `predict`) and that the produced `submission.csv` exactly matches `sample_submission.csv` ordering and columns.'

# 9. Code solution

## === cell 0
import os, glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

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
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
@tf.function(jit_compile=False)
def _augment_batch(batch):
    x = tf.convert_to_tensor(batch)
    if x.dtype not in (tf.uint8, tf.float16, tf.float32, tf.float64):
        x = tf.cast(x, tf.uint8)
    x = tf.cast(x, tf.float32)

    bsz = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]

    crop_px = tf.random.uniform([bsz], minval=0, maxval=129, dtype=tf.int32)
    new_h = tf.maximum(1, h - 2 * crop_px)
    new_w = tf.maximum(1, w - 2 * crop_px)

    max_off_y = tf.maximum(0, h - new_h)
    max_off_x = tf.maximum(0, w - new_w)

    oy = tf.where(
        max_off_y > 0,
        tf.random.uniform(
            [bsz], minval=0, maxval=tf.reduce_max(max_off_y) + 1, dtype=tf.int32
        ),
        tf.zeros([bsz], dtype=tf.int32),
    )
    ox = tf.where(
        max_off_x > 0,
        tf.random.uniform(
            [bsz], minval=0, maxval=tf.reduce_max(max_off_x) + 1, dtype=tf.int32
        ),
        tf.zeros([bsz], dtype=tf.int32),
    )
    oy = tf.minimum(oy, max_off_y)
    ox = tf.minimum(ox, max_off_x)

    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)

    y1 = tf.cast(oy, tf.float32) / h_f
    x1 = tf.cast(ox, tf.float32) / w_f
    y2 = tf.cast(oy + new_h, tf.float32) / h_f
    x2 = tf.cast(ox + new_w, tf.float32) / w_f

    boxes = tf.stack([y1, x1, y2, x2], axis=1)  # [bsz,4]
    box_indices = tf.range(bsz, dtype=tf.int32)

    out = tf.image.crop_and_resize(
        x,
        boxes=boxes,
        box_indices=box_indices,
        crop_size=[IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
    )
    out = tf.image.random_flip_left_right(out)
    out = tf.image.random_flip_up_down(out)
    return out


def seq(image: np.ndarray) -> np.ndarray:
    return _augment_batch(tf.expand_dims(image, 0))[0].numpy()




## === cell 5
def get_preds_model_list_norm_inds(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize_indices=None
):
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

    opts = tf.data.Options()
    opts.experimental_deterministic = False

    ds = (
        tf.data.Dataset.from_tensor_slices(files)
        .with_options(opts)
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()  # safe for test set size; removes redundant decode/resize work
        .prefetch(tf.data.AUTOTUNE)
    )

    num_classes = int(model_obj_list[0].output_shape[-1])

    norm_scales = tf.constant(
        [(1.0 - float(g)) + float(g) / 255.0 for g in normalize_indices],
        dtype=tf.float32,
    )
    n_scales = int(norm_scales.shape[0])

    fwd_fns = []
    for mod in model_obj_list:

        @tf.function(jit_compile=False, reduce_retracing=True)
        def _fwd(x, _m=mod):
            return tf.cast(_m(x, training=False), tf.float32)

        fwd_fns.append(_fwd)

    n_models = len(fwd_fns)
    denom = float((aug_num if TTA else 1) * n_scales * n_models)
    total_n = len(files)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _predict_labels(ds_in):
        ta = tf.TensorArray(tf.int64, size=total_n)
        idx = tf.constant(0, dtype=tf.int32)

        for batch in ds_in:
            bsz = tf.shape(batch)[0]
            batch_sum = tf.zeros([bsz, num_classes], dtype=tf.float32)

            if not TTA:
                for i in tf.range(n_scales):
                    x = batch * norm_scales[i]
                    for fwd in fwd_fns:
                        batch_sum += fwd(x)
            else:
                aug_list = tf.TensorArray(tf.float32, size=aug_num)
                for k in tf.range(aug_num):
                    aug_list = aug_list.write(k, _augment_batch(batch))
                aug_big = tf.concat(aug_list.stack(), axis=0)  # [aug_num*bsz, ...]
                for i in tf.range(n_scales):
                    x = aug_big * norm_scales[i]
                    for fwd in fwd_fns:
                        p = fwd(x)  # [aug_num*bsz, num_classes]
                        p = tf.reshape(p, [aug_num, bsz, num_classes])
                        batch_sum += tf.reduce_sum(p, axis=0)

            probs = batch_sum / denom
            labels = tf.argmax(probs, axis=1, output_type=tf.int64)

            ta = ta.scatter(tf.range(idx, idx + bsz), labels)
            idx += bsz

        return ta.stack()

    labels = _predict_labels(ds).numpy().astype(int)
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

    opts = tf.data.Options()
    opts.experimental_deterministic = False

    ds = (
        tf.data.Dataset.from_tensor_slices(files)
        .with_options(opts)
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()  # avoid repeating decode/resize; test set fits comfortably
        .prefetch(tf.data.AUTOTUNE)
    )

    num_classes = int(model_obj_list[0].output_shape[-1])

    scale = (
        tf.constant(1.0 / 255.0, tf.float32)
        if normalize
        else tf.constant(1.0, tf.float32)
    )

    fwd_fns = []
    for mod in model_obj_list:

        @tf.function(jit_compile=False, reduce_retracing=True)
        def _fwd(x, _m=mod):
            return tf.cast(_m(x, training=False), tf.float32)

        fwd_fns.append(_fwd)

    n_models = len(fwd_fns)
    denom = float((aug_num if TTA else 1) * n_models)
    total_n = len(files)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _predict_labels(ds_in):
        ta = tf.TensorArray(tf.int64, size=total_n)
        idx = tf.constant(0, dtype=tf.int32)

        for batch in ds_in:
            bsz = tf.shape(batch)[0]
            batch_sum = tf.zeros([bsz, num_classes], dtype=tf.float32)

            if not TTA:
                x = batch * scale
                for fwd in fwd_fns:
                    batch_sum += fwd(x)
            else:
                aug_list = tf.TensorArray(tf.float32, size=aug_num)
                for k in tf.range(aug_num):
                    aug_list = aug_list.write(k, _augment_batch(batch))
                aug_big = tf.concat(aug_list.stack(), axis=0)  # [aug_num*bsz, ...]
                x = aug_big * scale
                for fwd in fwd_fns:
                    p = fwd(x)  # [aug_num*bsz, num_classes]
                    p = tf.reshape(p, [aug_num, bsz, num_classes])
                    batch_sum += tf.reduce_sum(p, axis=0)

            probs = batch_sum / denom
            labels = tf.argmax(probs, axis=1, output_type=tf.int64)

            ta = ta.scatter(tf.range(idx, idx + bsz), labels)
            idx += bsz

        return ta.stack()

    labels = _predict_labels(ds).numpy().astype(int)
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

    opts = tf.data.Options()
    opts.experimental_deterministic = False

    ds = (
        tf.data.Dataset.from_tensor_slices(files)
        .with_options(opts)
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    total_n = len(files)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _predict_labels(ds_in):
        ta = tf.TensorArray(tf.int64, size=total_n)
        idx = tf.constant(0, dtype=tf.int32)
        for batch in ds_in:
            probs = tf.cast(model_obj(batch, training=False), tf.float32)
            labels = tf.argmax(probs, axis=1, output_type=tf.int64)
            bsz = tf.shape(batch)[0]
            ta = ta.scatter(tf.range(idx, idx + bsz), labels)
            idx += bsz
        return ta.stack()

    labels = _predict_labels(ds).numpy().astype(int)
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3703226505.py in <cell line: 0>()
----> 1 predict_df = get_preds_model_list(
      2     test_dir,
      3     mod_lst,
      4     normalize=False,  # keep raw [0..255] float32; EfficientNet model does preprocessing internally
      5     aug_num=5,

/tmp/ipykernel_11/2128366507.py in get_preds_model_list(image_dir, model_obj_list, TTA, aug_num, normalize)
     78         return ta.stack()
     79 
---> 80     labels = _predict_labels(ds).numpy().astype(int)
     81     return pd.DataFrame({"image_id": img_ids, "label": labels})
     82 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filete68yv8k.py in tf___predict_labels(ds_in)
    107                 aug_big = ag__.Undefined('aug_big')
    108                 probs = ag__.Undefined('probs')
--> 109                 ag__.for_stmt(ag__.ld(ds_in), None, loop_body_3, get_state_4, set_state_4, ('ta', 'idx'), {'iterate_names': 'batch'})
    110                 try:
    111                     do_return = True

/tmp/__autograph_generated_filete68yv8k.py in loop_body_3(itr_3)
     90                             batch_sum += ag__.converted_call(tf.reduce_sum, (p,), dict(axis=0), fscope)
     91                         ag__.for_stmt(ag__.ld(fwd_fns), None, loop_body_2, get_state_2, set_state_2, ('batch_sum',), {'iterate_names': 'fwd'})
---> 92                     ag__.if_stmt(ag__.not_(ag__.ld(TTA)), if_body, else_body, get_state_3, set_state_3, ('batch_sum',), 1)
     93                     probs = ag__.ld(batch_sum) / ag__.ld(denom)
     94                     labels = ag__.converted_call(ag__.ld(tf).argmax, (ag__.ld(probs),), dict(axis=1, output_type=ag__.ld(tf).int64), fscope)

/tmp/__autograph_generated_filete68yv8k.py in else_body()
     89                             batch_sum = ag__.ld(batch_sum)
     90                             batch_sum += ag__.converted_call(tf.reduce_sum, (p,), dict(axis=0), fscope)
---> 91                         ag__.for_stmt(ag__.ld(fwd_fns), None, loop_body_2, get_state_2, set_state_2, ('batch_sum',), {'iterate_names': 'fwd'})
     92                     ag__.if_stmt(ag__.not_(ag__.ld(TTA)), if_body, else_body, get_state_3, set_state_3, ('batch_sum',), 1)
     93                     probs = ag__.ld(batch_sum) / ag__.ld(denom)

/tmp/__autograph_generated_filete68yv8k.py in loop_body_2(itr_2)
     85                             nonlocal batch_sum
     86                             fwd = itr_2
---> 87                             p = ag__.converted_call(ag__.ld(fwd), (ag__.ld(x),), None, fscope)
     88                             p = ag__.converted_call(ag__.ld(tf).reshape, (ag__.ld(p), [ag__.ld(aug_num), ag__.ld(bsz), ag__.ld(num_classes)]), None, fscope)
     89                             batch_sum = ag__.ld(batch_sum)

/tmp/__autograph_generated_filelspn6u8g.py in tf___fwd(x, _m)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___fwd

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: in user code:

    File "/tmp/ipykernel_11/2128366507.py", line 68, in _predict_labels  *
        p = fwd(x)  # [aug_num*bsz, num_classes]
    File "/tmp/ipykernel_11/2128366507.py", line 40, in _fwd  *
        return tf.cast(_m(x, training=False), tf.float32)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py", line 245, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "functional" is incompatible with the layer: expected shape=(None, 512, 512, 3), found shape=(5, None, 512, 512, 3)


## === cell 9
print(predict_df.shape, predict_df.head())
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/780705536.py in <cell line: 0>()
----> 1 print(predict_df.shape, predict_df.head())
      2 print(pd.read_csv("submission.csv").head())

NameError: name 'predict_df' is not defined
