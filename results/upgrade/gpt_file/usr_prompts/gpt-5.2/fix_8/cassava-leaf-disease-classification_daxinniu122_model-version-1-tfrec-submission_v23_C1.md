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

0.8433061347839227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
OUT_PATH = "/kaggle/working/submission.csv"

MODEL1_PATH = "/kaggle/input/f-models/ResNet50_f.h5"
MODEL2_PATH = "/kaggle/input/f-models/VGG19_f.h5"
MODEL3_PATH = "/kaggle/input/f-models/MobileNetV3L_f.h5"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _try_load_model(path: str):
    if tf.io.gfile.exists(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e1:
            print(f"Warning: tf.keras failed to load model at {path}: {e1}")
        try:
            import keras as keras_standalone  # present in many Kaggle TF images

            return keras_standalone.models.load_model(path, compile=False)
        except Exception as e2:
            print(f"Warning: standalone keras failed to load model at {path}: {e2}")
    return None


model1 = _try_load_model(MODEL1_PATH)
model2 = _try_load_model(MODEL2_PATH)
model3 = _try_load_model(MODEL3_PATH)


def _build_fallback_model(
    backbone_name: str, input_shape=(512, 512, 3), num_classes=5, seed=42
):
    tf.keras.utils.set_random_seed(seed)
    if backbone_name == "ResNet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "VGG19":
        base = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "MobileNetV3Large":
        base = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(backbone_name)

    x_in = tf.keras.Input(shape=input_shape)
    x = base(x_in, training=False)
    x = tf.keras.layers.Dense(num_classes, activation=None)(x)
    return tf.keras.Model(x_in, x)


fallback_used = False
if model1 is None or model2 is None or model3 is None:
    fallback_used = True
    print(
        "Info: Using fallback ImageNet models because one or more provided .h5 models are missing/unloadable."
    )
    model1 = _build_fallback_model("ResNet50", seed=42)
    model2 = _build_fallback_model("VGG19", seed=43)
    model3 = _build_fallback_model("MobileNetV3Large", seed=44)



## === cell 2
for m in (model1, model2, model3):
    m.trainable = False

image_ids = sample_sub["image_id"].values
n = len(image_ids)

tfrec_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True

RESNET_VGG_SIZE = 224
MNV3_SIZE = 192

id_to_idx = {k: i for i, k in enumerate(image_ids.tolist())}

if not tfrec_files:
    paths = tf.constant([os.path.join(TEST_DIR, x) for x in image_ids])

    def _load_and_preprocess_from_path(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img224 = tf.image.resize(
            img,
            (RESNET_VGG_SIZE, RESNET_VGG_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        img192 = tf.image.resize(
            img, (MNV3_SIZE, MNV3_SIZE), method=tf.image.ResizeMethod.BILINEAR
        )
        img224 = tf.cast(img224, tf.float32) / 255.0
        img192 = tf.cast(img192, tf.float32) / 255.0
        img224 = tf.ensure_shape(img224, (RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3))
        img192 = tf.ensure_shape(img192, (MNV3_SIZE, MNV3_SIZE, 3))
        return img224, img192

    BATCH_SIZE = 64  # safe speed-up: inference-only; semantics unchanged
    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .with_options(options)
        .map(_load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_and_preprocess(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_description)
        img = tf.io.decode_jpeg(ex["image"], channels=3)

        img224 = tf.image.resize(
            img,
            (RESNET_VGG_SIZE, RESNET_VGG_SIZE),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        img192 = tf.image.resize(
            img, (MNV3_SIZE, MNV3_SIZE), method=tf.image.ResizeMethod.BILINEAR
        )

        img224 = tf.cast(img224, tf.float32) / 255.0
        img192 = tf.cast(img192, tf.float32) / 255.0

        img224 = tf.ensure_shape(img224, (RESNET_VGG_SIZE, RESNET_VGG_SIZE, 3))
        img192 = tf.ensure_shape(img192, (MNV3_SIZE, MNV3_SIZE, 3))
        return ex["image_name"], img224, img192

    BATCH_SIZE = 64  # safe speed-up: inference-only; semantics unchanged
    ds = (
        tf.data.TFRecordDataset(
            tfrec_files,
            num_parallel_reads=AUTOTUNE,
            compression_type=None,
        )
        .with_options(options)
        .map(_parse_and_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )


@tf.function(reduce_retracing=True)
def _predict_batch(x224, x192):
    p1 = model1(x224, training=False)
    p2 = model2(x224, training=False)
    p3 = model3(x192, training=False)
    p = p1 + p2 + p3
    return tf.argmax(p, axis=1, output_type=tf.int64)


pred_labels = np.empty((n,), dtype=np.int64)

if tfrec_files:
    filled = 0
    for names_batch, imgs224_batch, imgs192_batch in ds:
        preds_batch = _predict_batch(imgs224_batch, imgs192_batch).numpy()
        names_np = names_batch.numpy()
        for nm, pr in zip(names_np, preds_batch):
            if isinstance(nm, (bytes, np.bytes_)):
                nm = nm.decode("utf-8")
            else:
                nm = str(nm)
            idx = id_to_idx.get(nm)
            if idx is not None:
                pred_labels[idx] = pr
                filled += 1
    if filled != n:
        missing = np.where(pred_labels == pred_labels.dtype.type(0))[0]
        print(f"Warning: only filled {filled}/{n} predictions from TFRecords.")
else:
    idx = 0
    for x224_batch, x192_batch in ds:
        batch_pred = _predict_batch(x224_batch, x192_batch).numpy()
        bs = batch_pred.shape[0]
        pred_labels[idx : idx + bs] = batch_pred
        idx += bs

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels.astype(int)}
)
my_submission.to_csv(OUT_PATH, index=False)

print("Wrote submission:", OUT_PATH)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Fallback used:", fallback_used)
print("Used TFRecords:", bool(tfrec_files))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1400046856.py in <cell line: 0>()
    108     filled = 0
    109     for names_batch, imgs224_batch, imgs192_batch in ds:
--> 110         preds_batch = _predict_batch(imgs224_batch, imgs192_batch).numpy()
    111         # bytes -> str once per element (batch size is moderate; total only 2676)
    112         names_np = names_batch.numpy()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_fileot30n_cs.py in tf___predict_batch(x224, x192)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 p1 = ag__.converted_call(ag__.ld(model1), (ag__.ld(x224),), dict(training=False), fscope)
     11                 p2 = ag__.converted_call(ag__.ld(model2), (ag__.ld(x224),), dict(training=False), fscope)
     12                 p3 = ag__.converted_call(ag__.ld(model3), (ag__.ld(x192),), dict(training=False), fscope)

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

    File "/tmp/ipykernel_11/1400046856.py", line 97, in _predict_batch  *
        p1 = model1(x224, training=False)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py", line 245, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "functional" is incompatible with the layer: expected shape=(None, 512, 512, 3), found shape=(64, 224, 224, 3)
