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

0.8942278634028408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from tensorflow.keras import backend as K
import glob

from tensorflow.keras.applications import EfficientNetB5

try:
    from kaggle_datasets import KaggleDatasets  # noqa: F401
except Exception as e:
    print(f"WARNING: kaggle_datasets import failed (unused) and will be skipped: {e}")

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 20
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
LR = 0.0001
TTA = 10
VERBOSE = 2
N_CLASSES = 5

TEST_FILENAMES = "../input/cassava-leaf-disease-classification/test_images/*.jpg"



## === cell 3
_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_slack = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass


@tf.function(jit_compile=True)
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function(jit_compile=True)
def read_image(file_path):
    image_name = tf.strings.regex_replace(file_path, r"^.*[\\/]", "")
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, image_name


NUM_TESTING_IMAGES = len(
    os.listdir("../input/cassava-leaf-disease-classification/test_images/")
)




## === cell 4
def get_model(weights_mode="cassava_or_imagenet"):
    """
    Architecture and compile settings unchanged: EfficientNetB5 backbone + GAP + Dropout + Dense softmax,
    Adam LR, CategoricalCrossentropy(label_smoothing=0.4), CategoricalAccuracy.
    """
    with strategy.scope():
        inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))

        effnet_weights = None
        if weights_mode == "imagenet":
            effnet_weights = "imagenet"

        x = EfficientNetB5(weights=effnet_weights, include_top=False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=[inp], outputs=[output])

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.4)],
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model


def _predict_tta_mean_single_call(model, tta_image_ds, steps_per_pass, tta):
    total_steps = int(tta) * int(steps_per_pass)
    probs_all = model.predict(tta_image_ds, steps=total_steps, verbose=0)
    probs_all = probs_all[: int(tta) * NUM_TESTING_IMAGES]
    probs_all = probs_all.reshape((int(tta), NUM_TESTING_IMAGES, N_CLASSES))
    return probs_all.mean(axis=0)


def inference(model_paths):
    prediction = np.zeros((NUM_TESTING_IMAGES, N_CLASSES), dtype=np.float32)
    steps_per_pass = int(math.ceil(NUM_TESTING_IMAGES / BATCH_SIZE))

    test_filepaths = tf.io.gfile.glob(TEST_FILENAMES)
    test_filepaths.sort()
    image_name = np.asarray([os.path.basename(p) for p in test_filepaths], dtype="U")

    print("Building cached decoded test dataset (single decode pass)...")
    decoded_cached = tf.data.Dataset.from_tensor_slices(test_filepaths).with_options(
        _DATASET_OPTIONS
    )
    decoded_cached = decoded_cached.map(read_image, num_parallel_calls=AUTO).cache()

    @tf.function(jit_compile=True)
    def _augment_batch_vectorized(images, names):
        batch_size = tf.shape(images)[0]
        h = tf.cast(tf.shape(images)[1], tf.float32)
        w = tf.cast(tf.shape(images)[2], tf.float32)

        p_spatial = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_rotate = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_1 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_2 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_3 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_crop = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)

        x = tf.image.random_flip_left_right(images)
        x = tf.image.random_flip_up_down(x)

        x_t = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.where(p_spatial[:, None, None, None] > 0.75, x_t, x)

        r3 = tf.image.rot90(x, k=3)
        r2 = tf.image.rot90(x, k=2)
        r1 = tf.image.rot90(x, k=1)
        x = tf.where(p_rotate[:, None, None, None] > 0.75, r3, x)
        x = tf.where(
            tf.logical_and(p_rotate > 0.5, p_rotate <= 0.75)[:, None, None, None], r2, x
        )
        x = tf.where(
            tf.logical_and(p_rotate > 0.25, p_rotate <= 0.5)[:, None, None, None], r1, x
        )

        sat = tf.image.random_saturation(x, lower=0.7, upper=1.3)
        x = tf.where(p_pixel_1[:, None, None, None] >= 0.4, sat, x)

        con = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        x = tf.where(p_pixel_2[:, None, None, None] >= 0.4, con, x)

        bri = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.where(p_pixel_3[:, None, None, None] >= 0.4, bri, x)

        def _central_crop(frac):
            y = tf.image.central_crop(x, central_fraction=frac)
            y = tf.image.resize(y, size=IMAGE_SIZE)
            y = tf.reshape(y, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            return y

        def _random_crop_branch_vectorized():
            minsz = tf.cast(tf.math.round(h * 0.8), tf.int32)
            maxsz = tf.cast(h, tf.int32)
            crop_sizes = tf.random.uniform([batch_size], minsz, maxsz, dtype=tf.int32)
            crop_sizes_f = tf.cast(crop_sizes, tf.float32)

            max_off = tf.maximum(tf.cast(h, tf.int32) - crop_sizes, 0)
            off_y = tf.random.uniform([batch_size], 0, max_off + 1, dtype=tf.int32)
            off_x = tf.random.uniform([batch_size], 0, max_off + 1, dtype=tf.int32)

            y1 = tf.cast(off_y, tf.float32) / h
            x1 = tf.cast(off_x, tf.float32) / w
            y2 = (tf.cast(off_y, tf.float32) + crop_sizes_f) / h
            x2 = (tf.cast(off_x, tf.float32) + crop_sizes_f) / w
            boxes = tf.stack([y1, x1, y2, x2], axis=1)

            box_ind = tf.range(batch_size, dtype=tf.int32)
            y = tf.image.crop_and_resize(
                x, boxes, box_ind, crop_size=IMAGE_SIZE, method="bilinear"
            )
            y = tf.reshape(y, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            return y

        x0 = tf.image.resize(x, size=IMAGE_SIZE)
        x0 = tf.reshape(x0, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

        x07 = _central_crop(0.7)
        x08 = _central_crop(0.8)
        x09 = _central_crop(0.9)
        xrc = _random_crop_branch_vectorized()

        out = x0
        out = tf.where((p_crop > 0.9)[:, None, None, None], x07, out)
        out = tf.where(
            (tf.logical_and(p_crop > 0.8, p_crop <= 0.9))[:, None, None, None], x08, out
        )
        out = tf.where(
            (tf.logical_and(p_crop > 0.7, p_crop <= 0.8))[:, None, None, None], x09, out
        )
        out = tf.where(
            (tf.logical_and(p_crop > 0.4, p_crop <= 0.7))[:, None, None, None], xrc, out
        )

        return out, names

    tta_batched = decoded_cached.batch(BATCH_SIZE, drop_remainder=False)
    tta_batched = tta_batched.map(
        _augment_batch_vectorized, num_parallel_calls=AUTO
    ).repeat()
    tta_image = tta_batched.map(
        lambda image, image_name: image, num_parallel_calls=AUTO
    ).prefetch(AUTO)

    if not model_paths:
        print(
            "WARNING: No model .h5 files found in ../input/cassava-models/*.h5. "
            "Falling back to EfficientNetB5(weights='imagenet') to generate a valid submission."
        )
        K.clear_session()
        model = get_model(weights_mode="imagenet")
        prediction += _predict_tta_mean_single_call(
            model, tta_image, steps_per_pass=steps_per_pass, tta=TTA
        )
    else:
        inv_n_models = 1.0 / len(model_paths)
        for fold, model_path in enumerate(model_paths):
            print("\n" + "-" * 50)
            print(f"Predicting fold {fold + 1} | loading: {model_path}")
            K.clear_session()
            model = get_model(weights_mode="cassava_or_imagenet")
            model.load_weights(model_path)

            prediction += (
                _predict_tta_mean_single_call(
                    model, tta_image, steps_per_pass=steps_per_pass, tta=TTA
                )
                * inv_n_models
            )

    sub = pd.DataFrame(
        {
            "image_id": image_name,
            "label": np.argmax(prediction, axis=-1).astype(np.int64),
        }
    )
    sub.to_csv("submission.csv", index=False)
    return image_name, prediction, sub


model_paths = sorted(glob.glob("../input/cassava-models/*.h5"))

image_name, prediction, sub = inference(model_paths)
sub

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/96354555.py in <cell line: 0>()
    196 model_paths = sorted(glob.glob("../input/cassava-models/*.h5"))
    197 
--> 198 image_name, prediction, sub = inference(model_paths)
    199 sub

/tmp/ipykernel_11/96354555.py in inference(model_paths)
    151     # Runtime: keep a single dataset pipeline and repeat it for TTA; prefetch overlaps CPU aug with model.
    152     tta_batched = decoded_cached.batch(BATCH_SIZE, drop_remainder=False)
--> 153     tta_batched = tta_batched.map(
    154         _augment_batch_vectorized, num_parallel_calls=AUTO
    155     ).repeat()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file6_uj04n4.py in tf___augment_batch_vectorized(images, names)
     82                 x08 = ag__.converted_call(ag__.ld(_central_crop), (0.8,), None, fscope)
     83                 x09 = ag__.converted_call(ag__.ld(_central_crop), (0.9,), None, fscope)
---> 84                 xrc = ag__.converted_call(ag__.ld(_random_crop_branch_vectorized), (), None, fscope)
     85                 out = ag__.ld(x0)
     86                 out = ag__.converted_call(ag__.ld(tf).where, ((ag__.ld(p_crop) > 0.9)[:, None, None, None], ag__.ld(x07), ag__.ld(out)), None, fscope)

/tmp/__autograph_generated_file6_uj04n4.py in _random_crop_branch_vectorized()
     60                         crop_sizes_f = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(crop_sizes), ag__.ld(tf).float32), None, fscope_2)
     61                         max_off = ag__.converted_call(ag__.ld(tf).maximum, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope_2) - ag__.ld(crop_sizes), 0), None, fscope_2)
---> 62                         off_y = ag__.converted_call(ag__.ld(tf).random.uniform, ([ag__.ld(batch_size)], 0, ag__.ld(max_off) + 1), dict(dtype=ag__.ld(tf).int32), fscope_2)
     63                         off_x = ag__.converted_call(ag__.ld(tf).random.uniform, ([ag__.ld(batch_size)], 0, ag__.ld(max_off) + 1), dict(dtype=ag__.ld(tf).int32), fscope_2)
     64                         y1 = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(off_y), ag__.ld(tf).float32), None, fscope_2) / ag__.ld(h)

ValueError: in user code:

    File "/tmp/ipykernel_11/96354555.py", line 113, in _random_crop_branch_vectorized  *
        off_y = tf.random.uniform([batch_size], 0, max_off + 1, dtype=tf.int32)

    ValueError: maxval must be a scalar; got a tensor of shape [?] for '{{node random_uniform_10}} = RandomUniformInt[T=DT_INT32, Tout=DT_INT32, seed=123, seed2=12](random_uniform_10/shape, random_uniform_10/min, add)' with input shapes: [1], [], [?].
