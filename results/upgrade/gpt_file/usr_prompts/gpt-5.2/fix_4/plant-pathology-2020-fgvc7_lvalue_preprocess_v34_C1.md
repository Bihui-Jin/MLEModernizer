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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.44246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras

tqdm.pandas()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.config.experimental.enable_op_determinism(True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
USE_TPU = "TPU_NAME" in os.environ
if USE_TPU:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.MirroredStrategy()

print("Using strategy:", type(strategy).__name__)




## === cell 2
IMAGE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
assert (
    sub.columns.tolist() == ["image_id"] + TARGET_COLS
), "Unexpected submission columns"
assert (
    train_data.columns.tolist() == ["image_id"] + TARGET_COLS
), "Unexpected train columns"
assert test_data.columns.tolist() == ["image_id"], "Unexpected test columns"

print("Train:", train_data.shape, "Test:", test_data.shape)




## === cell 3
def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = cv2.GC_PR_FGD
    mask[2 * h // 5 : 3 * h // 5, 2 * w // 5 : 3 * w // 5] = cv2.GC_FGD
    return mask


def remove_background(image, h=136, w=205):
    orig_image = image
    image = cv2.resize(image, (w, h))
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    h0, w0 = orig_image.shape[:2]
    mask_binary = cv2.resize(mask_binary, (w0, h0))
    result = cv2.bitwise_and(orig_image, orig_image, mask=mask_binary)
    return result




## === cell 4
def rotate(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]
    x = tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    return tf.image.resize(x, shape)


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


_SCALES = np.arange(0.8, 1.0, 0.01, dtype=np.float32)
_BOXES = np.zeros((len(_SCALES), 4), dtype=np.float32)
for i, scale in enumerate(_SCALES):
    x1 = y1 = 0.5 - (0.5 * float(scale))
    x2 = y2 = 0.5 + (0.5 * float(scale))
    _BOXES[i] = [x1, y1, x2, y2]
_BOX_INDICES = np.zeros(len(_SCALES), dtype=np.int32)

_ZOOM_BOXES_T = tf.constant(_BOXES, dtype=tf.float32)
_ZOOM_BOXIND_T = tf.constant(_BOX_INDICES, dtype=tf.int32)
_NUM_SCALES = len(_SCALES)


def zoom(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)[:-1]

    def random_crop(img):
        crops = tf.image.crop_and_resize(
            [img], boxes=_ZOOM_BOXES_T, box_indices=_ZOOM_BOXIND_T, crop_size=shape
        )
        return crops[
            tf.random.uniform(shape=[], minval=0, maxval=_NUM_SCALES, dtype=tf.int32)
        ]

    choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
    return x




## === cell 5
def get_data_generators(
    preprocess=True, augment=True, IMAGE_SIZE=(408, 615), nfolds=5, batch_size=32
):
    def load_image(image_id):
        file_path = os.path.join(IMAGE_PATH, image_id + ".jpg")
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        image = cv2.resize(image, IMAGE_SIZE[::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if preprocess:
            image = remove_background(image)
        return image

    print("Preprocessing training images...")
    train_ids = train_data["image_id"].tolist()
    train_images = np.stack(
        [load_image(i) for i in tqdm(train_ids, total=len(train_ids))]
    )

    labels = train_data[TARGET_COLS].values.astype(np.float32)

    base = tf.data.Dataset.from_tensor_slices((train_images, labels))

    def map_base(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = (
            tf.cast(label, tf.float32) + 0.01
        ) / 1.04  # preserve original smoothing
        return image, label

    base = base.map(
        map_base,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    ).cache()

    if augment:

        def aug_map(image, label):
            image = flip(image)
            image = color(image)
            image = rotate(image)
            image = zoom(image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        base = base.map(
            aug_map,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True,
        )

    n = len(train_data)
    fold_size = n // nfolds
    fold_ranges = []
    for idx in range(nfolds):
        start = idx * fold_size
        end = (idx + 1) * fold_size if idx < nfolds - 1 else n
        fold_ranges.append((start, end))

    indexed = tf.data.Dataset.from_tensor_slices(tf.range(n, dtype=tf.int32)).zip(base)

    folds = []
    for start, end in fold_ranges:
        start_t = tf.constant(start, tf.int32)
        end_t = tf.constant(end, tf.int32)

        def in_fold(i, _):
            return tf.logical_and(i >= start_t, i < end_t)

        fold_ds = indexed.filter(in_fold).map(
            lambda _, xy: xy,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True,
        )
        folds.append(fold_ds)

    print("Preprocessing test images...")
    test_ids = test_data["image_id"].tolist()
    test_images = np.stack([load_image(i) for i in tqdm(test_ids, total=len(test_ids))])

    test = tf.data.Dataset.from_tensor_slices(test_images)
    test = test.map(
        lambda image: tf.cast(image, tf.float32) / 255.0,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    )
    test = test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    try:
        plt.figure(figsize=(6, 4))
        plt.imshow(train_images[0])
        plt.axis("off")
        plt.show()
    except Exception:
        pass

    return folds, test




## === cell 6
BATCH_SIZE = 4
folds, test = get_data_generators(preprocess=False, augment=True, batch_size=BATCH_SIZE)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2386728591.py in <cell line: 0>()
      1 BATCH_SIZE = 4
----> 2 folds, test = get_data_generators(preprocess=False, augment=True, batch_size=BATCH_SIZE)
      3 
      4 

/tmp/ipykernel_11/649038773.py in get_data_generators(preprocess, augment, IMAGE_SIZE, nfolds, batch_size)
     79             return tf.logical_and(i >= start_t, i < end_t)
     80 
---> 81         fold_ds = indexed.filter(in_fold).map(
     82             lambda _, xy: xy,
     83             num_parallel_calls=tf.data.experimental.AUTOTUNE,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in filter(self, predicate, name)
   2561     # pylint: disable=g-import-not-at-top,protected-access
   2562     from tensorflow.python.data.ops import filter_op
-> 2563     return filter_op._filter(self, predicate, name)
   2564     # pylint: enable=g-import-not-at-top,protected-access
   2565 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in _filter(input_dataset, predicate, name)
     23 
     24 def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
---> 25   return _FilterDataset(input_dataset, predicate, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in __init__(self, input_dataset, predicate, use_legacy_function, name)
     36     """See `Dataset.filter` for details."""
     37     self._input_dataset = input_dataset
---> 38     wrapped_func = structured_function.StructuredFunctionWrapper(
     39         predicate,
     40         self._transformation_name(),

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileroovmb78.py in tf__in_fold(i, _)
     12                 try:
     13                     do_return = True
---> 14                     retval_ = ag__.converted_call(ag__.ld(tf).logical_and, (ag__.ld(i) >= ag__.ld(start_t), ag__.ld(i) < ag__.ld(end_t)), None, fscope)
     15                 except:
     16                     do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_math_operator_overrides.py in wrapper(x, y, *args, **kwargs)
    148   def wrapper(x, y, *args, **kwargs):
    149     x, y = override_binary_operator.maybe_promote_tensors(x, y)
--> 150     return fn(x, y, *args, **kwargs)
    151 
    152   return tf_decorator.make_decorator(fn, wrapper)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_math_ops.py in greater_equal(x, y, name)
   4403   # Add nodes to the TensorFlow graph.
   4404   try:
-> 4405     _, _, _op, _outputs = _op_def_library._apply_op_helper(
   4406         "GreaterEqual", x=x, y=y, name=name)
   4407   except (TypeError, ValueError):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    650       if input_arg.type_attr in attrs:
    651         if attrs[input_arg.type_attr] != attr_value:
--> 652           raise TypeError(
    653               f"Input '{input_name}' of '{op_type_name}' Op has type "
    654               f"{dtypes.as_dtype(attr_value).name} that does not match type "

TypeError: in user code:

    File "/tmp/ipykernel_11/649038773.py", line 79, in in_fold  *
        return tf.logical_and(i >= start_t, i < end_t)

    TypeError: Input 'y' of 'GreaterEqual' Op has type int32 that does not match type float32 of argument 'x'.


## === cell 7
def get_model():
    model = keras.Sequential()
    model.add(
        keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            input_shape=(408, 615, 3),
            pooling=None,
        )
    )
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.summary()
    return model




## === cell 8
with strategy.scope():
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy],
    )




## === cell 9
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

callbacks = [
    EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=0.000001),
    CSVLogger("log.csv", append=True, separator=";"),
]


def get_train_val_split(val_fold):
    train_folds = folds[:val_fold] + folds[val_fold + 1 :]
    train_ds = train_folds[0]
    for fold in train_folds[1:]:
        train_ds = train_ds.concatenate(fold)
    val_ds = folds[val_fold]
    return train_ds, val_ds


VAL_FOLD = 0
train_ds, val_ds = get_train_val_split(VAL_FOLD)

train_ds = train_ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

n = len(train_data)
fold_size = n // 5
n_val = fold_size if VAL_FOLD < 4 else (n - 4 * fold_size)
n_train = n - n_val

steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
validation_steps = math.ceil(n_val / BATCH_SIZE)

history = None
try:
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_ds,
        validation_steps=validation_steps,
        validation_freq=1,
        verbose=1,
        callbacks=callbacks,
    )
except tf.errors.ResourceExhaustedError as e:
    print(
        "WARNING: OOM during training; proceeding to prediction with current weights."
    )
    print(str(e)[:500])




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2078971182.py in <cell line: 0>()
     20 
     21 VAL_FOLD = 0
---> 22 train_ds, val_ds = get_train_val_split(VAL_FOLD)
     23 
     24 train_ds = train_ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)

/tmp/ipykernel_11/2078971182.py in get_train_val_split(val_fold)
     11 
     12 def get_train_val_split(val_fold):
---> 13     train_folds = folds[:val_fold] + folds[val_fold + 1 :]
     14     train_ds = train_folds[0]
     15     for fold in train_folds[1:]:

NameError: name 'folds' is not defined

## === cell 10
test_pr = model.predict(test, verbose=1)

test_pr = np.asarray(test_pr)
if test_pr.ndim != 2 or test_pr.shape[1] != 4:
    raise ValueError(f"Unexpected prediction shape: {test_pr.shape}")

if test_pr.shape[0] != len(test_data):
    raise ValueError(
        f"Prediction count mismatch: got {test_pr.shape[0]} preds for {len(test_data)} test rows"
    )

submission = pd.DataFrame({"image_id": test_data["image_id"].values})
for i, c in enumerate(TARGET_COLS):
    submission[c] = test_pr[:, i]

submission = submission[["image_id"] + TARGET_COLS]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4138852627.py in <cell line: 0>()
----> 1 test_pr = model.predict(test, verbose=1)
      2 
      3 test_pr = np.asarray(test_pr)
      4 if test_pr.ndim != 2 or test_pr.shape[1] != 4:
      5     raise ValueError(f"Unexpected prediction shape: {test_pr.shape}")

NameError: name 'test' is not defined
