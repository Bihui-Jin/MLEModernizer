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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.9815668202764976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_hub as hub  # used only for custom_objects when loading certain models
except Exception:
    hub = None

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

try:
    import scipy.stats as ss
except Exception:
    ss = None

from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(False)

DATA_DIR = "../input/paddy-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("TF version:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def random_cut_out(images):
    if tfa is None:
        return images
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)


@tf.function
def center_crop_and_random_augmentations_tf(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image.numpy()




## === cell 2

BATCH_SIZE = 16
IMG_SIZE_256 = (256, 256)
IMG_SIZE_300 = (300, 300)
VAL_SPLIT = 0.2

AUTOTUNE = tf.data.AUTOTUNE

train_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

train_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

num_classes = len(train_ds_256.class_names)


def _prep_train_256(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = center_crop_and_random_augmentations_tf(x)
    return x, y


def _prep_valid(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y


def _prep_train_300(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = center_crop_and_random_augmentations_tf(x)
    return x, y


train_datagen = train_ds_256.map(_prep_train_256, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)
valid_datagen = valid_ds_256.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

train_datagen_300 = train_ds_300.map(
    _prep_train_300, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
valid_datagen_300 = valid_ds_300.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

print("Classes:", train_ds_256.class_names)
print("num_classes:", num_classes)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3401810801.py in <cell line: 0>()
     88 
     89 
---> 90 train_datagen = train_ds_256.map(_prep_train_256, num_parallel_calls=AUTOTUNE).prefetch(
     91     AUTOTUNE
     92 )

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

/tmp/__autograph_generated_fileuemet_14.py in tf___prep_train(x, y)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 x = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(x), ag__.ld(tf).float32), None, fscope) / 255.0
---> 11                 x = ag__.converted_call(ag__.ld(center_crop_and_random_augmentations_tf), (ag__.ld(x),), None, fscope)
     12                 try:
     13                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filevtdhty7o.py in tf__center_crop_and_random_augmentations_tf(image)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 image = ag__.converted_call(ag__.ld(tf).convert_to_tensor, (ag__.ld(image),), None, fscope)
---> 11                 image = ag__.converted_call(ag__.ld(tf).image.random_crop, (ag__.ld(image), (256, 256, 3)), None, fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).image.random_brightness, (ag__.ld(image), 0.2), None, fscope)
     13                 image = ag__.converted_call(ag__.ld(tf).image.random_contrast, (ag__.ld(image), 0.5, 2.0), None, fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/3401810801.py", line 73, in _prep_train_256  *
        x = center_crop_and_random_augmentations_tf(x)
    File "/tmp/ipykernel_11/2291438813.py", line 14, in center_crop_and_random_augmentations_tf  *
        image = tf.image.random_crop(image, (256, 256, 3))

    ValueError: Dimensions must be equal, but are 4 and 3 for '{{node random_crop/GreaterEqual}} = GreaterEqual[T=DT_INT32](random_crop/Shape, random_crop/size)' with input shapes: [4], [3].


## === cell 3


def _list_test_files(test_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in os.listdir(test_dir):
        if name.lower().endswith(exts):
            files.append(os.path.join(test_dir, name))
    files.sort()
    return files


test_files = _list_test_files(TEST_DIR)
print("Num test files:", len(test_files))
if len(test_files) == 0:
    raise RuntimeError("No test images found in TEST_DIR")


def _decode_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    return img


def _make_test_ds(target_hw):
    ds = tf.data.Dataset.from_tensor_slices(test_files)
    ds = ds.map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda x: tf.image.resize(x, target_hw, method="bilinear", antialias=False),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


test_data_256 = _make_test_ds(IMG_SIZE_256)
test_data_300 = _make_test_ds(IMG_SIZE_300)




## === cell 4
def _custom_objects_for_load():
    if hub is not None:
        return {"KerasLayer": hub.KerasLayer}
    return {}


def safe_load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(
            path, custom_objects=_custom_objects_for_load()
        )
    return None


m1 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5"
)
m2 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"
)
m3 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5"
)
m4 = safe_load_model("../input/paddydocoutputs/model_resnet150.hdf5")
m5 = safe_load_model(
    "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"
)
m6 = safe_load_model("../input/notebooka9ca40495e/model_effnet_s.hdf5")

loaded = [m is not None for m in [m1, m2, m3, m4, m5, m6]]
print("Loaded ensemble models:", loaded, "count:", sum(loaded))

if sum(loaded) == 0:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(256, 256, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    fallback_model = tf.keras.Model(inp, out)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS = 3
    fallback_model.fit(
        train_datagen,
        validation_data=valid_datagen,
        epochs=EPOCHS,
        verbose=1,
    )

    m1, m2, m3, m4, m5, m6 = fallback_model, None, None, None, None, None




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1662380555.py in <cell line: 0>()
     54     EPOCHS = 3
     55     fallback_model.fit(
---> 56         train_datagen,
     57         validation_data=valid_datagen,
     58         epochs=EPOCHS,

NameError: name 'train_datagen' is not defined

## === cell 5
def _needs_300(model):
    try:
        ish = model.input_shape
        if isinstance(ish, list):
            ish = ish[0]
        return ish[1] == 300 and ish[2] == 300
    except Exception:
        return False


model_train_test_score = []
for model in [m1, m2, m3, m4, m5, m6]:
    if model is None:
        model_train_test_score.append(None)
        continue
    ds = valid_datagen_300 if _needs_300(model) else valid_datagen
    model_train_test_score.append(model.evaluate(ds, verbose=0))

model_train_test_score




## === cell 6
def predict_model(model, ds256, ds300):
    if model is None:
        return None
    ds = ds300 if _needs_300(model) else ds256
    return model.predict(ds, verbose=1)


m1_p = predict_model(m1, test_data_256, test_data_300)
m2_p = predict_model(m2, test_data_256, test_data_300)
m3_p = predict_model(m3, test_data_256, test_data_300)
m4_p = predict_model(m4, test_data_256, test_data_300)
m5_p = predict_model(m5, test_data_256, test_data_300)
m6_p = predict_model(m6, test_data_256, test_data_300)

pred_probas = [p for p in [m1_p, m2_p, m3_p, m4_p, m5_p, m6_p] if p is not None]
print("Num prediction arrays:", len(pred_probas))
print("Prediction shape example:", pred_probas[0].shape if pred_probas else None)




## === cell 7
class_indices = {
    "bacterial_leaf_blight": 0,
    "bacterial_leaf_streak": 1,
    "bacterial_panicle_blight": 2,
    "blast": 3,
    "brown_spot": 4,
    "dead_heart": 5,
    "downy_mildew": 6,
    "hispa": 7,
    "normal": 8,
    "tungro": 9,
}
inverse_map = {v: k for k, v in class_indices.items()}




## === cell 8
if len(pred_probas) == 0:
    raise RuntimeError("No model predictions available; cannot create submission.")

avg_proba = np.mean(np.stack(pred_probas, axis=0), axis=0)
pred_idx = np.argmax(avg_proba, axis=1)
pred_label = [inverse_map[int(i)] for i in pred_idx]

image_ids = pd.Series([os.path.basename(p) for p in test_files])

pred_df = pd.DataFrame({"image_id": image_ids, "label": pred_label})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

pred_df["label"] = pred_df["label"].fillna("normal")

pred_df.head(), pred_df.shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3044706882.py in <cell line: 0>()
      1 if len(pred_probas) == 0:
----> 2     raise RuntimeError("No model predictions available; cannot create submission.")
      3 
      4 avg_proba = np.mean(np.stack(pred_probas, axis=0), axis=0)
      5 pred_idx = np.argmax(avg_proba, axis=1)

RuntimeError: No model predictions available; cannot create submission.

## === cell 9
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(pred_df.columns.tolist())
print(pred_df.isna().sum())
print(pred_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2121242872.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 pred_df.to_csv(submission_path, index=False)
      3 print("Wrote:", submission_path)
      4 print(pred_df.columns.tolist())
      5 print(pred_df.isna().sum())

NameError: name 'pred_df' is not defined
