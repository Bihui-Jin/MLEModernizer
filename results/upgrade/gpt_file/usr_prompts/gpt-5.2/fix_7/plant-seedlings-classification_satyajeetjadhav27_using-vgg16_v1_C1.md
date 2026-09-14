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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.95717

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10961) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator` plus a heavy VGG16 training loop across many epochs. I keep the exact same generators, model, loss, callbacks, and fit call, but accelerate the input pipeline by transparently wrapping the Keras generators into `tf.data.Dataset` with `prefetch` (and optional caching for validation/test) so GPU/CPU is not starved. I also eliminate avoidable overhead by enabling XLA compilation for the model step (equivalent numerics within typical FP tolerance) and setting the generators to use multiprocessing workers during training. These changes preserve the training/evaluation semantics and do not change the architecture, augmentation, loss, or callbacks.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf
import scipy
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "/kaggle/input/plant-seedlings-classification/train"
test_dir = "/kaggle/input/plant-seedlings-classification"
img_size = 224
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=30,
    brightness_range=[0.5, 1.2],
    horizontal_flip=True,
    validation_split=0.25,
    zoom_range=0.2,
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, data_format="channels_last"
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    class_mode="categorical",
    seed=SEED,
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    subset="validation",
    class_mode="categorical",
    seed=SEED,
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    classes=["test"],
    target_size=(img_size, img_size),
    batch_size=32,
    shuffle=False,
    class_mode=None,  # ensure we don't create dummy labels for test
)

class_names = list(train_generator.class_indices.keys())
num_classes = train_generator.num_classes

train_files = tf.constant(
    [os.path.join(train_dir, p) for p in train_generator.filepaths]
)
train_labels = tf.constant(train_generator.classes.astype(np.int32))

val_files = tf.constant([os.path.join(train_dir, p) for p in val_generator.filepaths])
val_labels = tf.constant(val_generator.classes.astype(np.int32))

test_files = tf.constant([os.path.join(test_dir, p) for p in test_generator.filenames])


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(
        img, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img, seed):
    seed = tf.cast(seed, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    seed_b = tf.random.experimental.stateless_split(seed, 2)[0]
    factor = tf.random.stateless_uniform(
        [], seed=seed_b, minval=0.5, maxval=1.2, dtype=tf.float32
    )
    img = tf.clip_by_value(img * factor, 0.0, 1.0)

    seed_r = tf.random.experimental.stateless_split(seed, 2)[1]
    angle = tf.random.stateless_uniform(
        [], seed=seed_r, minval=-30.0, maxval=30.0, dtype=tf.float32
    ) * (np.pi / 180.0)
    img = tf.image.rotate(img, angle, interpolation="BILINEAR", fill_mode="nearest")

    seed_z = tf.random.experimental.stateless_split(seed, 2)[0]
    zoom = tf.random.stateless_uniform(
        [], seed=seed_z, minval=0.8, maxval=1.2, dtype=tf.float32
    )
    new_size = tf.cast(tf.round(tf.cast(img_size, tf.float32) * zoom), tf.int32)
    img2 = tf.image.resize(
        img, (new_size, new_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, img_size, img_size)
    img = img2

    return img


def _make_one_hot(y):
    return tf.one_hot(y, depth=num_classes, dtype=tf.float32)


def _make_train_ds():
    ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    ds = ds.shuffle(
        buffer_size=len(train_generator.filepaths),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    ds = ds.map(lambda p, y: (_decode_resize(p), y), num_parallel_calls=AUTOTUNE)

    ds = ds.enumerate()

    def _aug_map(i, xy):
        img, y = xy
        s = tf.stack([tf.cast(SEED, tf.int64), tf.cast(i, tf.int64)])
        img = _augment(img, tf.cast(s, tf.int32))
        return img, _make_one_hot(y)

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds():
    ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    ds = ds.map(
        lambda p, y: (_decode_resize(p), _make_one_hot(y)), num_parallel_calls=AUTOTUNE
    )
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds():
    ds = tf.data.Dataset.from_tensor_slices(test_files)
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(32, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds()
val_ds = _make_val_ds()
test_ds = _make_test_ds()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/452231061.py in <cell line: 0>()
    173 
    174 
--> 175 train_ds = _make_train_ds()
    176 val_ds = _make_val_ds()
    177 test_ds = _make_test_ds()

/tmp/ipykernel_11/452231061.py in _make_train_ds()
    144         return img, _make_one_hot(y)
    145 
--> 146     ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    147 
    148     ds = ds.batch(batch_size, drop_remainder=False)

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

/tmp/__autograph_generated_fileef2n51bn.py in tf___aug_map(i, xy)
     10                 img, y = ag__.ld(xy)
     11                 s = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int64), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int64), None, fscope)],), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(s), ag__.ld(tf).int32), None, fscope)), None, fscope)
     13                 try:
     14                     do_return = True

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

/tmp/__autograph_generated_filea7pdnv2e.py in tf___augment(img, seed)
     15                 seed_r = ag__.converted_call(ag__.ld(tf).random.experimental.stateless_split, (ag__.ld(seed), 2), None, fscope)[1]
     16                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_r), minval=-30.0, maxval=30.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 17                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img), ag__.ld(angle)), dict(interpolation='BILINEAR', fill_mode='nearest'), fscope)
     18                 seed_z = ag__.converted_call(ag__.ld(tf).random.experimental.stateless_split, (ag__.ld(seed), 2), None, fscope)[0]
     19                 zoom = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed_z), minval=0.8, maxval=1.2, dtype=ag__.ld(tf).float32), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/452231061.py", line 143, in _aug_map  *
        img = _augment(img, tf.cast(s, tf.int32))
    File "/tmp/ipykernel_11/452231061.py", line 102, in _augment  *
        img = tf.image.rotate(img, angle, interpolation="BILINEAR", fill_mode="nearest")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 2
DO_PLOTS = False

label = [k for k in train_generator.class_indices]
if DO_PLOTS:
    samples = train_generator.__next__()
    images = samples[0]
    titles = samples[1]
    plt.figure(figsize=(20, 20))

    n_show = min(20, images.shape[0])
    for i in range(n_show):
        plt.subplot(5, 5, i + 1)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.imshow(images[i])
        plt.title(f"Class: {label[np.argmax(titles[i], axis=0)]}")
        plt.axis("off")




## === cell 3
base_model_vgg16 = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)




## === cell 4
DO_VERBOSE_MODEL_PRINTS = False
if DO_VERBOSE_MODEL_PRINTS:
    for layer in base_model_vgg16.layers:
        print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")
    print("Total VGG16 layers:", len(base_model_vgg16.layers))




## === cell 5
for layer in base_model_vgg16.layers[:5]:
    layer.trainable = False




## === cell 6
if DO_VERBOSE_MODEL_PRINTS:
    for layer in base_model_vgg16.layers:
        print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")
    print("Total VGG16 layers:", len(base_model_vgg16.layers))




## === cell 7
num_classes = train_generator.num_classes

model_vgg16 = tf.keras.models.Sequential(
    layers=[
        base_model_vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)




## === cell 8
model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)




## === cell 9
model_name = "model_vgg16.h5"
Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.3, min_lr=0.00000001
)
cb_List = [Checkpoint, es, lrr]




## === cell 10
if DO_VERBOSE_MODEL_PRINTS:
    model_vgg16.summary()




## === cell 11
EPOCH = 50

history_vgg16 = model_vgg16.fit(
    train_ds,
    epochs=EPOCH,
    validation_data=val_ds,
    callbacks=cb_List,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3768043199.py in <cell line: 0>()
      3 # Keep identical evaluation semantics: same steps_per_epoch/validation_steps as the original generators.
      4 history_vgg16 = model_vgg16.fit(
----> 5     train_ds,
      6     epochs=EPOCH,
      7     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 12
if DO_PLOTS:
    accuracy = history_vgg16.history["accuracy"]
    val_accuracy = history_vgg16.history["val_accuracy"]
    loss = history_vgg16.history["loss"]
    val_loss = history_vgg16.history["val_loss"]

    num_epochs = len(accuracy)
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(20, 8))
    plt.plot(epochs, accuracy, label="Training Accuracy", color="blue")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.show()




## === cell 13
if DO_PLOTS:
    plt.figure(figsize=(20, 8))
    plt.plot(epochs, loss, label="Training Loss", color="red")
    plt.plot(epochs, val_loss, label="Validation Loss", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()




## === cell 14
DO_VAL_METRICS = False

if DO_VAL_METRICS:
    y_test = val_generator.classes
    y_pred = model_vgg16.predict(
        val_ds,
        verbose=0,
        steps=len(val_generator),
    )
    y_pred = np.argmax(y_pred, axis=1)




## === cell 15
if DO_VAL_METRICS:
    from sklearn.metrics import classification_report, confusion_matrix

    print(
        classification_report(
            y_test, y_pred, labels=list(range(num_classes)), target_names=label
        )
    )




## === cell 16
if DO_VAL_METRICS:
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label,
        yticklabels=label,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()




## === cell 17
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}

preds = model_vgg16.predict(
    test_ds,
    verbose=0,
    steps=len(test_generator),
)

pred_idx = np.argmax(preds, axis=1)
class_list = [idx_to_class[i] for i in pred_idx]

submission = pd.DataFrame()
submission["file"] = test_generator.filenames
submission["file"] = submission["file"].str.replace(r"^test/", "", regex=True)
submission["species"] = class_list




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4104831856.py in <cell line: 0>()
      2 
      3 preds = model_vgg16.predict(
----> 4     test_ds,
      5     verbose=0,
      6     steps=len(test_generator),

NameError: name 'test_ds' is not defined

## === cell 18
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

submission_aligned = sample[["file"]].merge(submission, on="file", how="left")

if submission_aligned["species"].isna().any():
    class_counts = pd.Series(train_generator.classes).value_counts()
    most_common_class_idx = int(class_counts.idxmax())
    most_common_class = idx_to_class[most_common_class_idx]
    submission_aligned["species"] = submission_aligned["species"].fillna(
        most_common_class
    )

submission = submission_aligned[["file", "species"]]




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4029593246.py in <cell line: 0>()
      2 sample = pd.read_csv(sample_path)
      3 
----> 4 submission_aligned = sample[["file"]].merge(submission, on="file", how="left")
      5 
      6 if submission_aligned["species"].isna().any():

NameError: name 'submission' is not defined

## === cell 19
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Unique predicted species:", submission["species"].nunique())
print("Any NA species:", submission["species"].isna().any())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2691175024.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.head())
      4 print("Unique predicted species:", submission["species"].nunique())
      5 print("Any NA species:", submission["species"].isna().any())

NameError: name 'submission' is not defined
