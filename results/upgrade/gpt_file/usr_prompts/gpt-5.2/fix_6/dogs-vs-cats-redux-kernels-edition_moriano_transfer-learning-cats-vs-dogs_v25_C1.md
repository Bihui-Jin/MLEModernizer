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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.22786

# 6. Current score

0.73718

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.06312) has done: 'The timeout is dominated by slow Python-side image loading/resize in `build_batches()` (skimage+iterrows) plus extra non-essential work (per-image plotting, manual evaluate/predict loops) that repeats I/O. I keep the same model, training loop, augmentations, epochs, and prediction semantics, but make data input significantly faster by (1) using Keras generators’ built-in multiprocessing/prefetch and (2) rewriting `build_batches()` to use OpenCV decoding and `itertuples()` with preallocated NumPy arrays (same resize/scale). I also disable the expensive display/plot cells by default (they don’t affect submission accuracy) and avoid redundant validation “test” generator augmentation mismatch by keeping it unchanged but making it faster. All paths remain identical; outputs and training behavior remain equivalent aside from negligible float differences.'
- What this solution (achieved 0.73718) has done: 'The timeout is dominated by Python-level image loading/augmentation in `ImageDataGenerator` and your custom OpenCV generator, which both read/resize images on the fly with minimal pipelining. I keep the same VGG16 transfer model, same train/val split, same augmentations, same epochs/optimizer/loss, but switch the input pipelines to `tf.data` with parallel decode/resize, caching, and prefetch so the GPU/CPU stays busy. I also remove redundant array copies in the batch builder (used for prediction) and make prediction use a `tf.data` pipeline too, preserving identical rescale/resize semantics. These changes are equivalent in outputs (up to negligible float differences) but cut wall time substantially by reducing Python overhead and enabling parallelism.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

BASE_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test", "unknown")

if not os.path.exists(TEST_DIR):
    candidate_dirs = [
        os.path.join(BASE_DIR, "test", "test", "unknown"),
        os.path.join(BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "unknown"),
        os.path.join(
            BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "test", "unknown"
        ),
    ]
    for cd in candidate_dirs:
        if os.path.exists(cd):
            TEST_DIR = cd
            break

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print(
    "TRAIN_DIR subdirs:",
    os.listdir(TRAIN_DIR)[:10] if os.path.exists(TRAIN_DIR) else None,
)
print(
    "TEST_DIR sample:", os.listdir(TEST_DIR)[:10] if os.path.exists(TEST_DIR) else None
)



## === cell 1
from os import listdir

train_data = []
for cls in ["cat", "dog"]:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    label = "1" if cls == "dog" else "0"
    for file in listdir(cls_dir):
        rel_path = f"{cls}/{file}"  # relative to TRAIN_DIR
        train_data.append([rel_path, label])

df_all = pd.DataFrame(train_data, columns=["filename", "class"])
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(len(df_all) * 0.85)
train = df_all.iloc[:split_idx].copy()
test = df_all.iloc[split_idx:].copy()

print("Train size", len(train))
print("Val size", len(test))
for label in ["0", "1"]:
    print("------------")
    print("\tTrain has", len(train[train["class"] == label]), label)
    print("\tVal has", len(test[test["class"] == label]), label)



## === cell 2
import tf_keras as keras
import tensorflow as tf

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
@tf.function
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.BICUBIC
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img):
    angle = tf.random.uniform([], minval=-90.0, maxval=90.0, dtype=tf.float32) * (
        np.pi / 180.0
    )
    img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    img = tf.image.random_flip_left_right(img)
    return img


def make_train_val_datasets(train_df, val_df, batch_size):
    train_paths = (TRAIN_DIR + "/" + train_df["filename"].astype(str)).to_numpy()
    train_labels = train_df["class"].astype(np.float32).to_numpy()

    val_paths = (TRAIN_DIR + "/" + val_df["filename"].astype(str)).to_numpy()
    val_labels = val_df["class"].astype(np.float32).to_numpy()

    ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds_train = ds_train.shuffle(
        len(train_paths), seed=42, reshuffle_each_iteration=True
    )
    ds_train = ds_train.map(
        lambda p, y: (_augment(_decode_and_resize(p)), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    ds_val = ds_val.map(
        lambda p, y: (_decode_and_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_val = ds_val.batch(batch_size, drop_remainder=False)
    ds_val = ds_val.prefetch(AUTOTUNE)

    return ds_train, ds_val


train_ds, val_ds = make_train_val_datasets(train, test, BATCH_SIZE)


class _NWrap:
    def __init__(self, n):
        self.n = n


train_generator = _NWrap(len(train))
validation_generator = _NWrap(len(test))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2713745981.py in <cell line: 0>()
     57 
     58 
---> 59 train_ds, val_ds = make_train_val_datasets(train, test, BATCH_SIZE)
     60 
     61 

/tmp/ipykernel_11/2713745981.py in make_train_val_datasets(train_df, val_df, batch_size)
     37         len(train_paths), seed=42, reshuffle_each_iteration=True
     38     )
---> 39     ds_train = ds_train.map(
     40         lambda p, y: (_augment(_decode_and_resize(p)), y),
     41         num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_file2xusb2mp.py in <lambda>(p, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment, (ag__.converted_call(_decode_and_resize, (p,), None, lscope),), None, lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file2xusb2mp.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment, (ag__.converted_call(_decode_and_resize, (p,), None, lscope),), None, lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

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

/tmp/__autograph_generated_filei4fvrm1g.py in tf___augment(img)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([],), dict(minval=-90.0, maxval=90.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR'), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), None, fscope)
     13                 try:

AttributeError: in user code:

    File "/tmp/ipykernel_11/2713745981.py", line 40, in None  *
        lambda p, y: (_augment(_decode_and_resize(p)), y)
    File "/tmp/ipykernel_11/2713745981.py", line 21, in _augment  *
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 4
from tf_keras.applications import vgg16

model = vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
    pooling="max",
)



## === cell 5
for layer in model.layers[:-5]:
    layer.trainable = False



## === cell 6
from tf_keras.layers import Dense
from tf_keras.models import Sequential

transfer_model_vgg16 = Sequential()
for layer in model.layers:
    transfer_model_vgg16.add(layer)

transfer_model_vgg16.add(Dense(512, activation="relu"))
transfer_model_vgg16.add(Dense(1, activation="sigmoid"))

transfer_model_vgg16.summary()



## === cell 7
print("Skipping model_to_dot visualization (not available in this environment).")



## === cell 8
from tf_keras import optimizers

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

transfer_model_vgg16.compile(
    optimizer=adam,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 9
vgg16_model_history = transfer_model_vgg16.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1513509667.py in <cell line: 0>()
      2 # --- letting Keras infer steps avoids edge-case truncation and is equivalent for full-epoch training.
      3 vgg16_model_history = transfer_model_vgg16.fit(
----> 4     train_ds,
      5     validation_data=val_ds,
      6     epochs=5,

NameError: name 'train_ds' is not defined

## === cell 10
PLOT_DEBUG = False

if PLOT_DEBUG:
    from IPython.display import Image, display

    def plot_prediction(image_path, label):
        display(Image(filename=image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT))
        prediction = "dog"
        confidence = float(label)
        if confidence < 0.5:
            prediction = "cat"
            confidence = 1.0 - confidence
        legend = (
            "The image %s above is a %s with a confidence of %.2f%% (p_dog=%.6f)"
            % (
                image_path,
                prediction,
                confidence * 100,
                float(label),
            )
        )
        print(legend)

else:

    def plot_prediction(image_path, label):
        pass




## === cell 11
import cv2


def build_batches(
    df, has_labels=True, limit=500, batch_size=BATCH_SIZE, produce="images"
):
    """
    produce: "images" -> yields (X, y) if has_labels else yields X only
             "paths"  -> yields (paths, y) if has_labels else yields paths only
    Note: For has_labels=False, expects df columns: filename (or id/filename) and uses TEST_DIR.
    """
    n_rows = len(df)
    if limit != -1:
        n_rows = min(n_rows, int(limit))

    has_filename_col = "filename" in df.columns

    Xb = np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    yb = np.empty((batch_size,), dtype=np.float32) if has_labels else None
    paths = [None] * batch_size

    b = 0
    i = 0

    for row in df.itertuples(index=False):
        if i >= n_rows:
            break

        if has_labels:
            yb[b] = float(getattr(row, "class"))
            raw_image_path = os.path.join(TRAIN_DIR, getattr(row, "filename"))
        else:
            if has_filename_col:
                fn = getattr(row, "filename")
                raw_image_path = os.path.join(TEST_DIR, fn)
            else:
                raw_image_path = os.path.join(
                    TEST_DIR, f"{int(getattr(row, 'id'))}.jpg"
                )

        img = cv2.imread(raw_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {raw_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        )

        Xb[b] = img.astype(np.float32) / 255.0
        paths[b] = raw_image_path

        b += 1
        i += 1

        if b == batch_size:
            if produce == "images":
                if has_labels:
                    yield Xb, yb
                else:
                    yield Xb
            else:
                if has_labels:
                    yield paths, yb
                else:
                    yield paths
            b = 0

    if b > 0:
        if produce == "images":
            if has_labels:
                yield Xb[:b], yb[:b]
            else:
                yield Xb[:b]
        else:
            if has_labels:
                yield paths[:b], yb[:b]
            else:
                yield paths[:b]




## === cell 12
RUN_EVAL_DEBUG = False

if RUN_EVAL_DEBUG:
    samples = 64
    eval_steps = 1
    eval_result = transfer_model_vgg16.evaluate(
        build_batches(test, limit=samples, batch_size=BATCH_SIZE),
        steps=eval_steps,
        verbose=1,
    )
    print("Eval:", eval_result)



## === cell 13
RUN_PRED_DEBUG = False

if RUN_PRED_DEBUG:
    some_predictions = transfer_model_vgg16.predict(
        build_batches(test, limit=12, batch_size=1),
        steps=12,
        verbose=1,
    )



## === cell 14
if RUN_PRED_DEBUG:
    idx = 0
    for mini_batch_files in build_batches(
        test, limit=12, batch_size=1, produce="paths", has_labels=False
    ):
        mini_batch_file = mini_batch_files[0]
        predicted_label = float(some_predictions[idx][0])
        idx += 1
        plot_prediction(mini_batch_file, predicted_label)



## === cell 15
test_files = [f for f in listdir(TEST_DIR) if f.lower().endswith(".jpg")]
output = pd.DataFrame({"filename": test_files})
output["id"] = output["filename"].str.replace(".jpg", "", regex=False).astype(int)
output = output.sort_values("id").reset_index(drop=True)

print("Num test images:", len(output))
print(output.head())



## === cell 16
pred_batch_size = 64

test_paths = (TEST_DIR + "/" + output["filename"].astype(str)).to_numpy()
ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.map(
    _decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
)
ds_test = ds_test.batch(pred_batch_size, drop_remainder=False).prefetch(AUTOTUNE)

results = transfer_model_vgg16.predict(ds_test, verbose=1)
results = results.reshape(-1)[: len(output)]
print("Pred shape:", results.shape)



## === cell 17
output["label"] = results.astype(float)
print(output[["id", "label"]].head(10))

if PLOT_DEBUG:
    stop = 5
    for _, row in output.head(stop).iterrows():
        path = os.path.join(TEST_DIR, f"{int(row['id'])}.jpg")
        plot_prediction(path, row["label"])

submission = output[["id", "label"]].copy()
submission["id"] = submission["id"].astype(int)
submission["label"] = submission["label"].astype(float)

submission.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", submission.shape)
print(submission.head())
