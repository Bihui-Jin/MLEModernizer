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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1894736842105263

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print(
        "Determinism toggle unavailable in this environment; continuing. Error:",
        repr(e),
    )

try:
    tf.config.optimizer.set_jit(False)  # keep behavior stable; not a speed focus here
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode = False  # keep runtime lean (as intended)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options({"tf_io_file_read_cache": True})
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, sub.shape)
train.head()



## === cell 2
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew", "healthy"]
NUM_CLASSES = len(CLASSES)

_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

train["filepath"] = train_images_dir + "/" + train["image"].astype(str)

labels_split = train["labels"].astype(str).str.split(" ")
exploded = labels_split.explode()
ct = pd.crosstab(exploded.index, exploded).reindex(columns=CLASSES, fill_value=0)
targets = (ct.to_numpy() > 0).astype(np.float32)
train["target"] = list(targets)

print("Prepared train filepaths/targets.")



## === cell 3
AUTO = tf.data.AUTOTUNE
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep unchanged


def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.AREA, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label




## === cell 4
idx = np.arange(len(train))
np.random.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

x_trn = trn_df["filepath"].values
y_trn = np.stack(trn_df["target"].values)

x_val = val_df["filepath"].values
y_val = np.stack(val_df["target"].values)

print("Train:", x_trn.shape, y_trn.shape, "Val:", x_val.shape, y_val.shape)

cache_dir = "/kaggle/working/tf_cache_pp2021"
os.makedirs(cache_dir, exist_ok=True)


def _map_train(f, y):
    return decode_image(f, y)


def _map_infer(f):
    return decode_image(f, None)


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.threading.private_threadpool_size = 0  # let TF tune
options.threading.max_intra_op_parallelism = 0

train_base = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).with_options(options)
val_base = tf.data.Dataset.from_tensor_slices((x_val, y_val)).with_options(options)

train_ds = (
    train_base.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)
val_ds = (
    val_base.batch(BATCH_SIZE, drop_remainder=False)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/366851497.py in <cell line: 0>()
     51     train_base.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
     52     .batch(BATCH_SIZE, drop_remainder=False)
---> 53     .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
     54     .prefetch(AUTO)
     55 )

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

/tmp/__autograph_generated_file2tehfpqz.py in tf___map_train(f, y)
     10                 try:
     11                     do_return = True
---> 12                     retval_ = ag__.converted_call(ag__.ld(decode_image), (ag__.ld(f), ag__.ld(y)), None, fscope)
     13                 except:
     14                     do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filedsufnn3s.py in tf__decode_image(filename, label, image_size)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
     11                 image = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(bits),), dict(channels=3, dct_method='INTEGER_FAST'), fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), ag__.ld(image_size)), dict(method=ag__.ld(tf).image.ResizeMethod.AREA, antialias=False), fscope)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    586       pass  # Add nodes to the TensorFlow graph.
    587   # Add nodes to the TensorFlow graph.
--> 588   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    589         "ReadFile", filename=filename, name=name)
    590   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    794       # Add Op to graph
    795       # pylint: disable=protected-access
--> 796       op = g._create_op_internal(op_type_name, inputs, dtypes=None,
    797                                  name=scope, input_types=input_types,
    798                                  attrs=attr_protos, op_def=op_def)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
    668       inp = self.capture(inp)
    669       captured_inputs.append(inp)
--> 670     return super()._create_op_internal(  # pylint: disable=protected-access
    671         op_type, captured_inputs, dtypes, input_types, name, attrs, op_def,
    672         compute_device)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
   2699     # Session.run call cannot occur between creating and mutating the op.
   2700     with self._mutation_lock():
-> 2701       ret = Operation.from_node_def(
   2702           node_def,
   2703           self,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in from_node_def(***failed resolving arguments***)
   1194 
   1195     # Initialize c_op from node_def and other inputs
-> 1196     c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
   1197     self = Operation(c_op, SymbolicTensor)
   1198     self._init(g)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_11/366851497.py", line 25, in _map_train  *
        return decode_image(f, y)
    File "/tmp/ipykernel_11/1745035665.py", line 7, in decode_image  *
        bits = tf.io.read_file(filename)

    ValueError: Shape must be rank 0 but is rank 1 for '{{node ReadFile}} = ReadFile[](args_0)' with input shapes: [?].


## === cell 5
from tensorflow import keras

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

base.trainable = False  # keep training light and stable

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = keras.applications.resnet50.preprocess_input(x * 255.0)
x = base(x, training=False)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()



## === cell 6
feat_inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="feat_image")
feat_x = keras.applications.resnet50.preprocess_input(feat_inputs * 255.0)
feat_outputs = base(feat_x, training=False)  # 2048-d pooled features
feat_extractor = keras.Model(feat_inputs, feat_outputs, name="feat_extractor")

feat_cache_dir = os.path.join(cache_dir, "feat_cache")
os.makedirs(feat_cache_dir, exist_ok=True)
trn_feat_cache_path = os.path.join(feat_cache_dir, "trn_features.cache.npy")
val_feat_cache_path = os.path.join(feat_cache_dir, "val_features.cache.npy")


def _maybe_load_features(npy_path):
    if os.path.exists(npy_path):
        arr = np.load(npy_path, allow_pickle=False, mmap_mode=None)
        return arr
    return None


trn_features = _maybe_load_features(trn_feat_cache_path)
val_features = _maybe_load_features(val_feat_cache_path)

if trn_features is None or val_features is None:
    train_img_only = train_ds.map(
        lambda img, y: img, num_parallel_calls=AUTO, deterministic=True
    )
    val_img_only = val_ds.map(
        lambda img, y: img, num_parallel_calls=AUTO, deterministic=True
    )

    trn_steps = int(np.ceil(len(x_trn) / BATCH_SIZE))
    val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))

    trn_features = feat_extractor.predict(train_img_only, steps=trn_steps, verbose=1)
    val_features = feat_extractor.predict(val_img_only, steps=val_steps, verbose=1)

    np.save(trn_feat_cache_path, trn_features)
    np.save(val_feat_cache_path, val_features)

head_inp = keras.Input(
    shape=(2048,), name="features"
)  # ResNet50(pooling="avg") -> 2048
head_out = model.layers[-1](
    head_inp
)  # reuse the same Dense layer instance (shared weights)
head_model = keras.Model(head_inp, head_out, name="head_model")

head_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

train_feat_ds = (
    tf.data.Dataset.from_tensor_slices((trn_features, y_trn))
    .with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
val_feat_ds = (
    tf.data.Dataset.from_tensor_slices((val_features, y_val))
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

EPOCHS = 3
history = head_model.fit(
    train_feat_ds,
    validation_data=val_feat_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2427710819.py in <cell line: 0>()
     27     # Feed the already-decoded/normalized batches from train_ds/val_ds into the feature extractor.
     28     # Also set explicit steps to reduce iterator bookkeeping overhead; this does not change results.
---> 29     train_img_only = train_ds.map(
     30         lambda img, y: img, num_parallel_calls=AUTO, deterministic=True
     31     )

NameError: name 'train_ds' is not defined

## === cell 7
sub["filepath"] = test_images_dir + "/" + sub["image"].astype(str)
print("Prepared test filepaths.")

options_test = tf.data.Options()
options_test.experimental_deterministic = True
options_test.experimental_optimization.apply_default_optimizations = True
options_test.experimental_optimization.map_parallelization = True
options_test.experimental_optimization.parallel_batch = True
options_test.threading.private_threadpool_size = 0
options_test.threading.max_intra_op_parallelism = 0

test_ds = (
    tf.data.Dataset.from_tensor_slices(sub["filepath"].values)
    .with_options(options_test)
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(lambda f: _map_infer(f), num_parallel_calls=AUTO, deterministic=True)
    .prefetch(AUTO)
)

test_feat_cache_path = os.path.join(feat_cache_dir, "test_features.cache.npy")

test_features = _maybe_load_features(test_feat_cache_path)
if test_features is None:
    test_steps = int(np.ceil(len(sub) / BATCH_SIZE))
    test_features = feat_extractor.predict(test_ds, steps=test_steps, verbose=1)
    np.save(test_feat_cache_path, test_features)

probs = head_model.predict(test_features, verbose=1)
print("probs shape:", probs.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/112385940.py in <cell line: 0>()
     14     .with_options(options_test)
     15     .batch(BATCH_SIZE, drop_remainder=False)
---> 16     .map(lambda f: _map_infer(f), num_parallel_calls=AUTO, deterministic=True)
     17     .prefetch(AUTO)
     18 )

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

/tmp/__autograph_generated_file5m9fyinj.py in <lambda>(f)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f: ag__.with_function_scope(lambda lscope: ag__.converted_call(_map_infer, (f,), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file5m9fyinj.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f: ag__.with_function_scope(lambda lscope: ag__.converted_call(_map_infer, (f,), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filecy5kg0xd.py in tf___map_infer(f)
     10                 try:
     11                     do_return = True
---> 12                     retval_ = ag__.converted_call(ag__.ld(decode_image), (ag__.ld(f), None), None, fscope)
     13                 except:
     14                     do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filedsufnn3s.py in tf__decode_image(filename, label, image_size)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
     11                 image = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(bits),), dict(channels=3, dct_method='INTEGER_FAST'), fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), ag__.ld(image_size)), dict(method=ag__.ld(tf).image.ResizeMethod.AREA, antialias=False), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    586       pass  # Add nodes to the TensorFlow graph.
    587   # Add nodes to the TensorFlow graph.
--> 588   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    589         "ReadFile", filename=filename, name=name)
    590   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    794       # Add Op to graph
    795       # pylint: disable=protected-access
--> 796       op = g._create_op_internal(op_type_name, inputs, dtypes=None,
    797                                  name=scope, input_types=input_types,
    798                                  attrs=attr_protos, op_def=op_def)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
    668       inp = self.capture(inp)
    669       captured_inputs.append(inp)
--> 670     return super()._create_op_internal(  # pylint: disable=protected-access
    671         op_type, captured_inputs, dtypes, input_types, name, attrs, op_def,
    672         compute_device)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
   2699     # Session.run call cannot occur between creating and mutating the op.
   2700     with self._mutation_lock():
-> 2701       ret = Operation.from_node_def(
   2702           node_def,
   2703           self,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in from_node_def(***failed resolving arguments***)
   1194 
   1195     # Initialize c_op from node_def and other inputs
-> 1196     c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
   1197     self = Operation(c_op, SymbolicTensor)
   1198     self._init(g)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_11/112385940.py", line 16, in None  *
        lambda f: _map_infer(f)
    File "/tmp/ipykernel_11/366851497.py", line 29, in _map_infer  *
        return decode_image(f, None)
    File "/tmp/ipykernel_11/1745035665.py", line 7, in decode_image  *
        bits = tf.io.read_file(filename)

    ValueError: Shape must be rank 0 but is rank 1 for '{{node ReadFile}} = ReadFile[](args_0)' with input shapes: [?].


## === cell 8
thresholds = {c: 0.5 for c in CLASSES}  # keep unchanged

thr = np.array([thresholds[c] for c in CLASSES], dtype=probs.dtype)
healthy_idx = CLASSES.index("healthy")
nonhealthy_mask = np.ones(NUM_CLASSES, dtype=bool)
nonhealthy_mask[healthy_idx] = False

sel = (probs > thr) & nonhealthy_mask  # (N, C) boolean; healthy is never selected here

idxs = [np.flatnonzero(r).tolist() for r in sel]
pred_string = [
    "healthy" if len(js) == 0 else " ".join(CLASSES[j] for j in js) for js in idxs
]

submission = sub[["image"]].copy()
submission["labels"] = pred_string

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2299905029.py in <cell line: 0>()
      1 thresholds = {c: 0.5 for c in CLASSES}  # keep unchanged
      2 
----> 3 thr = np.array([thresholds[c] for c in CLASSES], dtype=probs.dtype)
      4 healthy_idx = CLASSES.index("healthy")
      5 nonhealthy_mask = np.ones(NUM_CLASSES, dtype=bool)

NameError: name 'probs' is not defined
