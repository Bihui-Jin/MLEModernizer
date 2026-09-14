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

3.8

# 3. Installed packages

geopandas==0.14.4
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

0.16863

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import zipfile
import matplotlib.pyplot as plt
import glob

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("OpenCV:", cv2.__version__)
print(
    "protobuf implementation override:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

TEST_ZIP = os.path.join(BASE_INPUT, "test.zip")
TRAIN_ZIP = os.path.join(BASE_INPUT, "train.zip")

assert os.path.exists(TEST_ZIP), f"Missing {TEST_ZIP}"
assert os.path.exists(TRAIN_ZIP), f"Missing {TRAIN_ZIP}"

print("Found zips:", TRAIN_ZIP, TEST_ZIP)




## === cell 2
def _has_images(root):
    if not os.path.isdir(root):
        return False
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
    return False


need_extract = (not _has_images("train")) or (not _has_images("test"))
if need_extract:
    with zipfile.ZipFile(TRAIN_ZIP, "r") as zf:
        zf.extractall(".")
    with zipfile.ZipFile(TEST_ZIP, "r") as zf:
        zf.extractall(".")

print("Has train images:", _has_images("train"))
print("Has test images:", _has_images("test"))
print("Top-level dirs:", sorted([d for d in os.listdir(".") if os.path.isdir(d)]))




## === cell 3
def list_images_glob(folder):
    if not os.path.isdir(folder):
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    return files


train_search_roots = [
    "train",
    os.path.join("train", "train"),
    os.path.join("train", "cat"),
    os.path.join("train", "dog"),
    os.path.join("train", "train", "cat"),
    os.path.join("train", "train", "dog"),
]

seen = set()
train_images_all = []
for r in train_search_roots:
    for p in list_images_glob(r):
        if p not in seen:
            train_images_all.append(p)
            seen.add(p)

if len(train_images_all) == 0:
    for p in list_images_glob("."):
        base = os.path.basename(p).lower()
        if base.startswith(("cat.", "dog.")) and base.endswith(".jpg"):
            train_images_all.append(p)
    train_images_all = list(set(train_images_all))

test_search_roots = [
    "test",
    os.path.join("test", "test"),
    os.path.join("test", "unknown"),
    os.path.join("test", "test", "unknown"),
]
seen = set()
test_images = []
for r in test_search_roots:
    for p in list_images_glob(r):
        if p not in seen:
            test_images.append(p)
            seen.add(p)

test_images = [
    p for p in test_images if os.path.splitext(os.path.basename(p))[0].isdigit()
]

train_images_all = sorted(train_images_all)
test_images = sorted(
    test_images, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)

print("Num train images:", len(train_images_all))
print("Num test images:", len(test_images))
print("Example train:", train_images_all[0] if train_images_all else None)
print("Example test:", test_images[0] if test_images else None)

assert len(train_images_all) > 0, "No training images found after extraction."
assert len(test_images) > 0, "No test images found after extraction."



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1088419763.py in <cell line: 0>()
     64 
     65 assert len(train_images_all) > 0, "No training images found after extraction."
---> 66 assert len(test_images) > 0, "No test images found after extraction."
     67 

AssertionError: No test images found after extraction.

## === cell 4
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_images_all))
train_images_all = [train_images_all[i] for i in perm]

limit = int(0.8 * len(train_images_all))
train_images = train_images_all[:limit]
validation_images = train_images_all[limit:]

print(
    "Num train:",
    len(train_images),
    "Num val:",
    len(validation_images),
    "Num test:",
    len(test_images),
)

if train_images:
    img = cv2.imread(train_images[0])
    if img is None:
        raise ValueError(f"Failed to read image: {train_images[0]}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(3, 3))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()



## === cell 5
rows, columns = 160, 160
image_shape = (rows, columns, 3)


def is_dog(path):
    p = path.replace("\\", "/").lower()
    return ("/dog" in p and "dog." in os.path.basename(p).lower()) or os.path.basename(
        p
    ).lower().startswith("dog.")


label = np.array([1.0 if is_dog(p) else 0.0 for p in train_images], dtype=np.float32)
validation_label = np.array(
    [1.0 if is_dog(p) else 0.0 for p in validation_images], dtype=np.float32
)

print(
    "Label mean (train):",
    float(label.mean()),
    "Label mean (val):",
    float(validation_label.mean()),
)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # dataset is jpg
    img = tf.image.resize(img, [rows, columns], method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img


def make_supervised_ds(paths, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(y)))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, t: (_decode_resize_preprocess(p), t),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


BATCH_SIZE = 128
train_ds = make_supervised_ds(train_images, label, batch_size=BATCH_SIZE, training=True)
val_ds = make_supervised_ds(
    validation_images, validation_label, batch_size=BATCH_SIZE, training=False
)
test_ds = make_test_ds(test_images, batch_size=BATCH_SIZE)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3984728508.py in <cell line: 0>()
     69     validation_images, validation_label, batch_size=BATCH_SIZE, training=False
     70 )
---> 71 test_ds = make_test_ds(test_images, batch_size=BATCH_SIZE)
     72 

/tmp/ipykernel_11/3984728508.py in make_test_ds(paths, batch_size)
     55 def make_test_ds(paths, batch_size):
     56     ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
---> 57     ds = ds.map(
     58         _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
     59     )

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

/tmp/__autograph_generated_filez7z23tzs.py in tf___decode_resize_preprocess(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).io.decode_jpeg, (ag__.ld(img_bytes),), dict(channels=3), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(rows), ag__.ld(columns)]), dict(method=ag__.ld(tf).image.ResizeMethod.BICUBIC), fscope)

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
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    576                   (input_name, op_type_name, observed))
    577         if input_arg.type != types_pb2.DT_INVALID:
--> 578           raise TypeError(f"{prefix} expected type of "
    579                           f"{dtypes.as_dtype(input_arg.type).name}.")
    580         else:

TypeError: in user code:

    File "/tmp/ipykernel_11/3984728508.py", line 30, in _decode_resize_preprocess  *
        img_bytes = tf.io.read_file(path)

    TypeError: Input 'filename' of 'ReadFile' Op has type float32 that does not match expected type of string.


## === cell 6
base_model = tf.keras.applications.ResNet101(
    weights="imagenet", include_top=False, input_shape=image_shape
)
base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.summary()



## === cell 7
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    shuffle=True,  # preserved (ds already shuffles; this keeps API semantics consistent)
    verbose=0,
)

last_val_loss = history.history.get("val_loss", [None])[-1]
print(
    "Training done. Last val_loss:",
    None if last_val_loss is None else float(last_val_loss),
)



## === cell 8
prediction = model.predict(test_ds, verbose=0).reshape(-1)
prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

assert prediction.shape[0] > 0, "prediction has 0 rows; no test images were loaded."

idx = 4 if len(test_images) > 4 else 0
img0 = cv2.imread(test_images[idx])
if img0 is not None:
    img0 = cv2.resize(img0, (columns, rows), interpolation=cv2.INTER_CUBIC)
    plt.figure(figsize=(3, 3))
    plt.title(f"pred(dog)={prediction[idx]:.4f}")
    plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print(
    "Pred shape:",
    prediction.shape,
    "min/max:",
    float(prediction.min()),
    float(prediction.max()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2315276736.py in <cell line: 0>()
      1 # Speedup: predict directly from tf.data (avoids materializing full test_x array; same forward pass).
----> 2 prediction = model.predict(test_ds, verbose=0).reshape(-1)
      3 prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
      4 
      5 assert prediction.shape[0] > 0, "prediction has 0 rows; no test images were loaded."

NameError: name 'test_ds' is not defined

## === cell 9
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
assert list(sample.columns) == [
    "id",
    "label",
], f"Unexpected sample_submission columns: {sample.columns.tolist()}"

test_ids_from_files = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in test_images
]
pred_map = dict(zip(test_ids_from_files, prediction))

out = sample.copy()
out["label"] = out["id"].map(pred_map).astype(float)
out["label"] = out["label"].fillna(0.5).clip(1e-7, 1 - 1e-7)

out.to_csv("submission.csv", index=False, header=True)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("id min/max:", int(out["id"].min()), int(out["id"].max()))
print("label min/max:", float(out["label"].min()), float(out["label"].max()))

assert (
    out.shape[0] == sample.shape[0]
), "Submission row count does not match sample_submission."
assert out["label"].notna().all(), "Submission contains NaN labels."

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793160196.py in <cell line: 0>()
      9     int(os.path.splitext(os.path.basename(p))[0]) for p in test_images
     10 ]
---> 11 pred_map = dict(zip(test_ids_from_files, prediction))
     12 
     13 out = sample.copy()

NameError: name 'prediction' is not defined
