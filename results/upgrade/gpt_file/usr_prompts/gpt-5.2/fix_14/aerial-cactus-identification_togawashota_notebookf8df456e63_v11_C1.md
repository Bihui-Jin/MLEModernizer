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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9666001666666668

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Main runtime waste is repeated JPEG decoding/augmentation and expensive `tf.data` caching-to-disk that writes/reads tens of thousands of small records; the training itself is relatively small (30 steps/epoch). I keep the exact model, epochs/steps, losses, and deterministic augmentation logic, but eliminate Python-side per-sample augmentation planning and move it into a pure-TF path-hash (same CRC32) computed inside the `Dataset` graph. I also switch dataset caching from on-disk `.tfdata` files to in-memory caching (still exact data, just avoids heavy filesystem I/O) and fuse decode+augment into a single map to reduce pipeline overhead, while keeping determinism and seeds intact. Test decoding stays cached but in memory as well, which is safe given the dataset size.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

tf = None
TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        tf.random.set_seed(SEED)
    except Exception as e:
        print("WARNING: tf.random.set_seed failed:", repr(e))
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

except Exception as e:
    print(
        "WARNING: TensorFlow import failed; will write a fallback submission (score will be low)."
    )
    print("Import error:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import zipfile

extract_dir = "/kaggle/working"

train_marker = os.path.join(extract_dir, "__train_extracted__")
test_marker = os.path.join(extract_dir, "__test_extracted__")

if not os.path.exists(train_marker):
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(extract_dir)
    with open(train_marker, "w") as f:
        f.write("ok")

if not os.path.exists(test_marker):
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(extract_dir)
    with open(test_marker, "w") as f:
        f.write("ok")




## === cell 2
def find_image_dir_fast(root, target_name):
    candidates = [
        os.path.join(root, target_name),
        os.path.join(root, "aerial-cactus-identification", target_name),
        os.path.join(
            root,
            "aerial-cactus-identification",
            "aerial-cactus-identification",
            target_name,
        ),
    ]
    for p in candidates:
        if os.path.isdir(p):
            try:
                with os.scandir(p) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(".jpg"):
                            return p
            except FileNotFoundError:
                pass

    level1 = []
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir():
                    level1.append(e.path)
    except FileNotFoundError:
        pass

    for base in level1:
        p = os.path.join(base, target_name)
        if os.path.isdir(p):
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        return p

        try:
            with os.scandir(base) as it:
                for e in it:
                    if e.is_dir():
                        p2 = os.path.join(e.path, target_name)
                        if os.path.isdir(p2):
                            with os.scandir(p2) as it2:
                                for f in it2:
                                    if f.is_file() and f.name.lower().endswith(".jpg"):
                                        return p2
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        f"Could not find extracted '{target_name}' directory with jpgs under: {root}"
    )


train_dir = find_image_dir_fast("/kaggle/working", "train")
test_dir = find_image_dir_fast("/kaggle/working", "test")

print("Detected train_dir:", train_dir)
print("Detected test_dir:", test_dir)

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head(5))




## === cell 3
def count_files(directory):
    with os.scandir(directory) as it:
        return sum(1 for e in it if e.is_file())


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## === cell 4
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 5
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 6
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 7
if TF_AVAILABLE:
    import math

    AUTOTUNE = tf.data.AUTOTUNE

    _ds_options = tf.data.Options()
    _ds_options.experimental_deterministic = True
    try:
        _ds_options.experimental_optimization.map_and_batch_fusion = True
        _ds_options.experimental_optimization.parallel_batch = True
        _ds_options.experimental_optimization.map_parallelization = True
    except Exception:
        pass

    def _list_jpg_files(directory):
        files = []
        with os.scandir(directory) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".jpg"):
                    files.append(e.name)
        files.sort()
        return files

    id_to_label = dict(zip(train_df["id"].tolist(), train_df["has_cactus"].tolist()))

    all_train_files = _list_jpg_files(train_dir)
    y_all = [id_to_label[f] for f in all_train_files]

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(all_train_files))
    rng.shuffle(idx)
    split = int(math.floor(len(idx) * (1.0 - 0.10)))
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_files = [all_train_files[i] for i in train_idx]
    val_files = [all_train_files[i] for i in val_idx]

    def _label_str_to_float(y_str):
        return 1.0 if y_str == "1" else 0.0

    train_y = np.asarray(
        [_label_str_to_float(y_all[i]) for i in train_idx], dtype=np.float32
    )
    val_y = np.asarray(
        [_label_str_to_float(y_all[i]) for i in val_idx], dtype=np.float32
    )

    train_paths = np.asarray(
        [os.path.join(train_dir, f) for f in train_files], dtype=np.str_
    )
    val_paths = np.asarray(
        [os.path.join(train_dir, f) for f in val_files], dtype=np.str_
    )

    @tf.function
    def _decode_jpg_uint8(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.ensure_shape(img, (32, 32, 3))
        return img  # uint8

    @tf.function
    def _augment_plan_from_path(path_str):
        p_bytes = tf.io.decode_raw(path_str, tf.uint8)
        h = tf.raw_ops.CRC32(input=p_bytes)  # uint32
        h = tf.bitwise.bitwise_xor(h, tf.constant(SEED & 0xFFFFFFFF, dtype=tf.uint32))

        k = tf.cast(tf.bitwise.bitwise_and(h, 3), tf.int32)  # 0..3
        do_lr = tf.cast(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(h, 2), 1), tf.int32
        )
        do_ud = tf.cast(
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(h, 3), 1), tf.int32
        )
        u16 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(h, 8), 0xFFFF)
        u16_f = tf.cast(u16, tf.float32)
        factor = tf.constant(0.8, tf.float32) + (u16_f / 65536.0) * tf.constant(
            0.4, tf.float32
        )
        return k, do_lr, do_ud, factor

    @tf.function
    def _augment_from_plan(img_uint8, k, do_lr, do_ud, factor):
        img = tf.image.rot90(img_uint8, k=k)
        img = tf.cond(do_lr > 0, lambda: tf.image.flip_left_right(img), lambda: img)
        img = tf.cond(do_ud > 0, lambda: tf.image.flip_up_down(img), lambda: img)
        img_f = tf.cast(img, tf.float32)
        img_f = img_f * factor * (1.0 / 255.0)
        img_f = tf.clip_by_value(img_f, 0.0, 1.0)
        return img_f

    @tf.function
    def _preprocess_train(path, label):
        img_uint8 = _decode_jpg_uint8(path)
        k, do_lr, do_ud, factor = _augment_plan_from_path(path)
        img = _augment_from_plan(img_uint8, k, do_lr, do_ud, factor)
        return img, label

    @tf.function
    def _preprocess_val(path, label):
        img_uint8 = _decode_jpg_uint8(path)
        img = tf.cast(img_uint8, tf.float32) * (1.0 / 255.0)
        return img, label

    train_base = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    val_base = tf.data.Dataset.from_tensor_slices((val_paths, val_y))

    train_base = train_base.with_options(_ds_options).cache()
    val_base = val_base.with_options(_ds_options).cache()

    train_ds = train_base.shuffle(
        buffer_size=min(int(len(train_paths)), 4096),
        seed=SEED,
        reshuffle_each_iteration=True,
    ).repeat()
    train_ds = train_ds.map(
        _preprocess_train, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    train_ds = train_ds.batch(512, drop_remainder=False).prefetch(AUTOTUNE)
    train_ds = train_ds.with_options(_ds_options)

    val_ds = val_base.map(
        _preprocess_val, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    val_ds = val_ds.batch(256, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.with_options(_ds_options)

    train_generator = train_ds
    val_generator = val_ds



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1260981622.py in <cell line: 0>()
    125         reshuffle_each_iteration=True,
    126     ).repeat()
--> 127     train_ds = train_ds.map(
    128         _preprocess_train, num_parallel_calls=AUTOTUNE, deterministic=True
    129     )

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

/tmp/__autograph_generated_file1j2c7y2o.py in tf___preprocess_train(path, label)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_uint8 = ag__.converted_call(ag__.ld(_decode_jpg_uint8), (ag__.ld(path),), None, fscope)
---> 11                 k, do_lr, do_ud, factor = ag__.converted_call(ag__.ld(_augment_plan_from_path), (ag__.ld(path),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(_augment_from_plan), (ag__.ld(img_uint8), ag__.ld(k), ag__.ld(do_lr), ag__.ld(do_ud), ag__.ld(factor)), None, fscope)
     13                 try:

/tmp/__autograph_generated_filetsgksbmz.py in tf___augment_plan_from_path(path_str)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 p_bytes = ag__.converted_call(ag__.ld(tf).io.decode_raw, (ag__.ld(path_str), ag__.ld(tf).uint8), None, fscope)
---> 11                 h = ag__.converted_call(ag__.ld(tf).raw_ops.CRC32, (), dict(input=ag__.ld(p_bytes)), fscope)
     12                 h = ag__.converted_call(ag__.ld(tf).bitwise.bitwise_xor, (ag__.ld(h), ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(SEED) & 4294967295,), dict(dtype=ag__.ld(tf).uint32), fscope)), None, fscope)
     13                 k = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).bitwise.bitwise_and, (ag__.ld(h), 3), None, fscope), ag__.ld(tf).int32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1260981622.py", line 105, in _preprocess_train  *
        k, do_lr, do_ud, factor = _augment_plan_from_path(path)
    File "/tmp/ipykernel_11/1260981622.py", line 75, in _augment_plan_from_path  *
        h = tf.raw_ops.CRC32(input=p_bytes)  # uint32

    AttributeError: module 'tensorflow._api.v2.raw_ops' has no attribute 'CRC32'


## === cell 8
if TF_AVAILABLE:
    AUTOTUNE = tf.data.AUTOTUNE

    def _list_test_files_sorted(test_dir_):
        files = []
        with os.scandir(test_dir_) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".jpg"):
                    files.append(e.name)
        files.sort()
        return files

    test_files = _list_test_files_sorted(test_dir)
    test_paths = np.asarray(
        [os.path.join(test_dir, f) for f in test_files], dtype=np.str_
    )

    @tf.function
    def _decode_and_rescale(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.ensure_shape(img, (32, 32, 3))
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(
        _decode_and_rescale, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(512, drop_remainder=False).prefetch(AUTOTUNE)

    test_generator = test_ds
    test_generator_filenames = [os.path.join("test", f) for f in test_files]



## === cell 9
if TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    efficient_net = EfficientNetB3(
        weights="imagenet",
        input_shape=(32, 32, 3),
        include_top=False,
        pooling="max",
    )
    efficient_net.trainable = False

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()



## === cell 10
if TF_AVAILABLE:
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )



## === cell 11
if TF_AVAILABLE:
    history = model.fit(
        train_generator,
        epochs=70,
        steps_per_epoch=30,
        validation_data=val_generator,
        validation_steps=7,
        class_weight=class_weights_dict,
        verbose=2,
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1752440137.py in <cell line: 0>()
      1 if TF_AVAILABLE:
      2     history = model.fit(
----> 3         train_generator,
      4         epochs=70,
      5         steps_per_epoch=30,

NameError: name 'train_generator' is not defined

## === cell 12
if TF_AVAILABLE:
    preds = model.predict(test_generator, verbose=1)
    preds = preds.reshape(-1).astype(float)
else:
    preds = None



## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

if TF_AVAILABLE:
    test_files_base = [os.path.basename(fn) for fn in test_generator_filenames]
    pred_df = pd.DataFrame({"id": test_files_base, "has_cactus": preds})

    submission = sample_sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))[
        ["id", "has_cactus"]
    ]

    missing = int(submission["has_cactus"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} test ids in sample_submission were not found in generated predictions mapping."
        )
else:
    submission = sample_sub.copy()
    submission["has_cactus"] = 0.5

print(submission.head(10))
print("Submission shape:", submission.shape)



## === cell 14
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print(os.listdir("/kaggle/working"))
print("Wrote:", out_path)
