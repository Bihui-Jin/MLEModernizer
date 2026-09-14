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

0.8079039704524489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.08684) has done: 'I remove the failing `tensorflow_addons` import that triggers the protobuf `MessageFactory` error, since it isn’t used anywhere in the pipeline. Then I fix model loading by providing a safe fallback model (same overall inference flow) when the external `.h5` file path doesn’t exist in your environment, so `preds` is always defined. Finally, I correct the submission label assignment logic (a `==` bug, chained assignment, and an always-true condition) and ensure the output CSV has exactly the required `image,labels` columns and a `.csv` suffix.'
- What this solution (achieved 0.30579) has done: 'The timeout is dominated by slow Python-side image loading/augmentation via `ImageDataGenerator.flow_from_dataframe` plus extra overhead from Keras predict/fit not using optimal threading/prefetching. To keep the exact same model/training logic and semantics, I switch the data pipeline to a cached, parallel `tf.data` pipeline that applies the same preprocessing and the same augmentations (for the training generator) while preserving determinism via seeds. I also ensure `steps_per_epoch`/`validation_steps` are explicit to avoid any generator bookkeeping overhead and tune `tf.data` prefetch/parallelism and TF threading for faster CPU input throughput. The threshold search and submission formatting be vectorized where safe (provably equivalent) to remove Python loops.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv(
    "../input/plant-pathology-2021-fgvc8/train.csv",
    engine="c",
    usecols=["image", "labels"],
)
train.head()



## === cell 2
submissions = pd.read_csv(
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
    engine="c",
    usecols=["image", "labels"],
)
submissions.head()



## === cell 3
h_target = 384
w_target = 384
batch_size = 32



## === cell 4
effnet_preprocess = tf.keras.applications.efficientnet.preprocess_input


@tf.function
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)

    shape = tf.image.extract_jpeg_shape(img_bytes)  # [H,W,3]
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    side = tf.minimum(h, w)
    offset_y = tf.cast(tf.round((h - side) * 0.5), tf.int32)
    offset_x = tf.cast(tf.round((w - side) * 0.5), tf.int32)
    crop_h = tf.cast(side, tf.int32)
    crop_w = tf.cast(side, tf.int32)
    crop_window = tf.stack([offset_y, offset_x, crop_h, crop_w])

    side_i = tf.cast(side, tf.int32)
    ratio = tf.constant(1, tf.int32)
    ratio = tf.where(side_i >= 8 * h_target, 8, ratio)
    ratio = tf.where((side_i >= 4 * h_target) & (ratio == 1), 4, ratio)
    ratio = tf.where((side_i >= 2 * h_target) & (ratio == 1), 2, ratio)

    img = tf.image.decode_and_crop_jpeg(
        img_bytes, crop_window=crop_window, channels=3, ratio=ratio
    )  # uint8

    img = tf.image.resize(
        img, (h_target, w_target), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = effnet_preprocess(img)
    return img


def make_test_ds(df, directory, batch_size):
    paths_np = (directory.rstrip("/") + "/" + df["image"].to_numpy()).astype("U")
    paths = tf.constant(paths_np)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.threading.private_threadpool_size = 0
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(
    submissions,
    directory="../input/plant-pathology-2021-fgvc8/test_images",
    batch_size=batch_size,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3251269066.py in <cell line: 0>()
     68 
     69 
---> 70 test_ds = make_test_ds(
     71     submissions,
     72     directory="../input/plant-pathology-2021-fgvc8/test_images",

/tmp/ipykernel_11/3251269066.py in make_test_ds(df, directory, batch_size)
     61     ds = ds.with_options(options)
     62 
---> 63     ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
     64     ds = ds.cache()
     65     ds = ds.batch(batch_size, drop_remainder=False)

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

/tmp/__autograph_generated_file59g_gzrv.py in tf___read_decode_resize(path)
     23                 ratio = ag__.converted_call(ag__.ld(tf).where, ((ag__.ld(side_i) >= 4 * ag__.ld(h_target)) & (ag__.ld(ratio) == 1), 4, ag__.ld(ratio)), None, fscope)
     24                 ratio = ag__.converted_call(ag__.ld(tf).where, ((ag__.ld(side_i) >= 2 * ag__.ld(h_target)) & (ag__.ld(ratio) == 1), 2, ag__.ld(ratio)), None, fscope)
---> 25                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_crop_jpeg, (ag__.ld(img_bytes),), dict(crop_window=ag__.ld(crop_window), channels=3, ratio=ag__.ld(ratio)), fscope)
     26                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), (ag__.ld(h_target), ag__.ld(w_target))), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
     27                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/3251269066.py", line 34, in _read_decode_resize  *
        img = tf.image.decode_and_crop_jpeg(

    TypeError: Expected int for argument 'ratio' not <tf.Tensor 'SelectV2_2:0' shape=() dtype=int32>.


## === cell 5
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
labels_arr = mlb.transform(label_split).astype(np.int8, copy=False)
labels_df = pd.DataFrame(labels_arr, columns=class_names)

print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 6
candidate_model_paths = [
    "../input/resnet101-512-to-384/resnet101.h5",
    "/kaggle/input/resnet101-512-to-384/resnet101.h5",
    "../input/resnet101-512-to-384/resnet101.h5",
    "../input/resnet101-512-to-384/resnet101.h5",
    "../input/resnet101-512-to-384/resnet101.h5",
    "/kaggle/input/resnet101-512-to-384/resnet101.h5",
]
model_path = next(
    (p for p in candidate_model_paths if os.path.exists(p)), candidate_model_paths[0]
)

model = None
loaded_external = False
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    loaded_external = True
    print("Loaded model from:", model_path)
else:
    print("WARNING: Model file not found:", model_path)
    print(
        "Falling back to ImageNet-pretrained EfficientNetB0 with a new classification head."
    )
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(h_target, w_target, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(class_names), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)




## === cell 7
def multilabel_f1_micro(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return f1


best_thresh = None


@tf.function
def _augment_train(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    k = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    scale = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(scale * h_target), tf.int32)
    new_w = tf.cast(tf.round(scale * w_target), tf.int32)
    zoomed = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
    zoomed = tf.image.resize_with_crop_or_pad(zoomed, h_target, w_target)
    return zoomed


def make_train_val_ds(train_df, val_df, train_y, val_y, directory, batch_size):
    train_paths_np = (
        directory.rstrip("/") + "/" + train_df["image"].to_numpy()
    ).astype("U")
    val_paths_np = (directory.rstrip("/") + "/" + val_df["image"].to_numpy()).astype(
        "U"
    )

    train_paths = tf.constant(train_paths_np)
    val_paths = tf.constant(val_paths_np)

    train_y = tf.constant(train_y, dtype=tf.float32)
    val_y = tf.constant(val_y, dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.threading.private_threadpool_size = 0
    train_ds = train_ds.with_options(opts)
    val_ds = val_ds.with_options(opts)

    shuffle_buf = int(min(len(train_df), 4096))
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )

    @tf.function
    def _train_map(path, y):
        img = _read_decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack(
            [tf.cast(h % (2**31 - 1), tf.int32), tf.cast(SEED, tf.int32)], axis=0
        )
        img = _augment_train(img, seed)
        return img, y

    @tf.function
    def _val_map(path, y):
        img = _read_decode_resize(path)
        return img, y

    train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    val_ds = val_ds.map(_val_map, num_parallel_calls=AUTOTUNE).cache()
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return train_ds, val_ds


if not loaded_external:
    idx = np.arange(len(train))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    val_size = int(0.10 * len(idx))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
    val_df = train.iloc[val_idx].reset_index(drop=True).copy()

    train_y = labels_df.iloc[tr_idx].to_numpy(dtype=np.float32, copy=False)
    val_y = labels_df.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)

    train_ds, val_ds = make_train_val_ds(
        train_df,
        val_df,
        train_y,
        val_y,
        directory="../input/plant-pathology-2021-fgvc8/train_images",
        batch_size=batch_size,
    )

    steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
    validation_steps = int(np.ceil(len(val_df) / batch_size))

    for layer in model.layers:
        layer.trainable = True
    if hasattr(model.layers[0], "layers") or isinstance(
        model.layers[0], tf.keras.Model
    ):
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.Dense):
                lyr.trainable = True
            else:
                lyr.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=3,
        verbose=1,
    )

    for lyr in model.layers:
        lyr.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=1,
        verbose=1,
    )

    val_preds = model.predict(val_ds, verbose=1)
    val_true = val_y.astype(np.int8, copy=False)

    grid = np.arange(0.05, 0.60, 0.05, dtype=np.float32)
    yt = val_true.reshape(-1).astype(np.int8, copy=False)
    vp = val_preds.reshape(-1).astype(np.float32, copy=False)

    best_global_t = 0.25
    best_global_f1 = -1.0

    yt1 = yt == 1
    yt0 = ~yt1

    for t in grid:
        yhat1 = vp >= t
        tp = int(np.dot(yt1, yhat1))
        fp = int(np.dot(yt0, yhat1))
        fn = int(np.dot(yt1, ~yhat1))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp) / denom if denom > 0 else 0.0
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = float(t)

    best_thresh = np.full(len(class_names), best_global_t, dtype=np.float32)
    print(
        f"Global threshold calibrated on validation: t={best_global_t:.2f}, microF1={best_global_f1:.5f}"
    )
else:
    best_thresh = np.full(len(class_names), 0.25, dtype=np.float32)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3823465279.py in <cell line: 0>()
    108     val_y = labels_df.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)
    109 
--> 110     train_ds, val_ds = make_train_val_ds(
    111         train_df,
    112         val_df,

/tmp/ipykernel_11/3823465279.py in make_train_val_ds(train_df, val_df, train_y, val_y, directory, batch_size)
     85         return img, y
     86 
---> 87     train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
     88     train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
     89 

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

/tmp/__autograph_generated_filekhdep4p6.py in tf___train_map(path, y)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_read_decode_resize), (ag__.ld(path),), None, fscope)
     11                 h = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
     12                 seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h) % (2 ** 31 - 1), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)],), dict(axis=0), fscope)

/tmp/__autograph_generated_file59g_gzrv.py in tf___read_decode_resize(path)
     23                 ratio = ag__.converted_call(ag__.ld(tf).where, ((ag__.ld(side_i) >= 4 * ag__.ld(h_target)) & (ag__.ld(ratio) == 1), 4, ag__.ld(ratio)), None, fscope)
     24                 ratio = ag__.converted_call(ag__.ld(tf).where, ((ag__.ld(side_i) >= 2 * ag__.ld(h_target)) & (ag__.ld(ratio) == 1), 2, ag__.ld(ratio)), None, fscope)
---> 25                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_crop_jpeg, (ag__.ld(img_bytes),), dict(crop_window=ag__.ld(crop_window), channels=3, ratio=ag__.ld(ratio)), fscope)
     26                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), (ag__.ld(h_target), ag__.ld(w_target))), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
     27                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/3823465279.py", line 74, in _train_map  *
        img = _read_decode_resize(path)
    File "/tmp/ipykernel_11/3251269066.py", line 34, in _read_decode_resize  *
        img = tf.image.decode_and_crop_jpeg(

    TypeError: Expected int for argument 'ratio' not <tf.Tensor 'SelectV2_2:0' shape=() dtype=int32>.


## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4268054041.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 print("preds shape:", preds.shape)
      3 print(preds[:2])
      4 

NameError: name 'test_ds' is not defined

## === cell 9
assert preds.shape[0] == len(submissions), (
    f"Prediction count must match submission rows. preds={preds.shape[0]} "
    f"submission={len(submissions)}"
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1898441338.py in <cell line: 0>()
----> 1 assert preds.shape[0] == len(submissions), (
      2     f"Prediction count must match submission rows. preds={preds.shape[0]} "
      3     f"submission={len(submissions)}"
      4 )
      5 

NameError: name 'preds' is not defined

## === cell 10
default_thresh = 0.25

P = preds.astype(np.float32, copy=False)

thr = (
    np.full((len(class_names),), default_thresh, dtype=np.float32)
    if best_thresh is None
    else best_thresh.astype(np.float32, copy=False)
)

mask = P >= thr[None, :]
argmax_idx = np.argmax(P, axis=1)

rows_no_label = ~mask.any(axis=1)
if np.any(rows_no_label):
    mask[rows_no_label, argmax_idx[rows_no_label]] = True

cn = np.asarray(class_names, dtype=str)
label_strs = np.fromiter(
    (" ".join(cn[row]) for row in mask),
    dtype=object,
    count=mask.shape[0],
)

sub_out = submissions.copy()
sub_out["labels"] = label_strs

sub_out[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/231927105.py in <cell line: 0>()
      1 default_thresh = 0.25
      2 
----> 3 P = preds.astype(np.float32, copy=False)
      4 
      5 thr = (

NameError: name 'preds' is not defined

## === cell 11
sub_out.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3168842127.py in <cell line: 0>()
----> 1 sub_out.head()

NameError: name 'sub_out' is not defined
