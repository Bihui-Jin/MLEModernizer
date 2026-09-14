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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.66977

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.80168) has done: 'I fix the TensorFlow import crash by pinning protobuf’s Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also fix the callback bug by updating the learning-rate assignment to use `optimizer.learning_rate` (the `lr` alias no longer exists), keeping the callback’s behavior the same. To nudge log-loss toward your target (lower is better) with minimal core-logic change, I also fix a label/column alignment issue by ensuring one-hot encoding follows the class order from `sample_submission.csv` (so your softmax outputs map to the correct breed columns). Finally, I keep the submission formatting consistent with `sample_submission.csv` and write `submission.csv` to `/kaggle/working/`.'
- What this solution (achieved 1.10526) has done: 'The timeout is dominated by training epochs that repeatedly wait on input I/O/decoding and by evaluation pipelines that unnecessarily drop the last partial batch, reducing validation signal and potentially extending training. I keep the exact same model and training loop/semantics, but optimize the `tf.data` pipelines to (1) fuse image+label loading into a single map, (2) cache deterministically to disk (instead of large in-RAM caches) to avoid OOM/slowdowns and eliminate repeated JPEG decode/resize cost across epochs, and (3) use `TFRecord`-like behavior via `snapshot`-style caching paths while preserving determinism. I also ensure validation uses `drop_remainder=False` (correct evaluation semantics) and set explicit `steps_per_epoch`/`validation_steps` based on dataset sizes to prevent any extra iteration bookkeeping overhead.'
- What this solution (achieved 0.82296) has done: 'The timeout is dominated by training: you run up to 100 epochs over a ~7k-image training split with MobileNetV2 feature extraction on 256×256 images, while your input pipeline also does expensive JPEG decode/resize/augmentation every epoch and caches *in RAM* (not to disk) so it can still thrash memory. To keep the same model/loop/metric semantics but finish under 600s, the key is to (1) enable the fast protobuf implementation (your current env vars force the slow Python one), (2) switch tf.data caching from in-memory to deterministic on-disk caches so decode/resize happens once, and (3) remove unnecessary `.repeat()` on val (Keras already iterates `validation_steps`) to avoid extra work/over-prefetch. These changes are equivalent in outputs (same images, same augmentation logic, same steps/epochs/callbacks) but significantly cut CPU overhead and memory pressure, which is typically what pushes this notebook past 10 minutes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON", None)

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
else:
    strategy = tf.distribute.get_strategy()

print(
    f"GPUs: {len(gpus)} | Strategy: {type(strategy).__name__} | Replicas: {strategy.num_replicas_in_sync}"
)



## === cell 2
PATH = "/kaggle/input/dog-breed-identification"
NUM_CHANNEL = 3
INPUT_SHAPE = 256
BATCH_SIZE = 32 * strategy.num_replicas_in_sync




## === cell 3
class ImageDatastore:
    def __init__(self, path, csv, output_shape, train_val_test, class_names=None):
        self.path = path
        self.csv = csv
        self.output_shape = output_shape
        self.train_val_test = train_val_test
        self.class_names = class_names
        self.image_paths, self.labels = self.get_files_and_labels()

    def get_files_and_labels(self):
        image_paths = [os.path.join(self.path, idx) + ".jpg" for idx in self.csv.index]
        if self.train_val_test == "test":
            labels = None
        else:
            if self.class_names is None:
                labels = pd.get_dummies(self.csv.breed).astype("uint8").to_numpy()
            else:
                labels = (
                    pd.get_dummies(self.csv.breed)
                    .reindex(columns=self.class_names, fill_value=0)
                    .astype("uint8")
                    .to_numpy()
                )
        return image_paths, labels

    def __call__(self):
        if self.train_val_test == "test":
            for p in self.image_paths:
                yield p
        else:
            for p, y in zip(self.image_paths, self.labels):
                yield p, y




## === cell 4
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor="loss", factor=0.5, patience=0, min_lr=0.01):
        super(CustomCallback, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.job = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if "loss" in self.monitor:
            self.best = np.inf
        else:
            self.best = -1

    def _get_lr(self):
        lr_obj = self.model.optimizer.learning_rate
        try:
            return float(K.get_value(lr_obj))
        except Exception:
            try:
                return float(tf.keras.backend.get_value(lr_obj))
            except Exception:
                return float(lr_obj)

    def _set_lr(self, new_lr: float):
        lr_obj = self.model.optimizer.learning_rate
        try:
            if hasattr(lr_obj, "assign"):
                lr_obj.assign(new_lr)
            else:
                self.model.optimizer.learning_rate = new_lr
        except Exception:
            try:
                K.set_value(lr_obj, new_lr)
            except Exception:
                self.model.optimizer.learning_rate = new_lr

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = False
        if "loss" in self.monitor and current < self.best:
            improved = True
        elif "acc" in self.monitor and current > self.best:
            improved = True

        if improved:
            self.best_weights = self.model.get_weights()
            self.best = current
            self.wait = 0
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        if self.job == 0:
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

            lr = self._get_lr()
            new_lr = lr * self.factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
                self.job = 1

            self._set_lr(new_lr)
            self.wait = 0
            print(
                f"\nLearning rate reduced from {'{:.3g}'.format(lr)} to {'{:.3g}'.format(new_lr)}"
            )
        elif self.job == 1:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")




## === cell 5
sub_template_df = pd.read_csv(
    os.path.join(PATH, "sample_submission.csv"), index_col="id"
)
CLASS_NAMES = list(sub_template_df.columns)
NUM_CLASS = len(CLASS_NAMES)

train_df = pd.read_csv(os.path.join(PATH, "labels.csv"), index_col="id")
train_df.head()



## === cell 6
val_ratio = 0.2
num_sample = max(1, int(len(train_df) * val_ratio / NUM_CLASS))



## === cell 7
val_df = pd.concat(
    [
        train_df[train_df.breed == lbl].sample(n=num_sample, random_state=SEED)
        for lbl in train_df.breed.unique()
    ],
    axis=0,
)
val_df = val_df.sample(frac=1, random_state=SEED)
train_df = train_df.drop(val_df.index)



## === cell 8
train_ds = ImageDatastore(
    os.path.join(PATH, "train"),
    train_df,
    (INPUT_SHAPE, INPUT_SHAPE),
    "train",
    class_names=CLASS_NAMES,
)
val_ds = ImageDatastore(
    os.path.join(PATH, "train"),
    val_df,
    (INPUT_SHAPE, INPUT_SHAPE),
    "val",
    class_names=CLASS_NAMES,
)



## === cell 9
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)




## === cell 10
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _decode_resize_rgb(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [INPUT_SHAPE, INPUT_SHAPE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.uint8)


@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=(None, INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype=tf.uint8
        ),
        tf.TensorSpec(shape=(None, NUM_CLASS), dtype=tf.uint8),
    ]
)
def _cast_train(X, y):
    return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(NUM_CLASS,), dtype=tf.uint8),
    ]
)
def _load_xy(path, y):
    return _decode_resize_rgb(path), y


def _dataset_options(deterministic: bool):
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    opts.autotune.enabled = True
    opts.experimental_optimization.map_fusion = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def _maybe_prefetch_to_device(ds):
    try:
        if len(tf.config.list_physical_devices("GPU")) >= 1:
            return ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    except Exception:
        pass
    return ds


def make_train_dataset(image_paths, labels, cache_path: str):
    image_paths = tf.convert_to_tensor(image_paths)
    labels = tf.convert_to_tensor(labels)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.with_options(_dataset_options(deterministic=False))

    ds = ds.map(_load_xy, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache(cache_path)

    buffer = min(int(image_paths.shape[0]), max(2048, BATCH_SIZE * 64))
    ds = ds.shuffle(buffer_size=buffer, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    ds = ds.map(
        lambda X, y: (data_augmentation(X, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(_cast_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_eval_dataset(image_paths, labels, cache_path: str):
    image_paths = tf.convert_to_tensor(image_paths)
    labels = tf.convert_to_tensor(labels)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.with_options(_dataset_options(deterministic=False))

    ds = ds.map(_load_xy, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache()

    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_cast_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_cache = os.path.join("/kaggle/working", f"cache_train_{INPUT_SHAPE}.tfdata")
val_cache = os.path.join("/kaggle/working", f"cache_val_{INPUT_SHAPE}.tfdata")

train = make_train_dataset(train_ds.image_paths, train_ds.labels, train_cache)
val = make_eval_dataset(val_ds.image_paths, val_ds.labels, val_cache)

train_steps = max(1, len(train_ds.image_paths) // BATCH_SIZE)
val_steps = max(1, int(np.ceil(len(val_ds.image_paths) / BATCH_SIZE)))

print("train_steps:", train_steps, "val_steps:", val_steps)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2152456828.py in <cell line: 0>()
    116 val_cache = os.path.join("/kaggle/working", f"cache_val_{INPUT_SHAPE}.tfdata")
    117 
--> 118 train = make_train_dataset(train_ds.image_paths, train_ds.labels, train_cache)
    119 val = make_eval_dataset(val_ds.image_paths, val_ds.labels, val_cache)
    120 

/tmp/ipykernel_11/2152456828.py in make_train_dataset(image_paths, labels, cache_path)
     82         num_parallel_calls=tf.data.AUTOTUNE,
     83     )
---> 84     ds = ds.map(_cast_train, num_parallel_calls=tf.data.AUTOTUNE)
     85 
     86     ds = ds.repeat()

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/function_type_utils.py in bind_function_inputs(args, kwargs, function_type, default_values)
    444     )
    445   except Exception as e:
--> 446     raise TypeError(
    447         f"Binding inputs to tf.function failed due to `{e}`. "
    448         f"Received args: {args} and kwargs: {sanitized_kwargs} for signature:"

TypeError: Binding inputs to tf.function failed due to `Can not cast TensorSpec(shape=(32, 256, 256, 3), dtype=tf.float32, name=None) to TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8, name=None)`. Received args: (<tf.Tensor 'args_0:0' shape=(32, 256, 256, 3) dtype=float32>, <tf.Tensor 'args_1:0' shape=(32, 120) dtype=uint8>) and kwargs: {} for signature: (X: TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8, name=None), y: TensorSpec(shape=(None, 120), dtype=tf.uint8, name=None)).

## === cell 11
def build_model():
    input_shape = (INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False
    base.training = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASS, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        jit_compile=True,
    )
    return model




## === cell 12
with strategy.scope():
    model = build_model()

model.summary()



## === cell 13
history = model.fit(
    train,
    epochs=100,
    steps_per_epoch=train_steps,
    validation_data=val,
    validation_steps=val_steps,
    callbacks=[
        CustomCallback(monitor="val_loss", factor=0.5, patience=10, min_lr=2e-6)
    ],
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/814659236.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train,
      3     epochs=100,
      4     steps_per_epoch=train_steps,
      5     validation_data=val,

NameError: name 'train' is not defined

## === cell 14
sub_test_df = sub_template_df.copy()



## === cell 15
sub_test_ds = ImageDatastore(
    os.path.join(PATH, "test"), sub_test_df, (INPUT_SHAPE, INPUT_SHAPE), "test"
)




## === cell 16
@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=(None, INPUT_SHAPE, INPUT_SHAPE, NUM_CHANNEL), dtype=tf.uint8
        )
    ]
)
def _cast_test(X):
    return tf.cast(X, tf.float32)


def make_test_dataset(image_paths, cache_path: str):
    image_paths = tf.convert_to_tensor(image_paths)
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.with_options(_dataset_options(deterministic=False))

    ds = ds.map(_decode_resize_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache()

    ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_cast_test, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


test_cache = os.path.join("/kaggle/working", f"cache_test_{INPUT_SHAPE}.tfdata")
sub_test = make_test_dataset(sub_test_ds.image_paths, test_cache)

pred = model.predict(sub_test, verbose=1)
pred = pred[: len(sub_test_df)]

sub_test_df.iloc[:, :] = pred

out_path = os.path.join("/kaggle", "working", "submission.csv")
sub_test_df.to_csv(out_path)
print("Wrote:", out_path)
print(sub_test_df.shape)
sub_test_df.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/3867841649.py in <cell line: 0>()
     31 sub_test = make_test_dataset(sub_test_ds.image_paths, test_cache)
     32 
---> 33 pred = model.predict(sub_test, verbose=1)
     34 pred = pred[: len(sub_test_df)]
     35 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

FailedPreconditionError: {{function_node __wrapped__DatasetToGraphV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Failed to serialize the input pipeline graph: GeneratorDatasetOp::Dataset does not support serialization [Op:DatasetToGraphV2] name:
