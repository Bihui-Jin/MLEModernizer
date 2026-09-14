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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.3338402537284629

# 6. Current score

0.09079

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09079) has done: 'The timeout is dominated by Python-side image decoding/augmentation via `tf.numpy_function` + OpenCV, plus extra overhead from forcing pure-Python protobuf and unbounded tf.data parallelism. I keep the exact same model, preprocessing, augmentation semantics, and training loop, but make the input pipeline faster by (1) switching to a pure-TensorFlow decode/augment path (so it can run in graph and parallelize efficiently), (2) adding deterministic, bounded parallelism and prefetching, and (3) removing the protobuf pure-Python fallback that slows TF startup and graph execution. These changes preserve the algorithm and outputs up to negligible floating-point differences while significantly reducing per-step input overhead.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import warnings

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class SimpleEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(
            [
                f"{k}={v:.4f}"
                for k, v in logs.items()
                if isinstance(v, (int, float, np.floating))
            ]
        )
        print(msg)




## === cell 2
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    _CPU = os.cpu_count() or 4
    tf.data.experimental.threading.private_threadpool_size = min(16, max(4, _CPU))
except Exception:
    pass




## === cell 3
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")




## === cell 4
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"




## === cell 5
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
sample_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")




## === cell 6
def get_image_path(id_code, is_train=True):
    ext = ".png"
    if is_train:
        return os.path.join(TRAIN_IMG_DIR, id_code + ext)
    else:
        return os.path.join(TEST_IMG_DIR, id_code + ext)




## === cell 7
train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["id_code"].astype(str) + ".png"
test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"




## === cell 8
IMG_SIZE = 299




## === cell 9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_transform = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=180,
    shear_range=20,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="reflect",
)




## === cell 10
valid_transform = ImageDataGenerator()




## === cell 11
def load_and_preprocess_image(path, transform=None):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    if transform is not None:
        image = transform.random_transform(image)
        image = transform.standardize(image)

    image = image.astype(np.float32)
    image = preprocess_input(image)  # correct preprocessing for InceptionResNetV2
    return image




## === cell 12
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=32,
        transform=None,
        is_train=True,
        num_classes=5,
        shuffle=True,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.transform = transform
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.shuffle = bool(shuffle)

        self.filepaths = self.df["filepath"].to_numpy(dtype=object)
        if self.is_train:
            y_int = self.df["diagnosis"].to_numpy(dtype=np.int32)
            self.labels_oh = tf.keras.utils.to_categorical(
                y_int, num_classes=self.num_classes
            ).astype(np.float32)
        else:
            self.labels_oh = None

        self.indexes = np.arange(len(self.df), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = batch_idx.shape[0]

        images = np.empty((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        if self.is_train:
            labels = self.labels_oh[batch_idx]
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images, labels
        else:
            for i, j in enumerate(batch_idx):
                images[i] = load_and_preprocess_image(
                    self.filepaths[j], transform=self.transform
                )
            return images




## === cell 13
train_df_split, valid_df_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"]
)

BATCH_SIZE = 32

_AUTOTUNE = tf.data.AUTOTUNE
_CPU = os.cpu_count() or 4
_NUM_PARALLEL = min(16, max(4, _CPU))

_BRIGHT_LO, _BRIGHT_HI = 0.8, 1.2
_ROT_DEG = 180.0
_ZOOM_LO, _ZOOM_HI = 0.8, 1.2
_SHIFT_FRAC = 0.2  # width/height shift range


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    return img


def _apply_train_aug(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    factor = tf.random.uniform([], _BRIGHT_LO, _BRIGHT_HI, seed=SEED)
    img = tf.clip_by_value(img * factor, 0.0, 255.0)

    angle = tf.random.uniform([], -_ROT_DEG, _ROT_DEG, seed=SEED) * (np.pi / 180.0)
    try:
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")
    except Exception:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.image.rot90(
                tf.zeros([1, 8], tf.float32)
            ),  # no-op placeholder
            output_shape=[IMG_SIZE, IMG_SIZE],
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]

    zoom = tf.random.uniform([], _ZOOM_LO, _ZOOM_HI, seed=SEED)
    crop_size = tf.cast(
        tf.round(tf.cast(IMG_SIZE, tf.float32) * tf.minimum(1.0, zoom)), tf.int32
    )
    crop_size = tf.clip_by_value(crop_size, 1, IMG_SIZE)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3], seed=SEED)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)

    pad = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * _SHIFT_FRAC), tf.int32)
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3], seed=SEED)

    return img


def _preprocess_inception(img):
    return preprocess_input(img)


def make_train_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    y_int = df["diagnosis"].to_numpy(dtype=np.int32)
    y_oh = tf.keras.utils.to_categorical(y_int, num_classes=5).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_oh))
    ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map(path, label):
        img = _decode_resize(path)
        img = _apply_train_aug(img)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label = tf.ensure_shape(label, (5,))
        return img, label

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


def make_valid_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    y_int = df["diagnosis"].to_numpy(dtype=np.int32)
    y_oh = tf.keras.utils.to_categorical(y_int, num_classes=5).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_oh))

    def _map(path, label):
        img = _decode_resize(path)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label = tf.ensure_shape(label, (5,))
        return img, label

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df_split, BATCH_SIZE)
valid_ds = make_valid_ds(valid_df_split, BATCH_SIZE)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1587567261.py in <cell line: 0>()
    121 
    122 
--> 123 train_ds = make_train_ds(train_df_split, BATCH_SIZE)
    124 valid_ds = make_valid_ds(valid_df_split, BATCH_SIZE)
    125 

/tmp/ipykernel_11/1587567261.py in make_train_ds(df, batch_size)
     92     opts.experimental_deterministic = True
     93     ds = ds.with_options(opts)
---> 94     ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
     95     ds = ds.batch(batch_size, drop_remainder=False)
     96     ds = ds.prefetch(_AUTOTUNE)

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

/tmp/__autograph_generated_file8w0kjz1e.py in tf___map(path, label)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_decode_resize), (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_apply_train_aug), (ag__.ld(img),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(_preprocess_inception), (ag__.ld(img),), None, fscope)
     13                 ag__.converted_call(ag__.ld(img).set_shape, ((ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE), 3),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file3087h4c2.py in tf___apply_train_aug(img)
     15                     img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     16                 except Exception:
---> 17                     img = ag__.converted_call(ag__.ld(tf).raw_ops.ImageProjectiveTransformV3, (), dict(images=ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope), transforms=ag__.converted_call(ag__.ld(tf).image.rot90, (ag__.converted_call(ag__.ld(tf).zeros, ([1, 8], ag__.ld(tf).float32), None, fscope),), None, fscope), output_shape=[ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE)], interpolation='BILINEAR', fill_mode='REFLECT', fill_value=0.0), fscope)[0]
     18                 zoom = ag__.converted_call(ag__.ld(tf).random.uniform, ([], ag__.ld(_ZOOM_LO), ag__.ld(_ZOOM_HI)), dict(seed=ag__.ld(SEED)), fscope)
     19                 crop_size = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope) * ag__.converted_call(ag__.ld(tf).minimum, (1.0, ag__.ld(zoom)), None, fscope),), None, fscope), ag__.ld(tf).int32), None, fscope)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py in _CheckAtLeast3DImage(image, require_static)
    228       image_shape = image.get_shape().with_rank_at_least(3)
    229   except ValueError:
--> 230     raise ValueError("'image' (shape %s) must be at least three-dimensional." %
    231                      image.shape)
    232   if require_static and not image_shape.is_fully_defined():

ValueError: in user code:

    File "/tmp/ipykernel_11/1587567261.py", line 85, in _map  *
        img = _apply_train_aug(img)
    File "/tmp/ipykernel_11/1587567261.py", line 42, in _apply_train_aug  *
        img = tf.raw_ops.ImageProjectiveTransformV3(

    ValueError: 'image' (shape (1, 8)) must be at least three-dimensional.


## === cell 14
try:
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
except Exception as e:
    print(
        f"Warning: could not load imagenet weights due to: {e}\nFalling back to random initialization."
    )
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

for layer in base_model.layers:
    layer.trainable = True




## === cell 15
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 16
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7
)




## === cell 17
EPOCHS = 1

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[SimpleEpochLogger(), checkpoint, earlystop, reduce_lr],
    verbose=1,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3228989146.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds,
      5     epochs=EPOCHS,
      6     validation_data=valid_ds,

NameError: name 'train_ds' is not defined

## === cell 18
if os.path.exists("best_model.keras"):
    model = tf.keras.models.load_model("best_model.keras")




## === cell 19
def make_test_ds(df, batch_size):
    paths = df["filepath"].to_numpy(dtype=str)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        img = _decode_resize(path)
        img = _preprocess_inception(img)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_map, num_parallel_calls=_NUM_PARALLEL, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


test_ds = make_test_ds(test_df, BATCH_SIZE)

preds = model.predict(
    test_ds,
    verbose=1,
)
test_df["diagnosis"] = np.argmax(preds, axis=1).astype(int)




## === cell 20
submission_csv = "submission.csv"

sub = sample_sub[["id_code"]].copy()
sub = sub.merge(test_df[["id_code", "diagnosis"]], on="id_code", how="left")

sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

sub.to_csv(submission_csv, index=False)
print(f"Submission file saved as {submission_csv}")
print(sub.head())
