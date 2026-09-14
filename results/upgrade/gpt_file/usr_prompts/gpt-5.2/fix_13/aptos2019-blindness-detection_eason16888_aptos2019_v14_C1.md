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

3.10

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

0.0313057905882835

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import gc
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    _CPU = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _CPU))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None

    if img.ndim == 2:
        mask = (img > tol).astype(np.uint8)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return img
        x, y, w, h = cv2.boundingRect(pts)
        return img[y : y + h, x : x + w]

    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = (gray_img > tol).astype(np.uint8)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return img
        x, y, w, h = cv2.boundingRect(pts)
        return img[y : y + h, x : x + w, :]

    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


_TOL_INT = 7
_SIGMAX = 10.0


def _tf_crop_from_gray(img_u8, tol=_TOL_INT):
    gray = tf.image.rgb_to_grayscale(img_u8)  # uint8 -> uint8
    gray2 = tf.squeeze(gray, axis=-1)  # [H,W]
    mask = gray2 > tf.cast(tol, gray2.dtype)

    def _no_crop():
        return img_u8

    def _do_crop():
        ys = tf.where(tf.reduce_any(mask, axis=1))[:, 0]
        xs = tf.where(tf.reduce_any(mask, axis=0))[:, 0]
        y0 = tf.reduce_min(ys)
        y1 = tf.reduce_max(ys) + 1
        x0 = tf.reduce_min(xs)
        x1 = tf.reduce_max(xs) + 1
        return img_u8[y0:y1, x0:x1, :]

    return tf.cond(tf.reduce_any(mask), _do_crop, _no_crop)


def _tf_load_ben_preprocess(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_png(bytes_, channels=3)  # uint8 RGB
    img = _tf_crop_from_gray(img, tol=_TOL_INT)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img_f = tf.cast(img, tf.float32)

    blur = tf.image.gaussian_filter2d(img_f, sigma=_SIGMAX)
    img_f = 4.0 * img_f + (-4.0) * blur + 128.0
    img_f = tf.clip_by_value(img_f, 0.0, 255.0) * (1.0 / 255.0)
    img_f.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img_f




## === cell 2
EFF_MODEL_PATH = "../input/eff-b0-model/eff_b0_model"


def build_fallback_model():
    base = DenseNet121(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_input_base_dir():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.basename(c) == "data" and os.path.exists(
                os.path.join(c, "aptos2019-blindness-detection")
            ):
                return os.path.join(c, "aptos2019-blindness-detection")
            return c
    return "/kaggle/input/aptos2019-blindness-detection"


input_dir = get_input_base_dir()
train_csv_path = os.path.join(input_dir, "train.csv")
test_csv_path = os.path.join(input_dir, "test.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
train_img_dir = os.path.join(input_dir, "train_images")
test_img_dir = os.path.join(input_dir, "test_images")

_CPU = os.cpu_count() or 2
_MAP_PARALLEL = tf.data.AUTOTUNE

if os.path.exists(EFF_MODEL_PATH):
    model = keras.models.load_model(EFF_MODEL_PATH, compile=False)
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception:
        pass
else:
    train_df = pd.read_csv(train_csv_path)
    trn_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
    )

    def make_dataset(df, images_dir, training, cache=False, cache_path=None):
        ids = df["id_code"].astype(str).values
        full_paths = (images_dir + os.sep + ids + ".png").astype("U")
        labels_np = df["diagnosis"].values.astype("int64")

        ds = tf.data.Dataset.from_tensor_slices((full_paths, labels_np))

        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)

        if training:
            ds = ds.shuffle(
                min(int(df.shape[0]), 2048), seed=SEED, reshuffle_each_iteration=True
            )

        def _map(path, label):
            img = _tf_load_ben_preprocess(path)
            label.set_shape(())
            return img, label

        ds = ds.map(_map, num_parallel_calls=_MAP_PARALLEL, deterministic=True)

        if cache:
            ds = ds.cache()

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(
        trn_df,
        train_img_dir,
        training=True,
        cache=True,
        cache_path="train_cache.tfdata",
    )
    val_ds = make_dataset(
        val_df, train_img_dir, training=False, cache=True, cache_path="val_cache.tfdata"
    )

    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=trn_df["diagnosis"].values,
    )
    cw = {i: float(w) for i, w in enumerate(cw)}

    model = build_fallback_model()
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
        ),
        ModelCheckpoint(
            "fallback_best.keras", monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,  # unchanged core logic
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )
    if os.path.exists("fallback_best.keras"):
        model = keras.models.load_model("fallback_best.keras")

    del train_ds, val_ds, train_df, trn_df, val_df
    gc.collect()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1444226953.py in <cell line: 0>()
     95         return ds
     96 
---> 97     train_ds = make_dataset(
     98         trn_df,
     99         train_img_dir,

/tmp/ipykernel_11/1444226953.py in make_dataset(df, images_dir, training, cache, cache_path)
     84             return img, label
     85 
---> 86         ds = ds.map(_map, num_parallel_calls=_MAP_PARALLEL, deterministic=True)
     87 
     88         # --- Speed fix (preserves correctness): avoid writing huge on-disk cache files.

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

/tmp/__autograph_generated_filev_5jkdlz.py in tf___map(path, label)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_tf_load_ben_preprocess), (ag__.ld(path),), None, fscope)
     11                 ag__.converted_call(ag__.ld(label).set_shape, ((),), None, fscope)
     12                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filekhx2nl3p.py in tf___tf_load_ben_preprocess(path)
     13                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), (ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE))), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
     14                 img_f = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
---> 15                 blur = ag__.converted_call(ag__.ld(tf).image.gaussian_filter2d, (ag__.ld(img_f),), dict(sigma=ag__.ld(_SIGMAX)), fscope)
     16                 img_f = 4.0 * ag__.ld(img_f) + -4.0 * ag__.ld(blur) + 128.0
     17                 img_f = ag__.converted_call(ag__.ld(tf).clip_by_value, (ag__.ld(img_f), 0.0, 255.0), None, fscope) * (1.0 / 255.0)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1444226953.py", line 82, in _map  *
        img = _tf_load_ben_preprocess(path)
    File "/tmp/ipykernel_11/1220538642.py", line 92, in _tf_load_ben_preprocess  *
        blur = tf.image.gaussian_filter2d(img_f, sigma=_SIGMAX)

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'gaussian_filter2d'


## === cell 3
test_csv = pd.read_csv(test_csv_path)
id_code = test_csv["id_code"].astype(str).values


def make_test_dataset(ids, images_dir):
    full_paths = (images_dir + os.sep + ids.astype(str) + ".png").astype("U")

    ds = tf.data.Dataset.from_tensor_slices(full_paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    def _map(path):
        img = _tf_load_ben_preprocess(path)
        return img

    ds = ds.map(_map, num_parallel_calls=_MAP_PARALLEL, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(id_code, test_img_dir)

pred = model.predict(test_ds, verbose=0)
test_prediction = np.asarray(np.argmax(pred, axis=1), dtype="int64")

del test_ds, pred
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2884933496.py in <cell line: 0>()
     24 
     25 
---> 26 test_ds = make_test_dataset(id_code, test_img_dir)
     27 
     28 pred = model.predict(test_ds, verbose=0)

/tmp/ipykernel_11/2884933496.py in make_test_dataset(ids, images_dir)
      4 
      5 def make_test_dataset(ids, images_dir):
----> 6     full_paths = (images_dir + os.sep + ids.astype(str) + ".png").astype("U")
      7 
      8     ds = tf.data.Dataset.from_tensor_slices(full_paths)

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U51'), dtype('<U12')) -> None

## === cell 4
sub = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction.astype("int64")})
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print(tmp)
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3766998422.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction.astype("int64")})
      2 sub.to_csv("submission.csv", index=False)
      3 
      4 unique, counts = np.unique(test_prediction, return_counts=True)
      5 tmp = dict(zip(unique.tolist(), counts.tolist()))

NameError: name 'test_prediction' is not defined
