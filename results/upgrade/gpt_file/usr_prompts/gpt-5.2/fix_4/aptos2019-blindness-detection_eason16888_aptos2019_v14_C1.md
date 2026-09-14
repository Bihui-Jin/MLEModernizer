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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
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
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
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


_TOL = tf.constant(7, tf.int32)
_SIGMAX = 10.0


def _tf_crop_image_from_gray_uint8(img_uint8_rgb, tol=_TOL):
    """Crop borders where grayscale <= tol. If mask is empty, return original."""
    gray = tf.image.rgb_to_grayscale(img_uint8_rgb)  # uint8 -> uint8 (internally)
    gray = tf.squeeze(gray, axis=-1)  # [H,W]
    mask = gray > tol

    any_row = tf.reduce_any(mask, axis=1)
    any_col = tf.reduce_any(mask, axis=0)

    def _crop():
        ys = tf.where(any_row)[:, 0]
        xs = tf.where(any_col)[:, 0]
        y0 = tf.reduce_min(ys)
        y1 = tf.reduce_max(ys) + 1
        x0 = tf.reduce_min(xs)
        x1 = tf.reduce_max(xs) + 1
        return img_uint8_rgb[y0:y1, x0:x1, :]

    return tf.cond(tf.reduce_any(mask), _crop, lambda: img_uint8_rgb)


def _tf_load_ben_preprocess(path):
    """Read PNG, crop, resize, Ben Graham enhancement, return float32 [0,1]."""
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_png(bytes_, channels=3)  # uint8 RGB
    img = _tf_crop_image_from_gray_uint8(img, tol=_TOL)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)

    gauss = tf.image.gaussian_filter2d(img, sigma=_SIGMAX, filter_shape=None)
    img = 4.0 * img - 4.0 * gauss + 128.0

    img = tf.clip_by_value(img, 0.0, 255.0) * (1.0 / 255.0)
    return img




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
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "/kaggle/input/aptos2019-blindness-detection"


input_dir = get_input_base_dir()
train_csv_path = os.path.join(input_dir, "train.csv")
test_csv_path = os.path.join(input_dir, "test.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
train_img_dir = os.path.join(input_dir, "train_images")
test_img_dir = os.path.join(input_dir, "test_images")

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

    def make_dataset(df, images_dir, training, cache=False):
        ids = df["id_code"].astype(str).values
        paths = tf.strings.join(
            [tf.constant(images_dir + os.sep), tf.constant(ids), tf.constant(".png")]
        )
        labels = tf.constant(df["diagnosis"].values.astype("int64"))

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        options = tf.data.Options()
        options.experimental_deterministic = True  # preserve reproducibility
        ds = ds.with_options(options)

        if training:
            ds = ds.shuffle(
                min(int(df.shape[0]), 2048), seed=SEED, reshuffle_each_iteration=True
            )

        def _map(path, label):
            img = _tf_load_ben_preprocess(path)
            img.set_shape((IMG_SIZE, IMG_SIZE, 3))
            label.set_shape(())
            return img, label

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

        if cache:
            ds = ds.cache()

        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(trn_df, train_img_dir, training=True, cache=False)
    val_ds = make_dataset(val_df, train_img_dir, training=False, cache=True)

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
        epochs=3,  # unchanged
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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/807902991.py in <cell line: 0>()
     88         return ds
     89 
---> 90     train_ds = make_dataset(trn_df, train_img_dir, training=True, cache=False)
     91     val_ds = make_dataset(val_df, train_img_dir, training=False, cache=True)
     92 

/tmp/ipykernel_11/807902991.py in make_dataset(df, images_dir, training, cache)
     80             return img, label
     81 
---> 82         ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
     83 
     84         if cache:

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

/tmp/__autograph_generated_fileo2kv17m1.py in tf___map(path, label)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_tf_load_ben_preprocess), (ag__.ld(path),), None, fscope)
     11                 ag__.converted_call(ag__.ld(img).set_shape, ((ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE), 3),), None, fscope)
     12                 ag__.converted_call(ag__.ld(label).set_shape, ((),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file7jrb0upc.py in tf___tf_load_ben_preprocess(path)
     11                 bytes_ = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.decode_png, (ag__.ld(bytes_),), dict(channels=3), fscope)
---> 13                 img = ag__.converted_call(ag__.ld(_tf_crop_image_from_gray_uint8), (ag__.ld(img),), dict(tol=ag__.ld(_TOL)), fscope)
     14                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE)]), dict(method='bilinear', antialias=False), fscope)
     15                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filecqie643d.py in tf___tf_crop_image_from_gray_uint8(img_uint8_rgb, tol)
     11                 gray = ag__.converted_call(ag__.ld(tf).image.rgb_to_grayscale, (ag__.ld(img_uint8_rgb),), None, fscope)
     12                 gray = ag__.converted_call(ag__.ld(tf).squeeze, (ag__.ld(gray),), dict(axis=-1), fscope)
---> 13                 mask = ag__.ld(gray) > ag__.ld(tol)
     14                 any_row = ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.ld(mask),), dict(axis=1), fscope)
     15                 any_col = ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.ld(mask),), dict(axis=0), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_math_operator_overrides.py in wrapper(x, y, *args, **kwargs)
    148   def wrapper(x, y, *args, **kwargs):
    149     x, y = override_binary_operator.maybe_promote_tensors(x, y)
--> 150     return fn(x, y, *args, **kwargs)
    151 
    152   return tf_decorator.make_decorator(fn, wrapper)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_math_ops.py in greater(x, y, name)
   4302   # Add nodes to the TensorFlow graph.
   4303   try:
-> 4304     _, _, _op, _outputs = _op_def_library._apply_op_helper(
   4305         "Greater", x=x, y=y, name=name)
   4306   except (TypeError, ValueError):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    650       if input_arg.type_attr in attrs:
    651         if attrs[input_arg.type_attr] != attr_value:
--> 652           raise TypeError(
    653               f"Input '{input_name}' of '{op_type_name}' Op has type "
    654               f"{dtypes.as_dtype(attr_value).name} that does not match type "

TypeError: in user code:

    File "/tmp/ipykernel_11/807902991.py", line 76, in _map  *
        img = _tf_load_ben_preprocess(path)
    File "/tmp/ipykernel_11/209830639.py", line 83, in _tf_load_ben_preprocess  *
        img = _tf_crop_image_from_gray_uint8(img, tol=_TOL)
    File "/tmp/ipykernel_11/209830639.py", line 61, in _tf_crop_image_from_gray_uint8  *
        mask = gray > tol

    TypeError: Input 'y' of 'Greater' Op has type int32 that does not match type uint8 of argument 'x'.


## === cell 3
test_csv = pd.read_csv(sample_sub_path)
id_code = test_csv["id_code"].values


def make_test_dataset(ids, images_dir):
    ids = tf.constant(ids.astype(str))
    paths = tf.strings.join(
        [tf.constant(images_dir + os.sep), ids, tf.constant(".png")]
    )

    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    def _map(path):
        img = _tf_load_ben_preprocess(path)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(id_code, test_img_dir)

pred = model.predict(test_ds, verbose=0)
test_prediction = np.asarray(np.argmax(pred, axis=1), dtype="int64")

del test_ds, pred
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/33653606.py in <cell line: 0>()
     25 
     26 
---> 27 test_ds = make_test_dataset(id_code, test_img_dir)
     28 
     29 pred = model.predict(test_ds, verbose=0)

/tmp/ipykernel_11/33653606.py in make_test_dataset(ids, images_dir)
     20         return img
     21 
---> 22     ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
     23     ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
     24     return ds

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

/tmp/__autograph_generated_fileoatjhv61.py in tf___map(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_tf_load_ben_preprocess), (ag__.ld(path),), None, fscope)
     11                 ag__.converted_call(ag__.ld(img).set_shape, ((ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE), 3),), None, fscope)
     12                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file7jrb0upc.py in tf___tf_load_ben_preprocess(path)
     11                 bytes_ = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.decode_png, (ag__.ld(bytes_),), dict(channels=3), fscope)
---> 13                 img = ag__.converted_call(ag__.ld(_tf_crop_image_from_gray_uint8), (ag__.ld(img),), dict(tol=ag__.ld(_TOL)), fscope)
     14                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(IMG_SIZE), ag__.ld(IMG_SIZE)]), dict(method='bilinear', antialias=False), fscope)
     15                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filecqie643d.py in tf___tf_crop_image_from_gray_uint8(img_uint8_rgb, tol)
     11                 gray = ag__.converted_call(ag__.ld(tf).image.rgb_to_grayscale, (ag__.ld(img_uint8_rgb),), None, fscope)
     12                 gray = ag__.converted_call(ag__.ld(tf).squeeze, (ag__.ld(gray),), dict(axis=-1), fscope)
---> 13                 mask = ag__.ld(gray) > ag__.ld(tol)
     14                 any_row = ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.ld(mask),), dict(axis=1), fscope)
     15                 any_col = ag__.converted_call(ag__.ld(tf).reduce_any, (ag__.ld(mask),), dict(axis=0), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_math_operator_overrides.py in wrapper(x, y, *args, **kwargs)
    148   def wrapper(x, y, *args, **kwargs):
    149     x, y = override_binary_operator.maybe_promote_tensors(x, y)
--> 150     return fn(x, y, *args, **kwargs)
    151 
    152   return tf_decorator.make_decorator(fn, wrapper)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_math_ops.py in greater(x, y, name)
   4302   # Add nodes to the TensorFlow graph.
   4303   try:
-> 4304     _, _, _op, _outputs = _op_def_library._apply_op_helper(
   4305         "Greater", x=x, y=y, name=name)
   4306   except (TypeError, ValueError):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    650       if input_arg.type_attr in attrs:
    651         if attrs[input_arg.type_attr] != attr_value:
--> 652           raise TypeError(
    653               f"Input '{input_name}' of '{op_type_name}' Op has type "
    654               f"{dtypes.as_dtype(attr_value).name} that does not match type "

TypeError: in user code:

    File "/tmp/ipykernel_11/33653606.py", line 18, in _map  *
        img = _tf_load_ben_preprocess(path)
    File "/tmp/ipykernel_11/209830639.py", line 83, in _tf_load_ben_preprocess  *
        img = _tf_crop_image_from_gray_uint8(img, tol=_TOL)
    File "/tmp/ipykernel_11/209830639.py", line 61, in _tf_crop_image_from_gray_uint8  *
        mask = gray > tol

    TypeError: Input 'y' of 'Greater' Op has type int32 that does not match type uint8 of argument 'x'.


## === cell 4
test_csv["diagnosis"] = test_prediction.astype("int64")
test_csv.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print(tmp)
print("Wrote submission.csv with shape:", test_csv.shape)
print("Done!")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3460773115.py in <cell line: 0>()
----> 1 test_csv["diagnosis"] = test_prediction.astype("int64")
      2 test_csv.to_csv("submission.csv", index=False)
      3 
      4 unique, counts = np.unique(test_prediction, return_counts=True)
      5 tmp = dict(zip(unique.tolist(), counts.tolist()))

NameError: name 'test_prediction' is not defined
