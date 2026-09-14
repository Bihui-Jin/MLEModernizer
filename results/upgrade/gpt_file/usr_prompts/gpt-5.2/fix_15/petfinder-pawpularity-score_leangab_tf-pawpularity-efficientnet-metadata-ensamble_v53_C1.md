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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

18.408625212578105

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

print("Python:", __import__("sys").version)
print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
lazy_submit = True  # set to False to train
SEED = 42

keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

IMAGE_SIZE = (224, 224)
folds = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DATA_ROOT = _resolve_path(
    "/kaggle/input/petfinder-pawpularity-score",
    "../input/petfinder-pawpularity-score",
)

TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_DS = os.path.join(DATA_ROOT, "train.csv")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test")
TEST_DS = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_DS = os.path.join(DATA_ROOT, "sample_submission.csv")

if lazy_submit:
    weights = "../input/weights-pawpularity"
else:
    weights = "../working/"
if not os.path.exists(weights):
    weights = "../working/"

WEIGHTS_PREFIX = "efficientnetb0_"
WEIGHTS_SUFFIX = ".weights.h5"

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 2
try:
    import subprocess

    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi not available:", repr(e))



## === cell 3
sample_img_path = _resolve_path(
    os.path.join(TRAIN_IMAGES_DIR, "00524dbf2637a80cbc80f70d3ff59616.jpg"),
    "../input/petfinder-pawpularity-score/train/00524dbf2637a80cbc80f70d3ff59616.jpg",
    "/kaggle/input/petfinder-pawpularity-score/train/00524dbf2637a80cbc80f70d3ff59616.jpg",
)
img = keras.utils.load_img(sample_img_path, target_size=IMAGE_SIZE)
img = keras.utils.img_to_array(img) / 255.0
print(img.shape)



## === cell 4
train_ds = pd.read_csv(TRAIN_DS)
test_ds = pd.read_csv(TEST_DS)
subm_ds = pd.read_csv(SUBMISSION_DS)
print(train_ds.shape, test_ds.shape, subm_ds.shape)



## === cell 5
meta_cols = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]




## === cell 6
def _build_paths_and_arrays(
    df: pd.DataFrame, img_dir: str, id_col: str, tab_cols, target_col=None
):
    ids = df[id_col].astype(str).to_numpy()
    img_paths = np.asarray([os.path.join(img_dir, f"{i}.jpg") for i in ids], dtype=str)

    tabs = df[tab_cols].to_numpy(dtype=np.float32, copy=True)
    if target_col is None:
        return img_paths, tabs
    y = df[target_col].to_numpy(dtype=np.float32, copy=True)
    return img_paths, tabs, y


def make_dataset(
    df: pd.DataFrame,
    img_dir: str,
    batch_size: int,
    tab_columns,
    id_col: str,
    target_col: str,
    is_train: bool,
    shuffle: bool = True,
    seed: int = SEED,
):
    """
    FIX: Ensure the dataset yields inputs in the exact structure expected by the model:
         ((image, tabular), y) for train/val, and (image, tabular) for test.
    """
    if is_train:
        img_paths, tabs, y = _build_paths_and_arrays(
            df, img_dir, id_col, tab_columns, target_col
        )
        ds = tf.data.Dataset.from_tensor_slices((img_paths, tabs, y))
    else:
        img_paths, tabs = _build_paths_and_arrays(
            df, img_dir, id_col, tab_columns, None
        )
        ds = tf.data.Dataset.from_tensor_slices((img_paths, tabs))

    if shuffle and is_train:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    def _load_img(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32)
        return img

    if is_train:

        def _map_fn(path, tab, y_):
            img = _load_img(path)
            return (img, tab), y_

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()
    else:

        def _map_fn(path, tab):
            img = _load_img(path)
            return (img, tab)

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 7
NCOL = len(meta_cols)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

_EFFNET_LOCAL = _resolve_path(
    "../input/efficientnet-b0/efficientnetb0_notop.h5",
    "/kaggle/input/efficientnet-b0/efficientnetb0_notop.h5",
)


def get_model():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomContrast(0.2, seed=SEED),
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.1, seed=SEED),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=SEED),
        ]
    )

    effnet_weights = _EFFNET_LOCAL if os.path.exists(_EFFNET_LOCAL) else "imagenet"

    conv_base = tf.keras.applications.efficientnet.EfficientNetB0(
        weights=effnet_weights,
        include_top=False,
        input_tensor=keras.Input(shape=INPUT_SHAPE),
    )
    conv_base.trainable = False

    inp = keras.Input(shape=INPUT_SHAPE)
    out = data_augmentation(inp)
    out = conv_base(out)
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.BatchNormalization()(out)

    meta_input = keras.Input(shape=(NCOL,))
    out_meta = layers.Dense(16, activation="relu")(meta_input)
    out_meta = layers.Dropout(0.1)(out_meta)
    out_meta = layers.Dense(16, activation="relu")(out_meta)
    out_meta = layers.Dropout(0.1)(out_meta)
    out_meta = layers.Dense(16, activation="relu")(out_meta)

    concat = layers.Concatenate(axis=1)([out, out_meta])
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16)(concat)
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(16)(concat)
    concat = layers.Dropout(0.1)(concat)
    concat = layers.Dense(1, activation="linear")(concat)

    model = keras.Model([inp, meta_input], concat)
    return model




## === cell 8
model = get_model()
model.summary()



## === cell 9
try:
    pass
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
BATCH_SIZE = 120
kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)


def train():
    EPOCHS = 10

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_root_mean_squared_error",
        patience=2,
        factor=0.5,
        verbose=1,
        mode="min",
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error", patience=5
    )

    def scheduler(epoch, lr):
        if epoch == 1:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1 * lr))

    sch = tf.keras.callbacks.LearningRateScheduler(scheduler)

    models = []
    evals = 0.0
    histories = []
    for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
        train_df = train_ds.iloc[train_idx]
        val_df = train_ds.iloc[val_idx]

        train_data = make_dataset(
            df=train_df,
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id_col="Id",
            target_col="Pawpularity",
            is_train=True,
            shuffle=True,
            seed=SEED,
        )
        val_data = make_dataset(
            df=val_df,
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id_col="Id",
            target_col="Pawpularity",  # unused for is_train=False
            is_train=False,
            shuffle=False,
            seed=SEED,
        ).map(lambda x: (x[0], x[1]), num_parallel_calls=AUTOTUNE, deterministic=True)

        model = get_model()
        optimizer = keras.optimizers.Adam(1e-3)

        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=[keras.metrics.RootMeanSquaredError()],
            run_eagerly=False,
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            f"../working/{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}",
            monitor="val_root_mean_squared_error",
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=[reduce_lr, checkpoint, es, sch],
            verbose=1,
        )

        evals += float(model.evaluate(val_data, verbose=0)[1])
        models.append(model)
        histories.append(history)

        del train_data, val_data
        gc.collect()

    evals /= folds
    return models, evals, histories




## === cell 11
lazy_eval = False  # Set to True to evaluate with the pretrained weights

need_train_fallback = False
if lazy_submit:
    for idx in range(folds):
        wpath = os.path.join(weights, f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        if not os.path.exists(wpath):
            wpath2 = os.path.join(
                "../working", f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}"
            )
            if not os.path.exists(wpath2):
                need_train_fallback = True
                break

if (not lazy_submit) or need_train_fallback:
    models, evals, histories = train()
    print("AVG RMSE:", evals)
else:
    models = []
    evals = 0.0
    for idx, (train_idx, val_idx) in enumerate(kf.split(train_ds)):
        val_df = train_ds.iloc[val_idx]

        val_data = make_dataset(
            df=val_df,
            img_dir=TRAIN_IMAGES_DIR,
            batch_size=BATCH_SIZE,
            tab_columns=meta_cols,
            id_col="Id",
            target_col="Pawpularity",  # unused for is_train=False
            is_train=False,
            shuffle=False,
            seed=SEED,
        ).map(lambda x: (x[0], x[1]), num_parallel_calls=AUTOTUNE, deterministic=True)

        model = get_model()
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(1e-3),
            metrics=[keras.metrics.RootMeanSquaredError()],
            run_eagerly=False,
        )

        wpath = os.path.join(weights, f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        if not os.path.exists(wpath):
            wpath = os.path.join("../working", f"{WEIGHTS_PREFIX}{idx}{WEIGHTS_SUFFIX}")
        model.load_weights(wpath)

        models.append(model)

        if lazy_eval:
            evals += float(model.evaluate(val_data, verbose=0)[1]) / folds
            print("AVG RMSE:", evals)

        del val_data
        gc.collect()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/69877353.py in <cell line: 0>()
     14 
     15 if (not lazy_submit) or need_train_fallback:
---> 16     models, evals, histories = train()
     17     print("AVG RMSE:", evals)
     18 else:

/tmp/ipykernel_55/217091025.py in train()
     56             shuffle=False,
     57             seed=SEED,
---> 58         ).map(lambda x: (x[0], x[1]), num_parallel_calls=AUTOTUNE, deterministic=True)
     59 
     60         model = get_model()

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

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.<lambda>() takes 1 positional argument but 2 were given


## === cell 12
predictions = []
test_data = make_dataset(
    df=test_ds,
    img_dir=TEST_IMAGES_DIR,
    batch_size=120,
    tab_columns=meta_cols,
    id_col="Id",
    target_col="Pawpularity",  # unused for is_train=False
    is_train=False,
    shuffle=False,
    seed=SEED,
)

for i in range(folds):
    predictions.append(models[i].predict(test_data, verbose=0))

pred = np.mean(np.array(predictions), axis=0).reshape(-1).astype("float32")
pred = np.clip(pred, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_ds["Id"].values, "Pawpularity": pred})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3703553578.py in <cell line: 0>()
     14 
     15 for i in range(folds):
---> 16     predictions.append(models[i].predict(test_data, verbose=0))
     17 
     18 pred = np.mean(np.array(predictions), axis=0).reshape(-1).astype("float32")

NameError: name 'models' is not defined

## === cell 13
print(submission.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2115235593.py in <cell line: 0>()
----> 1 print(submission.head())
      2 print(
      3     "submission.csv exists:",
      4     os.path.exists("submission.csv"),
      5     "size:",

NameError: name 'submission' is not defined
