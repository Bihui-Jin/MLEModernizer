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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.871992644695206

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

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn import model_selection


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def auto_select_accelerator():
    try:
        if "TPU_NAME" in os.environ or "COLAB_TPU_ADDR" in os.environ:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Running on TPU:", tpu.master())
        else:
            strategy = tf.distribute.get_strategy()
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


strategy = auto_select_accelerator()


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _resolve_path(*candidates):
    """Return first existing path among candidates (keeps original default paths but makes run robust)."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]  # fallback to first


config = _Obj(
    base=_Obj(
        seed=2048,
        train_path="../input/plant-pathology-2020-fgvc7/train.csv",
        test_path="../input/plant-pathology-2020-fgvc7/test.csv",
        ss_path="../input/plant-pathology-2020-fgvc7/sample_submission.csv",
        img_dir="../input/plant-pathology-2020-fgvc7/images",
        print_freq=100,
        num_workers=4,
        target_size=4,
        target_cols=None,  # set after reading sample_submission
        n_fold=4,
        trn_fold=[0],
        train=True,
        debug=False,
        oof=False,
    ),
    dataset=_Obj(
        augment=True,
        cache=True,
        repeat=True,
        shuffle=1024,
        cache_dir="./tfdata_cache",
        snapshot=False,  # keep off: snapshot can add overhead / buffering
    ),
    split=_Obj(
        name="KFold",
        param={"n_splits": 4, "shuffle": True, "random_state": 0},
    ),
    model=_Obj(
        model_name="EfficientNetB0",
        size=224,
        batch_size=128,
        pretrained=True,
        epochs=30,
        in_features=2048,
    ),
    loss=_Obj(name="binary_crossentropy", param={}),
    optimizer=_Obj(name="Adam", param={"learning_rate": 1e-4}),
    scheduler=_Obj(
        name="CosineAnnealingLR",
        param={"epochs_per_cycle": 5, "lr_max": 5e-3, "lr_min": 1e-4},
    ),
)

seed_everything(config.base.seed)

config.base.train_path = _resolve_path(
    config.base.train_path,
    "/kaggle/input/plant-pathology-2020-fgvc7/train.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/train.csv",
    "/kaggle/data/train.csv",
)
config.base.test_path = _resolve_path(
    config.base.test_path,
    "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
    "/kaggle/data/test.csv",
)
config.base.ss_path = _resolve_path(
    config.base.ss_path,
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
    "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)
config.base.img_dir = _resolve_path(
    config.base.img_dir,
    "/kaggle/input/plant-pathology-2020-fgvc7/images",
    "/kaggle/data/plant-pathology-2020-fgvc7/images",
    "/kaggle/data/images",
)

train = pd.read_csv(config.base.train_path)
test = pd.read_csv(config.base.test_path)
sub = pd.read_csv(config.base.ss_path)

target_cols = [c for c in sub.columns if c != "image_id"]
config.base.target_cols = target_cols
config.base.target_size = len(target_cols)

assert (
    list(sub.columns) == ["image_id"] + target_cols
), f"Unexpected submission columns: {sub.columns.tolist()}"
assert set(target_cols).issubset(train.columns), "Train missing target columns"
print("Loaded:", train.shape, test.shape, sub.shape)
print("Targets:", target_cols)
print("Image dir:", config.base.img_dir)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.AUTOTUNE
os.makedirs(config.dataset.cache_dir, exist_ok=True)

IMG_SIZE = int(config.model.size)


@tf.function
def decode_image_from_path_fixed(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def make_paths_tensor(image_ids, img_dir):
    ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    return tf.strings.join(
        [
            tf.constant(img_dir + "/", dtype=tf.string),
            ids,
            tf.constant(".jpg", dtype=tf.string),
        ]
    )


def _dataset_options(cfg):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = max(1, int(cfg.base.num_workers))
    options.threading.max_intra_op_parallelism = 0
    return options


def build_image_dataset(cfg, image_ids, labels=None, val=False, cache_key: str = ""):
    """
    Fix: tf.data.experimental.map_and_batch no longer supports `deterministic` kwarg in some TF builds.
    Keep core semantics by using .map(..., deterministic=...) then .batch(..., drop_remainder=...).
    """
    paths = make_paths_tensor(image_ids, cfg.base.img_dir)

    if labels is not None:
        y = tf.convert_to_tensor(labels.values, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.with_options(_dataset_options(cfg))

    if labels is not None:

        def _load_xy(path, label):
            return decode_image_from_path_fixed(path), label

        if (not val) and cfg.dataset.shuffle:
            ds = ds.shuffle(
                int(cfg.dataset.shuffle),
                seed=cfg.base.seed,
                reshuffle_each_iteration=True,
            )
        if (not val) and cfg.dataset.repeat:
            ds = ds.repeat()

        if cfg.dataset.cache:
            if cache_key:
                ds = ds.cache(os.path.join(cfg.dataset.cache_dir, f"{cache_key}.cache"))
            else:
                ds = ds.cache()

        if cfg.dataset.augment and (not val):
            aug = build_augmenter(with_labels=True)
            ds = ds.map(aug, num_parallel_calls=AUTO, deterministic=True)

        drop = not val
        ds = ds.map(_load_xy, num_parallel_calls=AUTO, deterministic=True)
        ds = ds.batch(cfg.model.batch_size, drop_remainder=drop)
    else:

        def _load_x(path):
            return decode_image_from_path_fixed(path)

        if cfg.dataset.cache:
            if cache_key:
                ds = ds.cache(os.path.join(cfg.dataset.cache_dir, f"{cache_key}.cache"))
            else:
                ds = ds.cache()

        ds = ds.map(_load_x, num_parallel_calls=AUTO, deterministic=True)
        ds = ds.batch(cfg.model.batch_size, drop_remainder=False)

    ds = ds.prefetch(AUTO)
    return ds




## === cell 2
class CosineAnnealingScheduler(callbacks.LearningRateScheduler):
    def __init__(self, epochs_per_cycle, lr_min, lr_max, verbose=0):
        super().__init__(self.schedule, verbose=verbose)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.epochs_per_cycle = int(epochs_per_cycle)

    def schedule(self, epoch, lr):
        return (
            self.lr_min
            + (self.lr_max - self.lr_min)
            * (
                1
                + math.cos(
                    math.pi * (epoch % self.epochs_per_cycle) / self.epochs_per_cycle
                )
            )
            / 2
        )


__SCHEDULERS__ = {"CosineAnnealingLR": CosineAnnealingScheduler}


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return getattr(model_selection, cfg.split.name)(**cfg.split.param)
    raise NotImplementedError(cfg.split.name)


def get_optimizer(cfg):
    if hasattr(tf.keras.optimizers, cfg.optimizer.name):
        return getattr(tf.keras.optimizers, cfg.optimizer.name)(**cfg.optimizer.param)
    raise NotImplementedError(cfg.optimizer.name)


def get_scheduler(cfg):
    if cfg.scheduler.name in __SCHEDULERS__:
        return __SCHEDULERS__[cfg.scheduler.name](**cfg.scheduler.param)
    if hasattr(tf.keras.optimizers.schedules, cfg.scheduler.name):
        return getattr(tf.keras.optimizers.schedules, cfg.scheduler.name)(
            **cfg.scheduler.param
        )
    raise NotImplementedError(cfg.scheduler.name)


def get_backbone(cfg):
    if cfg.model.model_name == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0(
            input_shape=(cfg.model.size, cfg.model.size, 3),
            weights="imagenet" if cfg.model.pretrained else None,
            include_top=False,
        )
    raise NotImplementedError(cfg.model.model_name)




## === cell 3
def train_loop(cfg, folds, fold, test_dataset):
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_ids = train_folds["image_id"].values
    valid_ids = valid_folds["image_id"].values

    train_dataset = build_image_dataset(
        cfg,
        image_ids=train_ids,
        labels=train_folds[cfg.base.target_cols],
        val=False,
        cache_key=f"fold{fold}_train_{cfg.model.size}",
    )
    valid_dataset = build_image_dataset(
        cfg,
        image_ids=valid_ids,
        labels=valid_folds[cfg.base.target_cols],
        val=True,
        cache_key=f"fold{fold}_val_{cfg.model.size}",
    )

    steps_per_epoch = max(1, train_folds.shape[0] // cfg.model.batch_size)
    val_steps = max(1, math.ceil(valid_folds.shape[0] / cfg.model.batch_size))

    with strategy.scope():
        model = tf.keras.Sequential(
            [
                get_backbone(cfg),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(cfg.base.target_size, activation="sigmoid"),
            ]
        )

        spe = max(1, min(32, steps_per_epoch))

        model.compile(
            optimizer=get_optimizer(cfg),
            loss=cfg.loss.name,
            metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
            steps_per_execution=spe,
            jit_compile=False,
        )

    weights_path = f"model_fold{fold}.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weights_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_auc",
        mode="max",
    )
    lr_reducer = get_scheduler(cfg)

    model.fit(
        train_dataset,
        epochs=cfg.model.epochs,
        verbose=2,
        callbacks=[checkpoint, lr_reducer],
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=val_steps,
    )

    model.load_weights(weights_path)

    pred = model.predict(test_dataset, verbose=1)
    return pred


def main(cfg):
    seed_everything(seed=cfg.base.seed)

    folds = train.copy()
    if cfg.base.debug:
        folds = folds.sample(n=100, random_state=cfg.base.seed).reset_index(drop=True)
        cfg.model.epochs = 1

    Fold = get_split(cfg)
    folds["fold"] = -1
    for n, (train_index, val_index) in enumerate(Fold.split(folds)):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)

    test_dataset = build_image_dataset(
        cfg,
        image_ids=test["image_id"].values,
        labels=None,
        val=True,
        cache_key=f"test_{cfg.model.size}",
    )

    sub_pred = np.zeros((len(test), cfg.base.target_size), dtype=np.float32)

    used_folds = 0
    for fold in range(cfg.base.n_fold):
        if fold in cfg.base.trn_fold:
            fold_pred = train_loop(cfg, folds, fold, test_dataset=test_dataset)
            sub_pred += fold_pred
            used_folds += 1

    if used_folds > 0:
        sub_pred /= used_folds

    submission = test[["image_id"]].copy()
    for i, c in enumerate(cfg.base.target_cols):
        submission[c] = sub_pred[:, i].astype(np.float32)

    submission = submission[["image_id"] + cfg.base.target_cols]

    submission.to_csv("submit.csv", index=False)
    print("Wrote submit.csv with shape:", submission.shape)
    print(submission.head())




## === cell 4
main(config)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1017567265.py in <cell line: 0>()
----> 1 main(config)

/tmp/ipykernel_11/3359969536.py in main(cfg)
     99     for fold in range(cfg.base.n_fold):
    100         if fold in cfg.base.trn_fold:
--> 101             fold_pred = train_loop(cfg, folds, fold, test_dataset=test_dataset)
    102             sub_pred += fold_pred
    103             used_folds += 1

/tmp/ipykernel_11/3359969536.py in train_loop(cfg, folds, fold, test_dataset)
      9     valid_ids = valid_folds["image_id"].values
     10 
---> 11     train_dataset = build_image_dataset(
     12         cfg,
     13         image_ids=train_ids,

/tmp/ipykernel_11/92281698.py in build_image_dataset(cfg, image_ids, labels, val, cache_key)
     93         if cfg.dataset.augment and (not val):
     94             aug = build_augmenter(with_labels=True)
---> 95             ds = ds.map(aug, num_parallel_calls=AUTO, deterministic=True)
     96 
     97         drop = not val

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

/tmp/__autograph_generated_filekb1j9erv.py in tf__augment_with_labels(img, label)
     11                 try:
     12                     do_return = True
---> 13                     retval_ = (ag__.converted_call(ag__.ld(augment), (ag__.ld(img),), None, fscope), ag__.ld(label))
     14                 except:
     15                     do_return = False

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

/tmp/__autograph_generated_filew9ogslwl.py in tf__augment(img)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_up_down, (ag__.ld(img),), None, fscope)
     12                 try:

ValueError: in user code:

    File "/tmp/ipykernel_11/92281698.py", line 30, in augment_with_labels  *
        return augment(img), label
    File "/tmp/ipykernel_11/92281698.py", line 25, in augment  *
        img = tf.image.random_flip_left_right(img)

    ValueError: 'image' (shape ()) must be at least three-dimensional.
