# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
        snapshot=False,  # --- FIX: snapshot can trigger large buffering / ResourceExhausted
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



## === cell 1
AUTO = tf.data.AUTOTUNE

os.makedirs(config.dataset.cache_dir, exist_ok=True)


@tf.function
def decode_image_from_path(path, h=224, w=224):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    img = tf.image.resize(
        img,
        [h, w],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.ensure_shape(img, [h, w, 3])
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


def build_image_dataset(cfg, image_ids, labels=None, val=False, cache_key: str = ""):
    paths = make_paths_tensor(image_ids, cfg.base.img_dir)

    if labels is not None:
        y = tf.convert_to_tensor(labels.values, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = max(1, int(cfg.base.num_workers))
    options.threading.max_intra_op_parallelism = 0
    ds = ds.with_options(options)

    def _load(path):
        return decode_image_from_path(path, h=cfg.model.size, w=cfg.model.size)

    if labels is not None:

        def _load_xy(path, label):
            return _load(path), label

        ds = ds.map(_load_xy, num_parallel_calls=AUTO, deterministic=True)
    else:
        ds = ds.map(_load, num_parallel_calls=AUTO, deterministic=True)

    if cfg.dataset.cache:
        if cache_key:
            cache_path = os.path.join(cfg.dataset.cache_dir, f"{cache_key}.cache")
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()

    if getattr(cfg.dataset, "snapshot", False) and cache_key:
        snap_dir = os.path.join(cfg.dataset.cache_dir, f"snapshot_{cache_key}")
        ds = ds.apply(tf.data.experimental.snapshot(snap_dir))

    if cfg.dataset.augment and (not val):
        aug = build_augmenter(with_labels=(labels is not None))
        ds = ds.map(aug, num_parallel_calls=AUTO, deterministic=True)

    if not val:
        if cfg.dataset.shuffle:
            ds = ds.shuffle(
                int(cfg.dataset.shuffle),
                seed=cfg.base.seed,
                reshuffle_each_iteration=True,
            )
        if cfg.dataset.repeat:
            ds = ds.repeat()

    drop = (not val) and (labels is not None)
    ds = ds.batch(cfg.model.batch_size, drop_remainder=drop)
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

        spe = 1

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
