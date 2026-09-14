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
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import efficientnet.tfkeras as efn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection
from omegaconf import OmegaConf


def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


strategy = auto_select_accelerator()

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

conf = """
base:
  seed: 2048
  train_path: '../input/plant-pathology-2020-fgvc7/train.csv'
  test_path: "../input/plant-pathology-2020-fgvc7/test.csv"
  ss_path: "../input/plant-pathology-2020-fgvc7/sample_submission.csv"
  img_dir: "../input/plant-pathology-2020-fgvc7/images"
  print_freq: 100
  num_workers: 4
  target_size: 4
  target_cols: ["healthy", "multiple_diseases", "rust", "scab"]
  n_fold: 4
  trn_fold: [0]
  train: True
  debug: False
  oof: False

dataset:
  augment: true
  cache: false
  repeat: true
  shuffle: 1024
  cache_dir: ""

split:
  name: "MultilabelStratifiedKFold"
  param: {
           "n_splits": 4,
           "shuffle": True,
           "random_state": 0
  }

model:
  model_name: "EfficientNetB0"
  size: 224
  batch_size: 128
  pretrained: true
  epochs: 30
  in_features: 2048

loss:
  name: "binary_crossentropy"
  param: {}

optimizer:
  name: "Adam"
  param: {
           "learning_rate": 1e-4
  }

scheduler:
  name: "CosineAnnealingLR"
  param: {
            "epochs_per_cycle": 5,
            "lr_max": 5e-3,
            "lr_min": 1e-4
  }
"""
config = OmegaConf.create(conf)

seed_everything(config.base.seed)

train = pd.read_csv(config.base.train_path)
test = pd.read_csv(config.base.test_path)
sub = pd.read_csv(config.base.ss_path)

assert (
    list(sub.columns) == ["image_id"] + target_cols
), f"Unexpected submission columns: {sub.columns.tolist()}"
assert set(target_cols).issubset(train.columns), "Train missing target columns"
print(train.shape, test.shape, sub.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.AUTOTUNE


def decode_image_from_path(path, h=224, w=224):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # /255
    img = tf.image.resize(img, [h, w])
    img = tf.reshape(img, [h, w, 3])
    return img


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def make_paths(image_ids, img_dir):
    return [os.path.join(img_dir, f"{img_id}.jpg") for img_id in image_ids]


def build_image_dataset(cfg, image_ids, labels=None, val=False):
    paths = tf.constant(make_paths(image_ids, cfg.base.img_dir))

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        return decode_image_from_path(path, h=cfg.model.size, w=cfg.model.size)

    ds = ds.map(_load, num_parallel_calls=AUTO)

    if labels is not None:
        y = tf.constant(labels.values, dtype=tf.float32)
        ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(y)))

    if cfg.dataset.cache:
        ds = ds.cache(cfg.dataset.cache_dir if cfg.dataset.cache_dir else None)

    if cfg.dataset.augment and (not val):
        aug = build_augmenter(with_labels=(labels is not None))
        ds = ds.map(aug, num_parallel_calls=AUTO)

    if not val:
        if cfg.dataset.shuffle:
            ds = ds.shuffle(
                int(cfg.dataset.shuffle),
                seed=cfg.base.seed,
                reshuffle_each_iteration=True,
            )
        if cfg.dataset.repeat:
            ds = ds.repeat()

    ds = ds.batch(cfg.model.batch_size, drop_remainder=False).prefetch(AUTO)
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


__SPLITS__ = {"MultilabelStratifiedKFold": MultilabelStratifiedKFold}
__SCHEDULERS__ = {"CosineAnnealingLR": CosineAnnealingScheduler}


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return getattr(model_selection, cfg.split.name)(**cfg.split.param)
    if cfg.split.name in __SPLITS__:
        return __SPLITS__[cfg.split.name](**cfg.split.param)
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2914003098.py in <cell line: 0>()
     20 
     21 
---> 22 __SPLITS__ = {"MultilabelStratifiedKFold": MultilabelStratifiedKFold}
     23 __SCHEDULERS__ = {"CosineAnnealingLR": CosineAnnealingScheduler}
     24 

NameError: name 'MultilabelStratifiedKFold' is not defined

## === cell 3
def train_loop(cfg, folds, fold):
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = build_image_dataset(
        cfg,
        image_ids=train_folds["image_id"].tolist(),
        labels=train_folds[cfg.base.target_cols],
        val=False,
    )
    valid_dataset = build_image_dataset(
        cfg,
        image_ids=valid_folds["image_id"].tolist(),
        labels=valid_folds[cfg.base.target_cols],
        val=True,
    )

    with strategy.scope():
        model = tf.keras.Sequential(
            [
                getattr(efn, cfg.model.model_name)(
                    input_shape=(cfg.model.size, cfg.model.size, 3),
                    weights="imagenet" if cfg.model.pretrained else None,
                    include_top=False,
                    drop_connect_rate=0.7,
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(cfg.base.target_size, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=get_optimizer(cfg),
            loss=cfg.loss.name,
            metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
        )

    steps_per_epoch = max(1, train_folds.shape[0] // cfg.model.batch_size)
    val_steps = max(1, math.ceil(valid_folds.shape[0] / cfg.model.batch_size))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"model_fold{fold}.keras", save_best_only=True, monitor="val_auc", mode="max"
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

    model = tf.keras.models.load_model(f"model_fold{fold}.keras")

    test_dataset = build_image_dataset(
        cfg,
        image_ids=test["image_id"].tolist(),
        labels=None,
        val=True,
    )
    pred = model.predict(test_dataset, verbose=1)
    return pred


def main(cfg):
    seed_everything(seed=cfg.base.seed)

    folds = train.copy()
    if cfg.base.debug:
        folds = folds.sample(n=100, random_state=cfg.base.seed).reset_index(drop=True)
        cfg.model.epochs = 1

    Fold = get_split(cfg)
    for n, (train_index, val_index) in enumerate(
        Fold.split(folds, folds[cfg.base.target_cols])
    ):
        folds.loc[val_index, "fold"] = int(n)
    folds["fold"] = folds["fold"].astype(int)

    sub_pred = np.zeros((len(test), cfg.base.target_size), dtype=np.float32)

    used_folds = 0
    for fold in range(cfg.base.n_fold):
        if fold in cfg.base.trn_fold:
            fold_pred = train_loop(cfg, folds, fold)
            sub_pred += fold_pred
            used_folds += 1

    if used_folds > 0:
        sub_pred /= used_folds

    submission = sub.copy()
    submission[cfg.base.target_cols] = sub_pred
    submission.to_csv("submit.csv", index=False)
    print("Wrote submit.csv with shape:", submission.shape)
    print(submission.head())




## === cell 4
main(config)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1017567265.py in <cell line: 0>()
----> 1 main(config)

NameError: name 'config' is not defined
