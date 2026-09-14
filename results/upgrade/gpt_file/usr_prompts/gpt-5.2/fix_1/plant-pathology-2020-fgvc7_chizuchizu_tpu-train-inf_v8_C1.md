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
!pip install omegaconf
!pip install -q efficientnet
!pip install iterative-stratification

## === cell 1
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/2191101355.py in <cell line: 0>()
      1 from kaggle_secrets import UserSecretsClient
      2 user_secrets = UserSecretsClient()
----> 3 user_credential = user_secrets.get_gcloud_credential()
      4 user_secrets.set_tensorflow_credential(user_credential)

/usr/local/lib/python3.11/dist-packages/kaggle_secrets.py in get_gcloud_credential(self)
     76         """
     77         try:
---> 78             return self.get_secret("__gcloud_sdk_auth__")
     79         except BackendError as backend_error:
     80             message = str(backend_error.args)

/usr/local/lib/python3.11/dist-packages/kaggle_secrets.py in get_secret(self, label)
     62             'Label': label,
     63         }
---> 64         response_json = self.web_client.make_post_request(request_body, self.GET_USER_SECRET_BY_LABEL_ENDPOINT)
     65         if 'secret' not in response_json:
     66             raise BackendError(

/usr/local/lib/python3.11/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     47                 response_json = json.loads(response.read())
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')
     51                 return response_json['result']

BackendError: Unexpected response from the service. Response: {'errors': ['Unauthenticated'], 'error': {'code': 16}, 'wasSuccessful': False}.

## === cell 2
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('1tfrecordapple')
GCS_DS_PATH

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/1284430377.py in <cell line: 0>()
      1 from kaggle_datasets import KaggleDatasets
      2 
----> 3 GCS_DS_PATH = KaggleDatasets().get_gcs_path('1tfrecordapple')
      4 GCS_DS_PATH

/usr/local/lib/python3.11/dist-packages/kaggle_datasets.py in get_gcs_path(self, dataset_dir)
     39             'IntegrationType': integration_type,
     40         }
---> 41         result = self.web_client.make_post_request(data, self.GET_GCS_PATH_ENDPOINT, self.TIMEOUT_SECS)
     42         return result['destinationBucket']

/usr/local/lib/python3.11/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     47                 response_json = json.loads(response.read())
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')
     51                 return response_json['result']

BackendError: Unexpected response from the service. Response: {'errors': ['Unauthenticated'], 'error': {'code': 16}, 'wasSuccessful': False}.

## === cell 3
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks
import math
import tensorflow.keras.layers as L
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import schedules
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection

from omegaconf import OmegaConf



target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

conf = """
base:
  train_path: '../input/plant-pathology-2020-fgvc7/train.csv'
  test_path: "../input/plant-pathology-2020-fgvc7/test.csv"
  ss_path: "../input/plant-pathology-2020-fgvc7/sample_submission.csv"
  train_tf_path: "/train.tfrecord"
  test_tf_path: "/test.tfrecord"
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
  cache: true
  repeat: true
  shuffle: 1024
  cache_dir: ""

split:
  # name: "MultilabelStratifiedKFold"
  name: "MultilabelStratifiedKFold"
  param: {
           "n_splits": 4,
           "shuffle": True,
           "random_state": 0
  }

model:
  model_name: "EfficientNetB0"
  size: 224  # 480
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
           "learning_rate": 1e-4,
           # "weight_decay": 1e-6,
           # "amsgrad": False
  }

scheduler:
  name: "CosineAnnealingLR"
  param: {
            "epochs_per_cycle": 5,
            "lr_max": 5e-3,
            "lr_min": 1e-4,
            # "last_epoch": -1
  }
"""
config = OmegaConf.create(conf)


def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    return strategy


strategy = auto_select_accelerator()


def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


seed = 2048
seed_everything(seed)
print("REPLICAS: ", strategy.num_replicas_in_sync)

train = pd.read_csv(config.base.train_path)
test = pd.read_csv(config.base.test_path)
sub = pd.read_csv(config.base.ss_path)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4

def decode_image(image_data, h=224, w=224):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, [h, w])
    image = tf.reshape(image, [h, w, 3])
    return image


def build_decoder(with_labels=True):
    def read_tfrecord(example):
        if with_labels:
            TFREC_FORMAT = {
                'image': tf.io.FixedLenFeature([], tf.string),
                config.base.target_cols[0]: tf.io.FixedLenFeature([], tf.int64),
                config.base.target_cols[1]: tf.io.FixedLenFeature([], tf.int64),
                config.base.target_cols[2]: tf.io.FixedLenFeature([], tf.int64),
                config.base.target_cols[3]: tf.io.FixedLenFeature([], tf.int64),
                'image_name': tf.io.FixedLenFeature([], tf.string),
            }
        else:
            TFREC_FORMAT = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'image_name': tf.io.FixedLenFeature([], tf.string),
            }
        example = tf.io.parse_single_example(example, TFREC_FORMAT)
        image = decode_image(example['image'])
        if with_labels:
            targets = [example[x] for x in config.base.target_cols]
            return image, targets
        else:
            return image

    return read_tfrecord


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label

    
    

def build_dataset(cfg, paths, idx, labels=None, decode_fn=None, augment_fn=None, val=False):
    if val:
        cfg.dataset.repeat = False
        cfg.dataset.shuffle = False
    else:
        cfg.dataset.repeat = True
        cfg.dataset.shuffle = True

    if cfg.dataset.cache_dir != "" and cfg.dataset.cache_dir is True:
        os.makedirs(cfg.dataset.cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder()

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    idx = tf.constant(idx, dtype=tf.int64)

    def is_index_in(index, rest):
        return tf.math.reduce_any(index == idx)

    def drop_index(index, rest):
        return rest

    AUTO = tf.data.experimental.AUTOTUNE

    dset = tf.data.TFRecordDataset(paths)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO).enumerate()
    dset = dset.filter(is_index_in)
    dset = dset.map(drop_index)

    dset = dset.cache(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if cfg.dataset.augment else dset

    dset = dset.repeat() if cfg.dataset.repeat else dset
    dset = dset.shuffle(cfg.dataset.shuffle) if cfg.dataset.shuffle else dset

    dset = dset.batch(cfg.model.batch_size).prefetch(AUTO)
    return dset


class CosineAnnealingScheduler(callbacks.LearningRateScheduler):
    def __init__(self, epochs_per_cycle, lr_min, lr_max, verbose=0):
        super(callbacks.LearningRateScheduler, self).__init__()
        self.verbose = verbose
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs_per_cycle = epochs_per_cycle

    def schedule(self, epoch, lr):
        return self.lr_min + (self.lr_max - self.lr_min) *\
               (1 + math.cos(math.pi * (epoch % self.epochs_per_cycle) / self.epochs_per_cycle)) / 2

__SPLITS__ = {
    "MultilabelStratifiedKFold": MultilabelStratifiedKFold,
}

__OPTIMIZER__ = {

}

__SCHEDULERS__ = {
    "CosineAnnealingLR": CosineAnnealingScheduler
}


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return model_selection.__getattribute__(cfg.split.name)(**cfg.split.param)
    elif __SPLITS__.get(cfg.split.name) is not None:
        return __SPLITS__[cfg.split.name](**cfg.split.param)
    else:
        raise NotImplementedError


def get_optimizer(cfg):
    if hasattr(tf.keras.optimizers, cfg.optimizer.name):
        return tf.keras.optimizers.__getattribute__(cfg.optimizer.name)(**cfg.optimizer.param)
    elif __OPTIMIZER__.get(cfg.optimizer.name) is not None:
        return __OPTIMIZER__[cfg.optimizer.name](**cfg.optimizer.param)
    else:
        raise NotImplementedError


def get_scheduler(cfg):
    if hasattr(schedules, cfg.scheduler.name):
        return schedules.__getattribute__(cfg.scheduler.name)(**cfg.scheduler.param)
    elif __SCHEDULERS__.get(cfg.scheduler.name) is not None:
        return __SCHEDULERS__[cfg.scheduler.name](**cfg.scheduler.param)
    else:
        raise NotImplementedError


def main(cfg):
    global train
    seed_everything(seed=cfg.base.seed)

    folds = train.copy()

    if cfg.base.debug:
        folds = folds.sample(n=100, random_state=cfg.base.seed).reset_index(drop=True)
        cfg.model.epochs = 1
    Fold = get_split(cfg)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[cfg.base.target_cols])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)

    oof_df = train.copy()


    for fold in range(cfg.base.n_fold):
        if fold in cfg.base.trn_fold:
            fold_pred = train_loop(cfg, folds, fold)
            sub.iloc[:, 1:] += fold_pred / cfg.base.n_fold

    sub.to_csv("submit.csv", index=False)


def train_loop(cfg, folds, fold):
    global rand


    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    decoder = build_decoder()
    train_tf_path = cfg.base.train_tf_path

    train_dataset = build_dataset(
        cfg,
        train_tf_path,
        trn_idx.values,
        decode_fn=decoder,
        labels=train_folds[cfg.base.target_cols],
        val=False
    )

    valid_dataset = build_dataset(
        cfg,
        train_tf_path,
        val_idx.values,
        decode_fn=decoder,
        labels=valid_folds[cfg.base.target_cols],
        val=True
    )

    with strategy.scope():
        model = tf.keras.Sequential([
            efn.__getattribute__(cfg.model.model_name)(
                input_shape=(cfg.model.size, cfg.model.size, 3),
                weights='imagenet',
                include_top=False,
                drop_connect_rate=0.7),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(cfg.base.target_size, activation='sigmoid')
        ])
        model.compile(
            optimizer=get_optimizer(cfg),
            loss=cfg.loss.name,
            metrics=[tf.keras.metrics.AUC(multi_label=True)])
        model.summary()

    steps_per_epoch = train_folds.shape[0] // cfg.model.batch_size
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model.h5', save_best_only=True, monitor='val_auc', mode='max')
    lr_reducer = get_scheduler(cfg)

    history = model.fit(
        train_dataset,
        epochs=cfg.model.epochs,
        verbose=2,
        callbacks=[checkpoint, lr_reducer],
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset
    )


    test_tf_path = cfg.base.test_tf_path
    test_augment = build_augmenter(with_labels=False)
    test_decoder = build_decoder(with_labels=False)
    test_dataset = build_dataset(
        cfg,
        test_tf_path,
        test.index.values,
        decode_fn=test_decoder,
        augment_fn=test_augment,
        val=True
    )

    pred = model.predict(test_dataset, verbose=1)

    return pred




## === cell 5
config.base.train_tf_path = GCS_DS_PATH  + config.base.train_tf_path
config.base.test_tf_path = GCS_DS_PATH + config.base.test_tf_path

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3570517239.py in <cell line: 0>()
----> 1 config.base.train_tf_path = GCS_DS_PATH  + config.base.train_tf_path
      2 config.base.test_tf_path = GCS_DS_PATH + config.base.test_tf_path

NameError: name 'GCS_DS_PATH' is not defined

## === cell 6
config.base.train_tf_path

## === cell 7
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/2191101355.py in <cell line: 0>()
      1 from kaggle_secrets import UserSecretsClient
      2 user_secrets = UserSecretsClient()
----> 3 user_credential = user_secrets.get_gcloud_credential()
      4 user_secrets.set_tensorflow_credential(user_credential)

/usr/local/lib/python3.11/dist-packages/kaggle_secrets.py in get_gcloud_credential(self)
     76         """
     77         try:
---> 78             return self.get_secret("__gcloud_sdk_auth__")
     79         except BackendError as backend_error:
     80             message = str(backend_error.args)

/usr/local/lib/python3.11/dist-packages/kaggle_secrets.py in get_secret(self, label)
     62             'Label': label,
     63         }
---> 64         response_json = self.web_client.make_post_request(request_body, self.GET_USER_SECRET_BY_LABEL_ENDPOINT)
     65         if 'secret' not in response_json:
     66             raise BackendError(

/usr/local/lib/python3.11/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     47                 response_json = json.loads(response.read())
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')
     51                 return response_json['result']

BackendError: Unexpected response from the service. Response: {'errors': ['Unauthenticated'], 'error': {'code': 16}, 'wasSuccessful': False}.

## === cell 8
main(config)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ConfigAttributeError                      Traceback (most recent call last)
/tmp/ipykernel_11/1017567265.py in <cell line: 0>()
----> 1 main(config)

/tmp/ipykernel_11/500349932.py in main(cfg)
    158 def main(cfg):
    159     global train
--> 160     seed_everything(seed=cfg.base.seed)
    161 
    162     folds = train.copy()

/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py in __getattr__(self, key)
    353             )
    354         except ConfigKeyError as e:
--> 355             self._format_and_raise(
    356                 key=key, value=None, cause=e, type_override=ConfigAttributeError
    357             )

/usr/local/lib/python3.11/dist-packages/omegaconf/base.py in _format_and_raise(self, key, value, cause, msg, type_override)
    229         type_override: Any = None,
    230     ) -> None:
--> 231         format_and_raise(
    232             node=self,
    233             key=key,

/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py in format_and_raise(node, key, value, msg, cause, type_override)
    897         ex.ref_type_str = ref_type_str
    898 
--> 899     _raise(ex, cause)
    900 
    901 

/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py in _raise(ex, cause)
    795     else:
    796         ex.__cause__ = None
--> 797     raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
    798 
    799 

/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py in __getattr__(self, key)
    349 
    350         try:
--> 351             return self._get_impl(
    352                 key=key, default_value=_DEFAULT_MARKER_, validate_key=False
    353             )

/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py in _get_impl(self, key, default_value, validate_key)
    440     ) -> Any:
    441         try:
--> 442             node = self._get_child(
    443                 key=key, throw_on_missing_key=True, validate_key=validate_key
    444             )

/usr/local/lib/python3.11/dist-packages/omegaconf/basecontainer.py in _get_child(self, key, validate_access, validate_key, throw_on_missing_value, throw_on_missing_key)
     71     ) -> Union[Optional[Node], List[Optional[Node]]]:
     72         """Like _get_node, passing through to the nearest concrete Node."""
---> 73         child = self._get_node(
     74             key=key,
     75             validate_access=validate_access,

/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py in _get_node(self, key, validate_access, validate_key, throw_on_missing_value, throw_on_missing_key)
    478         if value is None:
    479             if throw_on_missing_key:
--> 480                 raise ConfigKeyError(f"Missing key {key!s}")
    481         elif throw_on_missing_value and value._is_missing():
    482             raise MissingMandatoryValue("Missing mandatory value: $KEY")

ConfigAttributeError: Missing key seed
    full_key: base.seed
    object_type=dict
