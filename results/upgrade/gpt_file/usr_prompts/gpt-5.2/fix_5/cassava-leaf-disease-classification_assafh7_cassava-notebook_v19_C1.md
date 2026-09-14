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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
seaborn==0.12.2
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.0018

# 6. Current score

0.2201

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.2201) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus long training (up to 40 epochs) on a heavy backbone. To keep the exact same model and training semantics while reducing wall time, I (1) switch to Keras’ `tf.data`-based `image_dataset_from_directory` with the same 80/20 split and the same augmentation operations implemented as TensorFlow graph ops (so the augmentation *logic* is unchanged), (2) add caching/prefetching and parallel decoding, and (3) avoid any extra one-off batches/plots that trigger I/O. This keeps the architecture, loss, optimizer, early stopping, epochs, and evaluation logic intact, but removes the major input pipeline bottleneck that commonly causes >10 minute runs.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import warnings

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import albumentations as aug

warnings.simplefilter("ignore")

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

base_walk = "/kaggle/input"
for dirname, _, filenames in os.walk(base_walk):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    break

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
general_path = "../input/cassava-leaf-disease-classification/"
print("Exists:", os.path.exists(general_path))
print(os.listdir(general_path)[:10])




## === cell 2
with open(os.path.join(general_path, "label_num_to_disease_map.json"), "r") as file:
    map_classes = json.loads(file.read())
map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))




## === cell 3
input_files = os.listdir(os.path.join(general_path, "train_images"))
print(f"Number of train images: {len(input_files)}")




## === cell 4
img_shapes = {}
print("Skipping image shape scan for performance (not used downstream).")




## === cell 5
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()




## === cell 6
print("Skipping class distribution plot for performance (not used downstream).")




## === cell 7
def visualize_batch(image_ids, labels, class_names):
    plt.figure(figsize=(16, 12))
    for ind, (image_id, label, class_name) in enumerate(
        zip(image_ids, labels, class_names)
    ):
        if ind >= 9:
            break
        plt.subplot(3, 3, ind + 1)
        img = cv2.imread(os.path.join(general_path, "train_images", image_id))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Class {label}:  {class_name}", fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 8
print("Skipping class-0 sample visualization for performance.")




## === cell 9
print("Skipping class-1 sample visualization for performance.")




## === cell 10
print("Skipping class-2 sample visualization for performance.")




## === cell 11
print("Skipping class-3 sample visualization for performance.")




## === cell 12
print("Skipping class-4 sample visualization for performance.")




## === cell 13
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(general_path, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("original")

    for i in range(2, 5):
        plt.subplot(2, 2, i)
        x = transform(image=img)["image"]
        plt.imshow(x)
        plt.axis("off")
        plt.title(f"Augmentation -{i-1}")

    plt.show()




## === cell 14
transform_shift_scale_rotate = aug.ShiftScaleRotate(
    p=1.0,
    shift_limit=(-0.3, 0.3),
    scale_limit=(-0.1, 0.1),
    rotate_limit=(-180, 180),
    interpolation=0,
    border_mode=4,
)
print("Skipping augmentation demo (ShiftScaleRotate).")




## === cell 15
transform_coarse_dropout = aug.CoarseDropout(
    p=1.0,
    max_holes=100,
    max_height=50,
    max_width=50,
    min_holes=30,
    min_height=20,
    min_width=20,
)
print("Skipping augmentation demo (CoarseDropout).")




## === cell 16
transform_hsv = aug.HueSaturationValue(
    hue_shift_limit=0,
    sat_shift_limit=(40, 80),
    val_shift_limit=(40, 80),
    p=1.0,
)
print("Skipping augmentation demo (HueSaturationValue).")




## === cell 17
transform_clahe = aug.CLAHE(
    p=1.0,
    clip_limit=(10, 30),
    tile_grid_size=(10, 10),
)
print("Skipping augmentation demo (CLAHE).")




## === cell 18
transform_fog = aug.RandomFog(p=1.0)
print("Skipping augmentation demo (RandomFog).")




## === cell 19
transform_sunflare = aug.RandomSunFlare(p=1.0)
print("Skipping augmentation demo (RandomSunFlare).")




## === cell 20
transform_brightness_contrast = aug.RandomBrightnessContrast(p=1.0)
print("Skipping augmentation demo (RandomBrightnessContrast).")




## === cell 21
transform_randomcrop = aug.RandomCrop(p=1.0, height=512, width=512)
print("Skipping augmentation demo (RandomCrop).")




## === cell 22
transform_rgbshift = aug.RGBShift(p=1.0)
print("Skipping augmentation demo (RGBShift).")




## === cell 23
transform_snow = aug.RandomSnow(p=1.0)
print("Skipping augmentation demo (RandomSnow).")




## === cell 24
transform_hflip = aug.HorizontalFlip(p=1.0)
print("Skipping augmentation demo (HorizontalFlip).")




## === cell 25
transform_vflip = aug.VerticalFlip(p=1.0)
print("Skipping augmentation demo (VerticalFlip).")




## === cell 26
transform_transpose = aug.Transpose(p=1.0)
print("Skipping augmentation demo (Transpose).")




## === cell 27
img_width, img_height = 224, 224

train = pd.read_csv(os.path.join(general_path, "train.csv"))
train["label"] = train["label"].astype("string")
train.head()




## === cell 28
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
VAL_SPLIT = 0.2

train_images_dir = os.path.join(general_path, "train_images")

work_root = "/kaggle/working"
symlink_root = os.path.join(work_root, "train_images_by_label")
os.makedirs(symlink_root, exist_ok=True)

for lbl in map(str, range(5)):
    os.makedirs(os.path.join(symlink_root, lbl), exist_ok=True)

for image_id, lbl in zip(train["image_id"].values, train["label"].values):
    src = os.path.join(train_images_dir, image_id)
    dst = os.path.join(symlink_root, lbl, image_id)
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

train_ds = tf.keras.utils.image_dataset_from_directory(
    symlink_root,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(img_width, img_height),
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    symlink_root,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(img_width, img_height),
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2


def _augment(images, labels):
    images = tf.cast(images, tf.float32)

    images = tf.image.random_flip_left_right(images, seed=SEED)
    images = tf.image.random_flip_up_down(images, seed=SEED)

    batch_size = tf.shape(images)[0]
    zoom = tf.random.stateless_uniform(
        [batch_size, 1, 1, 1],
        seed=tf.stack([SEED, 123]),
        minval=1.0 - ZOOM_RANGE,
        maxval=1.0 + ZOOM_RANGE,
        dtype=tf.float32,
    )
    in_h = tf.cast(img_height, tf.float32)
    in_w = tf.cast(img_width, tf.float32)
    new_h = tf.cast(tf.round(in_h / zoom[:, 0, 0, 0]), tf.int32)
    new_w = tf.cast(tf.round(in_w / zoom[:, 0, 0, 0]), tf.int32)

    def _zoom_one(img, nh, nw):
        nh = tf.clip_by_value(nh, 1, img_height)
        nw = tf.clip_by_value(nw, 1, img_width)
        img2 = tf.image.resize_with_crop_or_pad(img, nh, nw)
        img2 = tf.image.resize(img2, [img_height, img_width], method="bilinear")
        return img2

    images = tf.map_fn(
        lambda t: _zoom_one(t[0], t[1], t[2]),
        (images, new_h, new_w),
        fn_output_signature=tf.float32,
        parallel_iterations=AUTOTUNE,
    )

    shear = tf.random.stateless_uniform(
        [batch_size],
        seed=tf.stack([SEED, 456]),
        minval=-SHEAR_RANGE,
        maxval=SHEAR_RANGE,
        dtype=tf.float32,
    )
    a0 = tf.ones_like(shear)
    a1 = -shear
    a2 = tf.zeros_like(shear)
    b0 = tf.zeros_like(shear)
    b1 = tf.ones_like(shear)
    b2 = tf.zeros_like(shear)
    c0 = tf.zeros_like(shear)
    c1 = tf.zeros_like(shear)
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1], axis=1)

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.constant([img_height, img_width], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return images, labels


train_ds = (
    train_ds.map(_augment, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
)
valid_ds = valid_ds.cache().prefetch(AUTOTUNE)

train_datagen_flow = train_ds
valid_datagen_flow = valid_ds




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/955660210.py in <cell line: 0>()
    134 
    135 train_ds = (
--> 136     train_ds.map(_augment, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    137 )
    138 valid_ds = valid_ds.cache().prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filekn00s8pe.py in tf___augment(images, labels)
     34                             raise
     35                         return fscope_1.ret(retval__1, do_return_1)
---> 36                 images = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.autograph_artifact(lambda t: ag__.converted_call(ag__.ld(_zoom_one), (ag__.ld(t)[0], ag__.ld(t)[1], ag__.ld(t)[2]), None, fscope)), (ag__.ld(images), ag__.ld(new_h), ag__.ld(new_w))), dict(fn_output_signature=ag__.ld(tf).float32, parallel_iterations=ag__.ld(AUTOTUNE)), fscope)
     37                 shear = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.ld(batch_size)],), dict(seed=ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(SEED), 456],), None, fscope), minval=-ag__.ld(SHEAR_RANGE), maxval=ag__.ld(SHEAR_RANGE), dtype=ag__.ld(tf).float32), fscope)
     38                 a0 = ag__.converted_call(ag__.ld(tf).ones_like, (ag__.ld(shear),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    430     raise TypeError("'body' must be callable.")
    431   if parallel_iterations < 1:
--> 432     raise TypeError("'parallel_iterations' must be a positive integer.")
    433 
    434   loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

TypeError: in user code:

    File "/tmp/ipykernel_11/955660210.py", line 94, in _augment  *
        images = tf.map_fn(

    TypeError: 'parallel_iterations' must be a positive integer.


## === cell 29
x, y = next(iter(train_datagen_flow.take(1)))
print("Batch X shape:", x.shape, "Batch y shape:", y.shape)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2679978360.py in <cell line: 0>()
      1 # --- Speed: avoid forcing a full generator/python iterator; just inspect one batch from tf.data.
----> 2 x, y = next(iter(train_datagen_flow.take(1)))
      3 print("Batch X shape:", x.shape, "Batch y shape:", y.shape)
      4 
      5 

NameError: name 'train_datagen_flow' is not defined

## === cell 30
from tensorflow.keras.applications import EfficientNetB0

backbone = EfficientNetB0(
    weights="imagenet",
    input_shape=(img_width, img_height, 3),
    pooling="avg",
    include_top=False,
)
backbone.summary()




## === cell 31
opt = Adam(learning_rate=5e-4)

model = Sequential()
model.add(backbone)
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()




## === cell 32
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)




## === cell 33
history = model.fit(
    train_datagen_flow,
    validation_data=valid_datagen_flow,
    epochs=40,
    verbose=1,
    callbacks=[early_stop],
)




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/987587366.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_datagen_flow,
      3     validation_data=valid_datagen_flow,
      4     epochs=40,
      5     verbose=1,

NameError: name 'train_datagen_flow' is not defined

## === cell 34
model.save("submission.h5")




## === cell 35
print("Skipping training curves plot for performance.")




## === cell 36
ss = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
test_images_dir = os.path.join(general_path, "test_images")

test_paths = tf.constant(
    [os.path.join(test_images_dir, x) for x in ss["image_id"].values]
)


def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_width, img_height], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_load_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

pred_proba = model.predict(
    test_ds,
    verbose=1,
)
preds = np.argmax(pred_proba, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))
print("submission.csv exists:", os.path.exists("submission.csv"))
