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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.96313

# 6. Current score

0.03382

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18332) has done: 'The fixes address the import error, incorrect ImageDataGenerator arguments, wrong file paths, misuse of TensorFlow Hub, and the loss of file‑paths after dataset mapping. Paths are switched to the Kaggle `/kaggle/input` location, the EfficientNet B4 model is built directly from `tf.keras.applications`, and the test dataset keeps its original file list for submission. All cells now run sequentially and produce a valid `model_submission_v17.csv` file.'
- What this solution (achieved 0.08724) has done: 'I replace the slower `ImageDataGenerator` pipeline with a `tf.data` pipeline that uses the same augmentation functions via `tf.numpy_function`. This removes the Python‑level per‑image loop while keeping identical augmentations, class ordering, and model architecture. I also adjust batch‑shape printing, visualization, and the TTA dataset to use the new pipeline, and I create a `class_indices` mapping from the sorted class list for the inverse label conversion.'
- What this solution (achieved 0.13874) has done: 'The fix updates the data pipelines so that augmentations are applied per‑image (not on whole batches), restores known tensor shapes for TensorFlow, corrects the TTA dataset creation, and ensures the model can be trained and a proper CSV submission is written.'
- What this solution (achieved 0.03382) has done: 'The updates fix the protobuf import issue, correct the data‑augmentation pipeline’s datatype expectations, remove the illegal `class_names` argument when loading unlabeled images, and use the original validation file paths for evaluation. We also increase the image size to 224 × 224 to boost model accuracy while keeping the overall architecture unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import cv2
import albumentations as A
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

train_meta_data = "/kaggle/input/paddy-disease-classification/train.csv"
train_data_dir = "/kaggle/input/paddy-disease-classification/train_images"
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images"

train_df = pd.read_csv(train_meta_data)
class_names = sorted(train_df["label"].unique())
class_indices = {name: idx for idx, name in enumerate(class_names)}

epochs = 30
lr = 1e-4
valid_split = 0.2
input_size = 224  # increased size for better accuracy
batch_size = 16
classes = len(class_names)
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()
initializer = tf.keras.initializers.HeUniform()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)




## === cell 2
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x, anchors_y = [], []
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_y:
                anchors_y.append(rv)
        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0
    return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    return image


def random_displacement(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        slices = np.split(image, 8, axis=ax)
        np.random.shuffle(slices)
        return np.row_stack(slices) if ax == 0 else np.column_stack(slices)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 8, 16)
    image = random_displacement(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    image = tf.image.random_saturation(image, 0.75, 1.25).numpy()
    image = tf.image.random_hue(image, 0.1).numpy()
    return image


def test_time_augmentation_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    return image




## === cell 3
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="training",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=1,
    label_mode="categorical",
    class_names=class_names,
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="validation",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=1,
    label_mode="categorical",
    class_names=class_names,
)


def _augment(image, label):
    aug = tf.numpy_function(
        func=center_crop_and_random_augmentations_fn,
        inp=[image],
        Tout=tf.float32,  # corrected output dtype
    )
    aug.set_shape([input_size, input_size, 3])
    aug = tf.cast(aug, tf.float32) / 255.0
    return aug, label


def _augment_no_label(image):
    aug = tf.numpy_function(
        func=center_crop_and_random_augmentations_fn,
        inp=[image],
        Tout=tf.float32,  # corrected output dtype
    )
    aug.set_shape([input_size, input_size, 3])
    aug = tf.cast(aug, tf.float32) / 255.0
    return aug


train_ds = raw_train_ds.unbatch().map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = raw_val_ds.unbatch().map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



## === cell 4
train_batch = next(iter(train_ds))[0]
val_batch = next(iter(val_ds))[0]
print(
    "Train batch shape:", train_batch.shape, "Validation batch shape:", val_batch.shape
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_55/2207526877.py in <cell line: 0>()
----> 1 train_batch = next(iter(train_ds))[0]
      2 val_batch = next(iter(val_ds))[0]
      3 print(
      4     "Train batch shape:", train_batch.shape, "Validation batch shape:", val_batch.shape
      5 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 1452532269 [Op:Slice] name: 
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/2257327884.py", line 33, in center_crop_and_random_augmentations_fn
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 6002, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 1452532269 [Op:Slice] name: 


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 5
base_model = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(input_size, input_size, 3),
    pooling="avg",
)

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Dense(classes, activation="softmax"),
    ]
)



## === cell 6
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



## === cell 7
model.summary()



## === cell 8
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_55/19020920.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     validation_data=val_ds,
      4     epochs=epochs,
      5     callbacks=[early_stop, reduce_lr],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) UNKNOWN:  InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 468207987 [Op:Slice] name: 
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/2257327884.py", line 33, in center_crop_and_random_augmentations_fn
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 6002, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 468207987 [Op:Slice] name: 


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_4]]
  (1) UNKNOWN:  InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 468207987 [Op:Slice] name: 
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/2257327884.py", line 33, in center_crop_and_random_augmentations_fn
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 6002, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 468207987 [Op:Slice] name: 


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_137028]

## === cell 9
tta_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=valid_split,
    subset="validation",
    seed=42,
    image_size=(input_size, input_size),
    batch_size=1,
    label_mode=None,
)

tta_dataset = (
    tta_dataset.unbatch()
    .map(_augment_no_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 10
eve_encodings = np.zeros((len(raw_val_ds.file_paths), classes))
for _ in range(5):
    enc = model.predict(tta_dataset, verbose=0)
    eve_encodings += enc
eve_encodings /= 5



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_55/39333805.py in <cell line: 0>()
      1 eve_encodings = np.zeros((len(raw_val_ds.file_paths), classes))
      2 for _ in range(5):
----> 3     enc = model.predict(tta_dataset, verbose=0)
      4     eve_encodings += enc
      5 eve_encodings /= 5

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

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 348749370 [Op:Slice] name: 
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_55/2257327884.py", line 33, in center_crop_and_random_augmentations_fn
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 6002, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Slice_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected begin[0] in [0, 224], but got 348749370 [Op:Slice] name: 


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 11
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = np.array(
    [
        class_indices[os.path.basename(p).split(os.sep)[-2]]
        for p in raw_val_ds.file_paths
    ]
)
print("Validation accuracy:", accuracy_score(true_classes, pred_classes))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/895712509.py in <cell line: 0>()
      1 pred_classes = np.argmax(eve_encodings, axis=1)
      2 true_classes = np.array(
----> 3     [
      4         class_indices[os.path.basename(p).split(os.sep)[-2]]
      5         for p in raw_val_ds.file_paths

/tmp/ipykernel_55/895712509.py in <listcomp>(.0)
      2 true_classes = np.array(
      3     [
----> 4         class_indices[os.path.basename(p).split(os.sep)[-2]]
      5         for p in raw_val_ds.file_paths
      6     ]

IndexError: list index out of range

## === cell 12
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1781478178.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 13
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3700018787.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
      4 )
      5 sns.lineplot(

NameError: name 'history' is not defined

## === cell 14
pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/820322879.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)
      2 

NameError: name 'history' is not defined

## === cell 15
model.save("model_effnet_b4.hdf5")
model.save_weights("model_effnet_b4.weights.h5")



## === cell 16
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels=None,
    image_size=(input_size, input_size),
    batch_size=1,
    shuffle=False,
)
test_file_paths = test_dataset.file_paths




## === cell 17
def normalize_batch(batch):
    batch = tf.cast(batch, tf.float32) / 255.0
    return batch


test_dataset = test_dataset.map(lambda x: normalize_batch(x))



## === cell 18
test_encodings = np.zeros((len(test_file_paths), classes))
for _ in range(5):
    enc = model.predict(test_dataset, verbose=0)
    test_encodings += enc
test_encodings /= 5



## === cell 19
predict_max = np.argmax(test_encodings, axis=1)



## === cell 20
inverse_map = {v: k for k, v in class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]



## === cell 21
files = [os.path.basename(p) for p in test_file_paths]
submission = pd.DataFrame({"image_id": files, "label": predictions})
submission.to_csv("model_submission_v17.csv", index=False)
print(submission.head())
