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

3.12

# 3. Installed packages

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

0.8482

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.72197) has done: 'I fix the environment-breaking protobuf/TF import error by switching to the supported `tf_keras` package that matches TensorFlow 2.18 on Kaggle, while keeping your model and training logic the same. Then I remove the now-invalid `workers/use_multiprocessing/max_queue_size` arguments from `model.fit()` and `model.predict()` because Keras 3’s TensorFlow trainer no longer accepts them (that’s what triggered your runtime errors). Finally, I ensure the submission is written as `submission.csv` with the exact required columns and row alignment with `sample_submission.csv`. These changes are execution/stability fixes and should also allow you to actually obtain a score (and likely a reasonable baseline near your target) without altering the core modeling approach.'
- What this solution (achieved 0.75374) has done: 'I fix the environment-breaking protobuf/TensorFlow import error by pinning protobuf to the TF 2.18-compatible major version at runtime before importing TensorFlow. Next, I resolve the `UnimplementedError` caused by enabling deterministic ops (which breaks fused batch-norm backprop during fine-tuning) by disabling op determinism so training can proceed normally. I keep your model, data pipeline, epochs, and fine-tuning logic the same, and ensure the submission is written as `submission.csv` with the required columns and row order aligned to `sample_submission.csv`. These changes should both unblock end-to-end execution and recover your intended fine-tuning step (likely improving accuracy toward the target).'
- What this solution (achieved 0.59604) has done: 'The timeout is dominated by the `ImageDataGenerator` Python pipeline doing per-image work on the CPU, and in particular by `_train_preprocess()` calling a KerasCV `RandAugment` layer and then `.numpy()` for every batch element (forces eager execution + device sync). I keep the same model, epochs, optimizers, augmentation semantics, and training loops, but move augmentation+preprocess into the TensorFlow graph via a `tf.data` pipeline that wraps the existing generator output (so labels/shuffling/class_weight behavior stays identical). I also remove the per-batch `.numpy()` conversion, enable dataset prefetching, and set TensorFlow threading to reduce input overhead; prediction is similarly wrapped for faster throughput. These changes are equivalent in semantics (same images, same augmentations, same preprocessing, same steps/epochs), but drastically cut Python overhead so it can finish under 600 seconds.'
- What this solution (achieved 0.54148) has done: 'Your current score is far below the target, so we should make a small, safe change that improves accuracy without changing the core model/training approach. The biggest issue is a label-index mismatch: `flow_from_dataframe` maps string labels to class indices (0..4) based on sorting, but at inference you output argmax indices directly, which may not correspond to the original Kaggle label IDs. I add an explicit, fixed class list `['0','1','2','3','4']` to both train/valid generators to force a consistent mapping, and I map predicted indices back to the correct label IDs using `train_gen.class_indices`. This preserves architecture, epochs, loss, and training loops, but fixes submission semantics so the score should move substantially toward your target.'
- What this solution (achieved 0.69806) has done: 'Your score is far below the target, so we should make a small change that improves accuracy without changing the model/training “shape.” The biggest training-quality issue here is that you are *double augmenting*: images are augmented once by `ImageDataGenerator` (rotation/shift/zoom/flip) and then again by `RandAugment`, which often hurts accuracy for cassava at small epoch budgets. I keep the same EfficientNetV2B0 backbone, same head, same epochs, same optimizers, same loss, and the same tf.data wrapping; I only disable the `ImageDataGenerator` geometric augmentations so augmentation happens exactly once (via RandAugment), which is typically a safe and meaningful accuracy lift. I also fix `class_weight` keys to match the generator’s class indices explicitly (stability/correctness) while keeping the same weighting values.'
- What this solution (achieved 0.42302) has done: 'To move your accuracy up toward the 0.8482 target without changing the model/training “shape,” I make two minimal training-quality fixes that are commonly worth a sizeable jump on Cassava: (1) enable ImageNet-pretrained BatchNorm layers to update their moving statistics during fine-tuning (currently they’re frozen, which often hurts domain adaptation), and (2) set `base(..., training=True)` during the fine-tuning phase so the backbone actually runs in training mode (BN + dropout behavior), while keeping the same epochs, losses, optimizers, and data pipeline. These are narrow changes limited to fine-tuning behavior; everything else (architecture, augmentations, epochs, class weighting, submission alignment) stays the same. This should increase score (your current 0.698 is far below target) while staying stable and within time.'
- What this solution (achieved 0.70329) has done: 'Your current score (0.42302) is far below the target (0.8482), and the biggest “minimal-change” accuracy issue is that your focal-loss augmentation layer is being applied to an entire batch tensor at once, which KerasCV RandAugment does not reliably handle in graph mode and can silently distort images; we apply RandAugment per-image (vectorized via `tf.map_fn`) to restore intended augmentation behavior without changing the model/training loop. Next, we make the fine-tuning stage actually fine-tune by reusing the same model and just recompiling after unfreezing (instead of rebuilding a new graph that can reset some layer behaviors), keeping the architecture identical but improving training stability/consistency. Finally, we ensure the test pipeline uses the exact same preprocessing function signature (batch-safe) to avoid shape retracing and potential inconsistencies. These are narrow, execution-safe fixes that should move accuracy upward toward your target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception:
        pass


_ensure_protobuf_compatible()

import random
import numpy as np
import pandas as pd
import tensorflow as tf

import tf_keras as keras
from tf_keras import layers, models
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.applications import efficientnet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(2, os.cpu_count() // 2))
except Exception:
    pass

try:
    tf.config.experimental.disable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())
print("Using tf_keras:", keras.__version__)



## === cell 1
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["label"] = train_df["label"].astype(str)

print(train_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_df))




## === cell 2
class SigmoidFocalCrossEntropy(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 3
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED
)

classes_sorted = np.array(sorted(tr_df["label"].unique(), key=lambda x: int(x)))
cw = compute_class_weight(
    class_weight="balanced", classes=classes_sorted, y=tr_df["label"].values
)

import keras_cv

randaug = keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.2)


@tf.function(reduce_retracing=True)
def _train_preprocess_tf(batch_img):
    batch_img = tf.cast(batch_img, tf.float32)

    def _aug_one(img):
        img = randaug(img, training=True)
        img = efficientnet_v2.preprocess_input(img)
        return img

    return tf.map_fn(_aug_one, batch_img, fn_output_signature=tf.float32)


@tf.function(reduce_retracing=True)
def _valid_preprocess_tf(batch_img):
    batch_img = tf.cast(batch_img, tf.float32)
    return efficientnet_v2.preprocess_input(batch_img)


train_datagen = ImageDataGenerator(preprocessing_function=None)
valid_datagen = ImageDataGenerator(preprocessing_function=None)

FIXED_CLASSES = [str(i) for i in range(NUM_CLASSES)]

train_gen = train_datagen.flow_from_dataframe(
    dataframe=tr_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    classes=FIXED_CLASSES,
)

valid_gen = valid_datagen.flow_from_dataframe(
    dataframe=va_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    classes=FIXED_CLASSES,
)

print("Class indices:", train_gen.class_indices)

class_weight = {
    train_gen.class_indices[str(cls)]: float(w) for cls, w in zip(classes_sorted, cw)
}
print("Computed class_weight:", class_weight)

output_signature_train = (
    tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
)
output_signature_valid = (
    tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_gen, output_signature=output_signature_train
)
train_ds = train_ds.map(
    lambda x, y: (_train_preprocess_tf(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_gen, output_signature=output_signature_valid
)
valid_ds = valid_ds.map(
    lambda x, y: (_valid_preprocess_tf(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # keep baseline phase identical

BACKBONE_TRAINING = tf.Variable(False, dtype=tf.bool, trainable=False)


def _backbone_call(x):
    return base(x, training=BACKBONE_TRAINING)


inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Lambda(_backbone_call)(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 3

steps_per_epoch = int(np.ceil(train_gen.samples / train_gen.batch_size))
validation_steps = int(np.ceil(valid_gen.samples / valid_gen.batch_size))

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    class_weight=class_weight,
)

base.trainable = True
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True

BACKBONE_TRAINING.assign(True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

FT_EPOCHS = 1
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS + FT_EPOCHS,
    initial_epoch=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    class_weight=class_weight,
)

BACKBONE_TRAINING.assign(False)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2819169594.py in <cell line: 0>()
    161 
    162 FT_EPOCHS = 1
--> 163 model.fit(
    164     train_ds,
    165     validation_data=valid_ds,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py in tf__train_function(iterator)
     16                 except:
     17                     do_return = False
---> 18                     raise
     19                 return fscope.ret(retval_, do_return)
     20         return tf__train_function

/tmp/ipykernel_11/2819169594.py in _backbone_call(x)
    115 
    116 def _backbone_call(x):
--> 117     return base(x, training=BACKBONE_TRAINING)
    118 
    119 

TypeError: in user code:

    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1398, in train_function  *
        return step_function(self, iterator)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1381, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1370, in run_step  **
        outputs = model.train_step(data)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1147, in train_step
        y_pred = self(x, training=True)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/tmp/ipykernel_11/2819169594.py", line 117, in _backbone_call
        return base(x, training=BACKBONE_TRAINING)

    TypeError: Exception encountered when calling layer 'stem_bn' (type BatchNormalization).
    
    Variable is unhashable. Instead, use variable.ref() as the key. (Variable: <tf.Variable 'Variable:0' shape=() dtype=bool>)
    
    Call arguments received by layer 'stem_bn' (type BatchNormalization):
      • inputs=tf.Tensor(shape=(None, 112, 112, 32), dtype=float32)
      • training=<tf.Variable 'Variable:0' shape=() dtype=bool>
      • mask=None


## === cell 4
test_datagen = ImageDataGenerator(preprocessing_function=None)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=sample_df,
    directory=TEST_IMG_DIR,
    x_col="image_id",
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

test_steps = int(np.ceil(test_gen.samples / test_gen.batch_size))

output_signature_test = tf.TensorSpec(
    shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32
)
test_ds = tf.data.Dataset.from_generator(
    lambda: test_gen, output_signature=output_signature_test
)

test_ds = test_ds.map(
    _valid_preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

probs = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
pred_indices = probs.argmax(axis=1).astype(int)

assert len(pred_indices) == len(sample_df), (len(pred_indices), len(sample_df))

idx_to_label = {v: int(k) for k, v in train_gen.class_indices.items()}
pred_labels = np.vectorize(idx_to_label.get)(pred_indices).astype(int)

submission = pd.DataFrame(
    {"image_id": sample_df["image_id"].values, "label": pred_labels}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2641901950.py in <cell line: 0>()
     25 ).prefetch(tf.data.AUTOTUNE)
     26 
---> 27 probs = model.predict(
     28     test_ds,
     29     steps=test_steps,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py in tf__predict_function(iterator)
     16                 except:
     17                     do_return = False
---> 18                     raise
     19                 return fscope.ret(retval_, do_return)
     20         return tf__predict_function

/tmp/ipykernel_11/2819169594.py in _backbone_call(x)
    115 
    116 def _backbone_call(x):
--> 117     return base(x, training=BACKBONE_TRAINING)
    118 
    119 

TypeError: in user code:

    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2436, in predict_function  *
        return step_function(self, iterator)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2421, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2409, in run_step  **
        outputs = model.predict_step(data)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2377, in predict_step
        return self(x, training=False)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/tmp/ipykernel_11/2819169594.py", line 117, in _backbone_call
        return base(x, training=BACKBONE_TRAINING)

    TypeError: Exception encountered when calling layer 'stem_bn' (type BatchNormalization).
    
    Variable is unhashable. Instead, use variable.ref() as the key. (Variable: <tf.Variable 'Variable:0' shape=() dtype=bool>)
    
    Call arguments received by layer 'stem_bn' (type BatchNormalization):
      • inputs=tf.Tensor(shape=(None, 112, 112, 32), dtype=float32)
      • training=<tf.Variable 'Variable:0' shape=() dtype=bool>
      • mask=None
