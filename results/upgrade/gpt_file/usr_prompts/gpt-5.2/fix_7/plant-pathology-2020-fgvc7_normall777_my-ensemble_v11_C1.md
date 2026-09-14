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

0.9734033978905674

# 6. Current score

0.4399

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.4348) has done: 'The timeout is dominated by training two very large backbones (EfficientNetB7 and DenseNet201) for 40 epochs on 512×512 images; on CPU/GPU this easily exceeds 10 minutes. To keep the core logic identical while making it finish in time, the key fix is to always use the provided pretrained `.h5` weights (and fail fast with a clear error if they’re missing) so no training occurs in this runtime-constrained environment. Additionally, the input pipeline is made faster but equivalent by caching decoded/resized images (before random augmentations), setting deterministic options, using `drop_remainder=True`, and enabling dataset optimizations/prefetching to reduce input overhead during prediction. Model architecture, loss, training loop semantics (only used if weights exist), and prediction blending remain unchanged.'
- What this solution (achieved 0.4348) has done: 'The timeout is dominated by training two very large backbones (EfficientNetB7 + DenseNet201) for 5 epochs at 512×512, and the current input pipeline also uses `.cache()` in-memory for all decoded 512×512 float images, which is slow/heavy and can thrash memory. To keep identical learning logic while reducing wall time, the main fix is to always use the provided pretrained `.h5` weights when available and avoid accidentally training (and to fail fast if the weight files aren’t present in this environment). Additionally, the data pipeline is made faster-but-equivalent by caching decoded images to disk (not RAM) and by setting TF runtime thread settings (CPU path) without changing any model/math. Prediction is also sped up slightly by ensuring datasets don’t do extra work and by keeping deterministic behavior intact.'
- What this solution (achieved 0.4399) has done: 'The timeout is most likely from training two very large backbones at 512×512 when the provided `.h5` weights are missing, plus redundant work in the input pipeline (caching decoded images to disk and still repeating/augmenting each epoch). To keep the exact core logic (same models, losses, training loops, and final blending), I (1) make the input pipeline faster and more deterministic by enabling fused map/batch, raising parallelism, and caching in-memory (instead of writing large cache files), and (2) ensure we never accidentally train during submission by failing fast if the external pretrained weights are not available (since training is what blows the 600s budget). Predictions remain identical (up to negligible FP differences) when weights exist, and evaluation semantics/ensemble logic are unchanged.'

# 9. Code solution

## === cell 0
import os, random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import DenseNet201

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.get_logger().setLevel("ERROR")

try:
    ncpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, ncpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(8, ncpu))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"

EPOCHS = 5
BATCH_SIZE = 8 * strategy.num_replicas_in_sync




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(DATA_DIR, "images", f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = [c for c in sub.columns if c != "image_id"]

train_paths = (DATA_DIR + "/images/" + train["image_id"].astype(str) + ".jpg").values
test_paths = (DATA_DIR + "/images/" + test["image_id"].astype(str) + ".jpg").values

train_labels = train[target_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print("Train/Valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", target_cols)




## === cell 4
img_size = 512


@tf.function
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


@tf.function
def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    return image, label




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_and_batch_fusion = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
)




## === cell 6
def get_model(use_model, weights="imagenet"):
    base_model = use_model(
        weights=weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output

    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


def maybe_load_weights(model, weight_path):
    if weight_path and tf.io.gfile.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Loaded weights: {weight_path}")
        return True
    print(f"Pretrained .h5 weights not found (will train): {weight_path}")
    return False




## === cell 7
with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"

loaded1 = maybe_load_weights(model1, w1)
loaded2 = maybe_load_weights(model2, w2)

if (not loaded1) or (not loaded2):
    raise FileNotFoundError(
        "Missing pretrained weights (.h5). Training would likely exceed the 600s timeout and change results. "
        f"Found: EfficientNetB7={loaded1}, DenseNet201={loaded2}. "
        "Please ensure the Kaggle dataset with these weight files is added/enabled."
    )




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3131063275.py in <cell line: 0>()
     14 # different results anyway (and likely time out), so we fail fast instead of timing out.
     15 if (not loaded1) or (not loaded2):
---> 16     raise FileNotFoundError(
     17         "Missing pretrained weights (.h5). Training would likely exceed the 600s timeout and change results. "
     18         f"Found: EfficientNetB7={loaded1}, DenseNet201={loaded2}. "

FileNotFoundError: Missing pretrained weights (.h5). Training would likely exceed the 600s timeout and change results. Found: EfficientNetB7=False, DenseNet201=False. Please ensure the Kaggle dataset with these weight files is added/enabled.

## === cell 8
best_alpha = 0.90
print("Вычисляем предсказания...")

probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

sub = sub[["image_id"] + target_cols].copy()
sub[target_cols] = probabilities

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
