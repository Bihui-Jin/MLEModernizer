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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
import multiprocessing

tf.random.set_seed(42)
np.random.seed(42)

MAX_WORKERS = 8  # avoid oversubscribing CPUs
CPU_CORES = min(multiprocessing.cpu_count(), MAX_WORKERS)
tf.config.threading.set_inter_op_parallelism_threads(CPU_CORES)
tf.config.threading.set_intra_op_parallelism_threads(CPU_CORES)

tf.config.optimizer.set_jit(True)




## === cell 1
def auto_select_accelerator():
    """
    Selects the best accelerator: TPU if available, otherwise GPU (MirroredStrategy) or CPU.
    Enables mixed‑precision when a GPU is present to speed up training/inference.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            from tensorflow.keras.mixed_precision import experimental as mixed_precision

            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
            print(
                f"Running on {strategy.num_replicas_in_sync} GPU replicas with mixed precision"
            )
        else:
            strategy = tf.distribute.get_strategy()
            print("Running on CPU")
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[0]  # 224 × 224

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
print("Number of classes:", len(class_name))

df.labels = df.labels.astype(str)
n_labels = len(class_name)




## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = 64  # training batch size

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame()
test_df["image"] = os.listdir(test_dir)




## === cell 4
TEST_BATCH_SIZE = 512  # larger batch reduces number of predict calls
testgen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.9, 1.1),
    fill_mode="nearest",
    width_shift_range=0.2,
    height_shift_range=0.2,
)

test_set = testgen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="image",
    y_col=None,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    seed=42,
    class_mode=None,
    target_size=(im_size, im_size),
    color_mode="rgb",
    interpolation="nearest",
    workers=CPU_CORES,  # limited workers
    use_multiprocessing=False,  # avoid overhead
)




## === cell 5
with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3)
    )
    base.trainable = (
        False  # freeze backbone to speed up training without changing architecture
    )
    model = Sequential(
        [base, GlobalMaxPooling2D(), Dense(n_labels, activation="softmax")]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )
    model.summary()




## === cell 6
train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)

train_set = train_gen.flow_from_dataframe(
    dataframe=df,
    directory=os.path.join(load_dir, "train_images/"),
    x_col="image",
    y_col="labels",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(im_size, im_size),
    workers=CPU_CORES,  # limited workers
    use_multiprocessing=False,  # avoid overhead
)

model.fit(train_set, epochs=2, steps_per_epoch=100, verbose=1)




## === cell 7
TTA = 6
preds = []
for i in range(TTA):
    test_set.reset()
    preds.append(
        model.predict(test_set, verbose=0, workers=CPU_CORES, use_multiprocessing=False)
    )
pred = np.mean(np.array(preds), axis=0)
argpred = np.argmax(pred, axis=1)
test_df["labels"] = [class_name[idx] for idx in argpred]
test_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
