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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.7613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/histopathologic-cancer-detection"

TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

train_data = pd.read_csv(TRAIN_CSV)
sample_data = pd.read_csv(SAMPLE_SUBMISSION_CSV)




## === cell 2
def print_short_summary(name, data):
    print(name)
    print("\n1. Data head:")
    print(data.head())
    print("\n2. Data shape: {}".format(data.shape))
    print("\n3. Data info:")
    data.info()


def print_number_files(dirpath):
    print(f"{dirpath}: {len(os.listdir(dirpath))} files")




## === cell 3
print_short_summary("Train data", train_data)
print_number_files(TRAIN_DIR)
print_number_files(TEST_DIR)




## === cell 4
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=0
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)




## === cell 5
image_paths = (
    balanced_train_data["id"]
    .apply(lambda x: os.path.join(TRAIN_DIR, f"{x}.tif"))
    .values
)
labels = balanced_train_data["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.25, shuffle=True, random_state=0
)




## === cell 6
def get_decoded_image(image_path, label=None):
    """Load a TIFF image with TensorFlow, convert to RGBA, resize to 32×32, and normalise."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=4, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    img = tf.image.resize(img, [32, 32])
    if label is None:
        return img
    else:
        return img, label


def get_prefetched_data(data, batch_size):
    """Create a tf.data.Dataset from image paths (and optional labels) with caching."""
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, tuple):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            lambda p, l: get_decoded_image(p, l), num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            lambda p: get_decoded_image(p), num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.cache()  # cache decoded tensors in memory
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 7
BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE)
val_dataset = get_prefetched_data((X_val, y_val), BATCH_SIZE)




## === cell 8
def get_model_base():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model




## === cell 9
def get_compiled_model(model_fn):
    """Compile model with MirroredStrategy if GPUs are present."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices:", strategy.num_replicas_in_sync)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("No GPU available, using CPU.")

    with strategy.scope():
        model = model_fn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
    return model




## === cell 10
def get_model_results(model_name, model_fn):
    """Train the model for a few epochs and return runtime + scores."""
    model = get_compiled_model(model_fn)
    start = time.time()
    model.fit(train_dataset, epochs=5, validation_data=val_dataset, verbose=0)
    runtime = time.time() - start

    model.save(f"{model_name}.h5")

    train_scores = model.history.history["auc"]
    val_scores = model.history.history["val_auc"]
    tf.keras.backend.clear_session()
    return runtime, (train_scores, val_scores), model  # also return the trained model




## === cell 11
runtime_base, scores_base, trained_base_model = get_model_results(
    "model_base", get_model_base
)
print(f"Base model runtime: {runtime_base:.2f}s")
print(
    f"Final train AUC: {scores_base[0][-1]:.4f}, validation AUC: {scores_base[1][-1]:.4f}"
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/2457383500.py in <cell line: 0>()
----> 1 runtime_base, scores_base, trained_base_model = get_model_results(
      2     "model_base", get_model_base
      3 )
      4 print(f"Base model runtime: {runtime_base:.2f}s")
      5 print(

/tmp/ipykernel_55/2978669031.py in get_model_results(model_name, model_fn)
      3     model = get_compiled_model(model_fn)
      4     start = time.time()
----> 5     model.fit(train_dataset, epochs=5, validation_data=val_dataset, verbose=0)
      6     runtime = time.time() - start
      7 

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 12
test_paths = (
    sample_data["id"].apply(lambda x: os.path.join(TEST_DIR, f"{x}.tif")).values
)
test_dataset = get_prefetched_data(test_paths, BATCH_SIZE)

pred_probs = trained_base_model.predict(test_dataset, verbose=0).ravel()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2419280860.py in <cell line: 0>()
      4 test_dataset = get_prefetched_data(test_paths, BATCH_SIZE)
      5 
----> 6 pred_probs = trained_base_model.predict(test_dataset, verbose=0).ravel()
      7 
      8 

NameError: name 'trained_base_model' is not defined

## === cell 13
submission = pd.DataFrame({"id": sample_data["id"], "label": pred_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3255962698.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": sample_data["id"], "label": pred_probs})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")

NameError: name 'pred_probs' is not defined
