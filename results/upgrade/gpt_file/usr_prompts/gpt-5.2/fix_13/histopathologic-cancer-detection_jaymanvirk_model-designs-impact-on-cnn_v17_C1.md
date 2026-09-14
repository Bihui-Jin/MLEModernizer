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

0.7367

# 6. Current score

0.89128

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the two root blockers: the protobuf/tensorflow-io import crash and the brittle file discovery that reads non-CSV files. Then I make dataset creation robust for both (paths, labels) and (paths-only) cases, and ensure model training actually runs and the saved model exists before loading. Finally, I generate probabilities (not rounded 0/1) for `label` to match the ROC-AUC metric and write a valid `submission.csv` with the required columns and order.'
- What this solution (achieved 0.5) has done: 'I fix the two execution blockers that prevent training/inference: (1) TensorFlow crashes due to a protobuf incompatibility and (2) tensorflow-io loads but fails at runtime with a missing `libtensorflow_io.so` symbol, so it must be disabled and we should decode TIFFs via PIL consistently. Then I ensure `train_dataset`/`test_dataset` are always defined (remove the premature `del` usage side-effects) so the model actually trains and the intended `.h5` file exists before loading. Finally, I generate probabilistic predictions (not hard 0/1) for ROC-AUC and write a properly aligned `submission.csv` with the required columns and order; this should move the score above the current 0.5 baseline toward the target.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by (1) writing TFRecords by decoding every image via PIL in pure Python (train + test) and (2) building a single huge TFRecord for the entire test set, both of which perform tens/hundreds of thousands of slow per-file operations before training/prediction even starts. To preserve identical model/training logic while removing this bottleneck, I keep the same resize/normalization/channel handling but switch the input pipeline to a fully-TensorFlow `tf.data` pipeline using `tf.io.read_file` + `tf.image.decode_image`, with parallel mapping, caching (in-memory for train/val), and prefetch. I also remove the forced pure-Python protobuf implementation (which slows TFRecord/TF ops) since we no longer rely on TFRecord caching at all, preserving correctness. All training epochs, model architectures, losses/metrics, and evaluation semantics remain unchanged.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the TIFF decoding bug by replacing `tf.image.decode_image` (which can’t decode `.tif`) with a `tf.py_function` wrapper around PIL so the input pipeline can actually read the dataset without “Unknown image file format”. Finally, I make training and saving resilient: if training fails for any reason, the script still train a fallback model and ensure a model object exists before predicting, so a valid probabilistic `submission.csv` is always written (helping ROC-AUC move above the 0.5 baseline).'
- What this solution (achieved 0.89128) has done: 'I fix the two runtime blockers preventing any training/inference: the TensorFlow/protobuf crash (by removing the incompatible forced pure-Python protobuf setting) and the TIFF decode pipeline bug (your `tf.py_function` receives a `tf.Tensor`, so we must convert it to bytes via `.numpy()` before decoding). These changes preserve the same model architectures, training loop, epochs, and loss/metric, but make the `tf.data` pipeline actually yield images to the model. I also keep predictions as probabilities (not rounded) and ensure the submission file is written as `submission.csv` with `id,label` in the same order as `sample_submission.csv`, which should move the ROC-AUC above the 0.5 baseline toward your target. All paths remain unchanged and it run end-to-end.'
- What this solution (achieved 0.89128) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *before* importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in this environment). Then I keep your existing tf.data + `tf.py_function` TIFF decoding pipeline, but make it slightly more robust by ensuring the path tensor is converted to bytes safely and by setting deterministic behavior consistently. Finally, I ensure the submission is written exactly as `submission.csv` with `id,label` aligned to `sample_submission.csv` order; these changes are score-neutral aside from restoring successful execution (your score is already above target, so we avoid any intentional performance changes).'
- What this solution (achieved 0.89128) has done: 'I fix the immediate runtime crash by removing the forced pure-Python protobuf backend, which is what triggers the `MessageFactory.GetPrototype` AttributeError with TensorFlow 2.18 in this environment. Then I keep your existing tf.data + `tf.py_function` + PIL TIFF decoding pipeline and model/training logic exactly the same, only making a small safety tweak to ensure the path tensor is always converted to a proper UTF-8 string before PIL opens it. Finally, I ensure the submission is always written as `submission.csv` with `id,label` aligned to `sample_submission.csv` order and with probabilistic (not thresholded) predictions; no intentional score-improving changes are introduced since your current score is already above the target.'
- What this solution (achieved 0.89128) has done: 'We need to fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow; this is the root execution blocker. Then we keep your model/training/inference logic unchanged (same architectures, epochs, loss, tf.data pipeline with PIL decode) to avoid unnecessary score changes since your current score is already above the target band. Finally, we ensure the PIL path conversion is robust for `tf.py_function` and that the script always writes a valid `submission.csv` with `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.89128) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf backend setting, which is the direct cause of the `MessageFactory.GetPrototype` error in this environment. I keep your tf.data + `tf.py_function` + PIL TIFF decoding pipeline and the model/training logic unchanged, only making the PIL path conversion a bit safer to avoid any intermittent decode issues. Since your current score (0.89128) is already well above the target (0.7367), I won’t introduce any score-improving changes; the goal is to restore end-to-end execution and produce a valid `submission.csv`. The script still write `submission.csv` with `id,label` aligned to `sample_submission.csv` order and probabilistic predictions for ROC-AUC.'
- What this solution (achieved 0.89128) has done: 'We need to fix the immediate runtime crash in the first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) which happens on importing TensorFlow due to protobuf backend incompatibility in this environment. The minimal, score-neutral fix is to force the pure-Python protobuf implementation **before** importing TensorFlow (and ensure TensorFlow is not imported earlier anywhere). Everything else (data paths, tf.data+PIL TIFF decoding, model architectures, training epochs, probabilistic submission) can remain unchanged to avoid further score changes since your current score is already above the target. I also keep the submission writing logic intact and ensure the output is `submission.csv` with `id,label` in sample order.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tfio = None
_HAS_TFIO = False

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = "/kaggle/input/histopathologic-cancer-detection"

sample_path = os.path.join(input_dir, "sample_submission.csv")
train_labels_path = os.path.join(input_dir, "train_labels.csv")
train_dir = os.path.join(input_dir, "train") + "/"
test_dir = os.path.join(input_dir, "test") + "/"

assert os.path.exists(sample_path), f"Missing: {sample_path}"
assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.isdir(train_dir), f"Missing: {train_dir}"
assert os.path.isdir(test_dir), f"Missing: {test_dir}"

sample_data = pd.read_csv(sample_path)
train_data = pd.read_csv(train_labels_path)

list_l = [sample_path, train_labels_path, test_dir, train_dir]
list_l



## === cell 2
del list_l




## === cell 3
def print_short_summary(name, data):
    """
    Prints data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print("\n1. Data head:")
    print(data.head())
    print("\n2. Data shape: {}".format(data.shape))
    print("\n3. Data info:")
    data.info()


def print_number_files(dirpath):
    print("{}: {} files".format(dirpath, len(os.listdir(dirpath))))




## === cell 4
print_short_summary("Train data", train_data)



## === cell 5
print_short_summary("Sample data", sample_data)



## === cell 6
pass



## === cell 7
pass



## === cell 8
del print_short_summary, print_number_files



## === cell 9
pass



## === cell 10
pass




## === cell 11
def get_images_to_plot(file_names):
    """
    Returns list of images
    Args:
        file_names: list of filenames
    Returns:
        list of image objects
    """
    return [Image.open(f) for f in file_names]


def get_image_label(dirname, data, labels, n=5):
    """
    Return dictionary with label-imagepath
    Args:
        dirname: name of the directory
        data: dataset of file names
        labels: list of labels
        n (opt): number of images per label
    Returns:
        dict_img: dictionary with label-imagepath pairs
    """
    dict_img = {}
    for l in labels:
        indexes = data["label"] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp["id"] + ".tif"
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp

    return dict_img




## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
try:
    del get_images_to_plot, get_image_label, img_path, img
    del data, fig, axes, labels, row, col
except Exception:
    pass



## === cell 16
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=SEED
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=SEED).reset_index(
    drop=True
)



## === cell 17
del no_cancer, cancer, no_cancer_downsampled



## === cell 18
image_paths = train_dir + balanced_train_data["id"] + ".tif"
image_paths = image_paths.values

labels = balanced_train_data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.25,
    shuffle=True,
    random_state=SEED,
    stratify=labels,
)



## === cell 19
del image_paths, labels




## === cell 20
def _pil_decode_resize_rgba(path_tensor):
    if hasattr(path_tensor, "numpy"):
        path_bytes = path_tensor.numpy()
    else:
        path_bytes = path_tensor

    if isinstance(path_bytes, (bytes, bytearray, np.bytes_)):
        path = bytes(path_bytes).decode("utf-8")
    else:
        path = str(path_bytes)

    with Image.open(path) as im:
        im = im.convert("RGBA")
        im = im.resize((32, 32), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _decode_tiff_to_tensor(image_path):
    """Decode .tif into float32 tensor in [0,1], resized to 32x32, with 4 channels (RGBA)."""
    img = tf.py_function(_pil_decode_resize_rgba, [image_path], Tout=tf.float32)
    img.set_shape((32, 32, 4))
    return img


def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images.
    Decode image with 4 channels RGBA.
    Resize image to 32x32px.
    Scale pixels from 0 to 1.
    """
    img = _decode_tiff_to_tensor(image_path)
    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size):
    """
    Create a TensorFlow dataset from image paths and labels or paths-only.
    Parallel decode + cache (train/val) + prefetch.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, (tuple, list)) and len(data) == 2:
        paths, labs = data
        paths = tf.convert_to_tensor(paths, dtype=tf.string)
        labs = tf.convert_to_tensor(labs, dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labs))
        ds = ds.map(
            lambda p, y: (get_decoded_image(p, None), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    else:
        paths = tf.convert_to_tensor(data, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: get_decoded_image(p, None),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds




## === cell 21
BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE)



## === cell 22
del X_train, y_train, X_test, y_test




## === cell 23
def get_model_base():
    """
    Return base model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 24
def get_model_base_deep():
    """
    Return deeper model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 25
def get_model_base_wide():
    """
    Return wider model architecture
    """
    model_drop_bn = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model_drop_bn




## === cell 26
def get_model_base_maxpool():
    """
    Return maxpool model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 27
def get_model_base_dropout():
    """
    Return dropout model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 28
def get_compiled_model(func):
    """
    Create model to be trained with a multi-GPU strategy.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("No GPU available, falling back to CPU.")

    with strategy.scope():
        compiled_model = func()
        compiled_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

    return compiled_model




## === cell 29
def plot_model_scores(scores, model_name):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, label="Train score")
    plt.plot(epochs, test_scores, label="Test score")
    plt.title("Train and test ROC AUC scores of the {}".format(model_name))
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_model_results(model_name, model):
    """
    Return tuple of runtime, train and test scores.
    Compile, fit and save model along the way.
    """
    m = get_compiled_model(model)

    st = time.time()
    history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
    runtime = time.time() - st

    m.save("{}.h5".format(model_name))

    train_scores = history.history["auc"]
    test_scores = history.history["val_auc"]

    tf.keras.backend.clear_session()

    return (runtime, (train_scores, test_scores))




## === cell 30
try:
    runtime_base_deep, scores_base_deep = get_model_results(
        "model_base_deep", get_model_base_deep
    )
except Exception as e:
    print("Deep model training failed, falling back to base model. Error:", repr(e))
    runtime_base_deep, scores_base_deep = get_model_results(
        "model_base_deep", get_model_base
    )



## === cell 31
pass



## === cell 32
runtime_base = None
scores_base = None



## === cell 33
runtime_base_wide = None
scores_base_wide = None



## === cell 34
runtime_base_maxpool = None
scores_base_maxpool = None



## === cell 35
runtime_base_dropout = None
scores_base_dropout = None



## === cell 36
pass



## === cell 37
try:
    del tmp, table
except NameError:
    pass



## === cell 38
model_path = "model_base_deep.h5"
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
else:
    print("Model file missing; training a model in-memory for submission.")
    model = get_compiled_model(get_model_base_deep)
    model.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)

try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
except Exception:
    pass



## === cell 39
submis_data = test_dir + sample_data["id"] + ".tif"
submis_data = submis_data.values

submis_dataset = get_prefetched_data(submis_data, BATCH_SIZE)



## === cell 40
results = model.predict(submis_dataset, verbose=1)



## === cell 41
sample_data["label"] = np.ravel(results).astype(np.float32)



## === cell 42
sample_data.head()



## === cell 43
submission = sample_data[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 44
try:
    del submis_data, submis_dataset, submission
except NameError:
    pass
