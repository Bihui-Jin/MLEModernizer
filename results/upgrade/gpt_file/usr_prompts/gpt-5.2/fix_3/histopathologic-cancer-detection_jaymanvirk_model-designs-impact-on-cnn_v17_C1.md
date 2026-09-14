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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the two root blockers: the protobuf/tensorflow-io import crash and the brittle file discovery that reads non-CSV files. Then I make dataset creation robust for both (paths, labels) and (paths-only) cases, and ensure model training actually runs and the saved model exists before loading. Finally, I generate probabilities (not rounded 0/1) for `label` to match the ROC-AUC metric and write a valid `submission.csv` with the required columns and order.'
- What this solution (achieved 0.5) has done: 'I fix the two execution blockers that prevent training/inference: (1) TensorFlow crashes due to a protobuf incompatibility and (2) tensorflow-io loads but fails at runtime with a missing `libtensorflow_io.so` symbol, so it must be disabled and we should decode TIFFs via PIL consistently. Then I ensure `train_dataset`/`test_dataset` are always defined (remove the premature `del` usage side-effects) so the model actually trains and the intended `.h5` file exists before loading. Finally, I generate probabilistic predictions (not hard 0/1) for ROC-AUC and write a properly aligned `submission.csv` with the required columns and order; this should move the score above the current 0.5 baseline toward the target.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
print_number_files(train_dir)



## === cell 7
print_number_files(test_dir)



## === cell 8
del print_short_summary, print_number_files



## === cell 9
plt.figure(figsize=(16, 9))
tmp = train_data["label"].value_counts()
sns.barplot(y=["No Cancer", "Cancer"], x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Label")
plt.title("Number of records per label")
plt.show()



## === cell 10
del tmp




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
img_path = train_dir + train_data["id"][0] + ".tif"
img = Image.open(img_path)
print("Original image size: {}".format(img.size))



## === cell 13
data = get_image_label(train_dir, train_data, [0, 1])



## === cell 14
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 9))

labels = ["No Cancer", "Cancer"]
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(data[row][col])
    axes[row, col].set_title(labels[row])
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()



## === cell 15
del get_images_to_plot, get_image_label, img_path, img
del data, fig, axes, labels, row, col



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
def _decode_tiff_to_tensor(image_path):
    """Decode .tif into float32 tensor in [0,1], resized to 32x32, with 4 channels (RGBA)."""

    def _py_load(p):
        p = p.decode("utf-8")
        im = Image.open(p).convert("RGBA").resize((32, 32))
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (32,32,4)
        return arr

    img = tf.py_function(func=_py_load, inp=[image_path], Tout=tf.float32)
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
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, (tuple, list)) and len(data) == 2:
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        dataset = dataset.map(
            lambda p, y: get_decoded_image(p, y), num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




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
runtime_base, scores_base = get_model_results("model_base", get_model_base)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/3774139332.py in <cell line: 0>()
----> 1 runtime_base, scores_base = get_model_results("model_base", get_model_base)
      2 

/tmp/ipykernel_11/2873180582.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

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

UnknownError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filenri6xqiw.py", line 17, in _py_load
    p = ag__.converted_call(ag__.ld(p).decode, ('utf-8',), None, fscope_1)
                            ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 31
plot_model_scores(scores_base, "base model")



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2757851445.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base, "base model")
      2 

NameError: name 'scores_base' is not defined

## === cell 32
runtime_base_deep, scores_base_deep = get_model_results(
    "model_base_deep", get_model_base_deep
)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1501219057.py in <cell line: 0>()
----> 1 runtime_base_deep, scores_base_deep = get_model_results(
      2     "model_base_deep", get_model_base_deep
      3 )
      4 

/tmp/ipykernel_11/2873180582.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

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

UnknownError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filenri6xqiw.py", line 17, in _py_load
    p = ag__.converted_call(ag__.ld(p).decode, ('utf-8',), None, fscope_1)
                            ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 33
plot_model_scores(scores_base_deep, "base + additional layers model")



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435582006.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_deep, "base + additional layers model")
      2 

NameError: name 'scores_base_deep' is not defined

## === cell 34
runtime_base_wide, scores_base_wide = get_model_results(
    "model_base_wide", get_model_base_wide
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/389434075.py in <cell line: 0>()
----> 1 runtime_base_wide, scores_base_wide = get_model_results(
      2     "model_base_wide", get_model_base_wide
      3 )
      4 

/tmp/ipykernel_11/2873180582.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

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

UnknownError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filenri6xqiw.py", line 17, in _py_load
    p = ag__.converted_call(ag__.ld(p).decode, ('utf-8',), None, fscope_1)
                            ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 35
plot_model_scores(scores_base_wide, "base + wider layers model")



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/178024283.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_wide, "base + wider layers model")
      2 

NameError: name 'scores_base_wide' is not defined

## === cell 36
runtime_base_maxpool, scores_base_maxpool = get_model_results(
    "model_base_maxpool", get_model_base_maxpool
)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/3139894174.py in <cell line: 0>()
----> 1 runtime_base_maxpool, scores_base_maxpool = get_model_results(
      2     "model_base_maxpool", get_model_base_maxpool
      3 )
      4 

/tmp/ipykernel_11/2873180582.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

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

UnknownError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filenri6xqiw.py", line 17, in _py_load
    p = ag__.converted_call(ag__.ld(p).decode, ('utf-8',), None, fscope_1)
                            ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 37
plot_model_scores(scores_base_maxpool, "base + max pooling model")



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3164274992.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_maxpool, "base + max pooling model")
      2 

NameError: name 'scores_base_maxpool' is not defined

## === cell 38
runtime_base_dropout, scores_base_dropout = get_model_results(
    "model_base_dropout", get_model_base_dropout
)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1796779764.py in <cell line: 0>()
----> 1 runtime_base_dropout, scores_base_dropout = get_model_results(
      2     "model_base_dropout", get_model_base_dropout
      3 )
      4 

/tmp/ipykernel_11/2873180582.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

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

UnknownError: {{function_node __wrapped__IteratorGetNextAsOptional_device_/job:localhost/replica:0/task:0/device:GPU:0}} AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filenri6xqiw.py", line 17, in _py_load
    p = ag__.converted_call(ag__.ld(p).decode, ('utf-8',), None, fscope_1)
                            ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py", line 260, in __getattr__
    self.__getattribute__(name)

AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'decode'


	 [[{{node EagerPyFunc}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNextAsOptional] name: 

## === cell 39
plot_model_scores(scores_base_dropout, "base + dropout model")



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/110224612.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_dropout, "base + dropout model")
      2 

NameError: name 'scores_base_dropout' is not defined

## === cell 40
results = [
    ("Base", runtime_base, scores_base),
    ("Base + Add. layers", runtime_base_deep, scores_base_deep),
    ("Base + Wider layers", runtime_base_wide, scores_base_wide),
    ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),
    ("Base + Dropout", runtime_base_dropout, scores_base_dropout),
]
table = []
for i in range(len(results)):
    tmp = {
        "model": results[i][0],
        "runtime (sec)": results[i][1],
        "train_roc_auc_score": results[i][2][0][-1],
        "test_roc_auc_score": results[i][2][1][-1],
    }
    table.append(tmp)

pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime (sec)"], ascending=[False, True]
)



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2027199582.py in <cell line: 0>()
      1 results = [
----> 2     ("Base", runtime_base, scores_base),
      3     ("Base + Add. layers", runtime_base_deep, scores_base_deep),
      4     ("Base + Wider layers", runtime_base_wide, scores_base_wide),
      5     ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),

NameError: name 'runtime_base' is not defined

## === cell 41
try:
    del tmp, table
except NameError:
    pass



## === cell 42
model = load_model("model_base_deep.h5", compile=False)



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/259727692.py in <cell line: 0>()
      1 # Load top performed model (as in original notebook intent).
      2 # If you prefer a different one, adjust the filename only.
----> 3 model = load_model("model_base_deep.h5", compile=False)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_base_deep.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 43
submis_data = test_dir + sample_data["id"] + ".tif"
submis_data = submis_data.values

submis_dataset = get_prefetched_data(submis_data, BATCH_SIZE)



## === cell 44
results = model.predict(submis_dataset, verbose=1)



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/763274262.py in <cell line: 0>()
----> 1 results = model.predict(submis_dataset, verbose=1)
      2 

NameError: name 'model' is not defined

## === cell 45
sample_data["label"] = np.ravel(results).astype(np.float32)



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1850854202.py in <cell line: 0>()
      1 # SCORE FIX: competition expects probabilities; rounding to 0/1 hurts ROC-AUC.
----> 2 sample_data["label"] = np.ravel(results).astype(np.float32)
      3 

NameError: name 'results' is not defined

## === cell 46
sample_data.head()



## === cell 47
submission = sample_data[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 48
try:
    del submis_data, submis_dataset, submission
except NameError:
    pass
