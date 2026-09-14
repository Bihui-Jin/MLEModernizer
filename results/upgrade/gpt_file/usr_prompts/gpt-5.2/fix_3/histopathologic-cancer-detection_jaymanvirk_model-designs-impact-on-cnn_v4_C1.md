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

0.4964

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from PIL import Image
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")
np.random.seed(0)
tf.random.set_seed(0)



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

sample_data = pd.read_csv(sample_path)
train_data = pd.read_csv(train_labels_path)

(train_dir, test_dir, sample_data.shape, train_data.shape)




## === cell 2
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




## === cell 3
print_short_summary("Train data", train_data)



## === cell 4
print_short_summary("Sample data", sample_data)



## === cell 5
print_number_files(train_dir)



## === cell 6
print_number_files(test_dir)



## === cell 7
plt.figure(figsize=(16, 9))
tmp = train_data["label"].value_counts()
sns.barplot(y=["No Cancer", "Cancer"], x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Label")
plt.title("Number of records per label")
plt.show()




## === cell 8
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
    dict_img = {}
    for l in labels:
        indexes = data["label"] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp["id"] + ".tif"
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp

    return dict_img




## === cell 9
img_path = train_dir + train_data["id"][0] + ".tif"
img = Image.open(img_path)
print("Original image size: {}".format(img.size))



## === cell 10
data = get_image_label(train_dir, train_data, [0, 1])



## === cell 11
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



## === cell 12
SAMPLE_SIZE = 0.2
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]
cancer = cancer[: int(SAMPLE_SIZE * len(cancer))]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=0
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)

balanced_train_data.shape



## === cell 13
image_paths = train_dir + balanced_train_data["id"] + ".tif"
image_paths = image_paths.values
labels = balanced_train_data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.25, shuffle=True, random_state=0
)

(X_train.shape, X_test.shape)




## === cell 14
def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images using TensorFlow.
    Decode image, resize to 32x32px, scale pixels to [0,1].
    Ensure 4 channels (RGBA) to match the existing model architecture.
    """
    img_bytes = tf.io.read_file(image_path)

    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1], float32
    img = tf.image.resize(img, [32, 32], method="bilinear", antialias=True)

    alpha = tf.ones([32, 32, 1], dtype=img.dtype)
    img = tf.concat([img, alpha], axis=-1)

    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size, buffer_size, training=True):
    """
    Create a TensorFlow dataset from image paths (and optional labels).
    If training=True, shuffle; if not, keep deterministic order for submission alignment.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if isinstance(data, tuple):
        dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
        )

    if training:
        dataset = dataset.shuffle(
            buffer_size=buffer_size, seed=0, reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 15
BATCH_SIZE = 64
TRAIN_BUFFER_SIZE = X_train.shape[0]
TEST_BUFFER_SIZE = X_test.shape[0]

train_dataset = get_prefetched_data(
    (X_train, y_train), BATCH_SIZE, TRAIN_BUFFER_SIZE, training=True
)
test_dataset = get_prefetched_data(
    (X_test, y_test), BATCH_SIZE, TEST_BUFFER_SIZE, training=False
)




## === cell 16
def roc_auc_score_(y_true, y_pred):
    """
    Calculate ROC AUC score using sklearn built-in function.
    Used in a model.compile as a custom metric.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    def _sk_auc(t, p):
        return np.float64(roc_auc_score(t, p))

    out = tf.py_function(_sk_auc, (y_true, y_pred), Tout=tf.float64)
    out.set_shape([])  # critical for Keras metric reduction
    return out




## === cell 17
model_base = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 18
model_drop_bn = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 19
model_tuned = models.Sequential(
    [
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((4, 4)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(1, activation="sigmoid"),
    ]
)




## === cell 20
def plot_model_scores(scores):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, "b", label="Train score")
    plt.plot(epochs, test_scores, "r", label="Test score")
    plt.title("Train and test ROC AUC scores")
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
    st = time.time()
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_score_]
    )
    history = model.fit(
        train_dataset, epochs=5, validation_data=test_dataset, verbose=2
    )
    runtime = time.time() - st

    model.save(f"{model_name}.keras")

    train_scores = history.history["roc_auc_score_"]
    test_scores = history.history["val_roc_auc_score_"]
    del model

    return (runtime, (train_scores, test_scores))




## === cell 21
runtime_base, scores_base = get_model_results("base", model_base)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/351769605.py in <cell line: 0>()
----> 1 runtime_base, scores_base = get_model_results("base", model_base)
      2 

/tmp/ipykernel_11/1822142920.py in get_model_results(model_name, model)
     26         optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_score_]
     27     )
---> 28     history = model.fit(
     29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )

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

InvalidArgumentError: Graph execution error:

Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_1650]

## === cell 22
plot_model_scores(scores_base)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3689712647.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base)
      2 

NameError: name 'scores_base' is not defined

## === cell 23
runtime_drop_bn, scores_drop_bn = get_model_results("drop_bn", model_drop_bn)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2928168416.py in <cell line: 0>()
----> 1 runtime_drop_bn, scores_drop_bn = get_model_results("drop_bn", model_drop_bn)
      2 

/tmp/ipykernel_11/1822142920.py in get_model_results(model_name, model)
     26         optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_score_]
     27     )
---> 28     history = model.fit(
     29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )

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

InvalidArgumentError: Graph execution error:

Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_3482]

## === cell 24
plot_model_scores(scores_drop_bn)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3214938728.py in <cell line: 0>()
----> 1 plot_model_scores(scores_drop_bn)
      2 

NameError: name 'scores_drop_bn' is not defined

## === cell 25
runtime_tuned, scores_tuned = get_model_results("tuned", model_tuned)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1937299287.py in <cell line: 0>()
----> 1 runtime_tuned, scores_tuned = get_model_results("tuned", model_tuned)
      2 

/tmp/ipykernel_11/1822142920.py in get_model_results(model_name, model)
     26         optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_score_]
     27     )
---> 28     history = model.fit(
     29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )

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

InvalidArgumentError: Graph execution error:

Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
Detected at node decode_image/DecodeImage defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Unknown image file format. One of JPEG, PNG, GIF, BMP required.
	 [[{{node decode_image/DecodeImage}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_5314]

## === cell 26
plot_model_scores(scores_tuned)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/400259038.py in <cell line: 0>()
----> 1 plot_model_scores(scores_tuned)
      2 

NameError: name 'scores_tuned' is not defined

## === cell 27
table = [
    {
        "model": "Base",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_base,
        "train_roc_auc_score": scores_base[0][-1],
        "test_roc_auc_score": scores_base[1][-1],
    },
    {
        "model": "Drop and BN",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_drop_bn,
        "train_roc_auc_score": scores_drop_bn[0][-1],
        "test_roc_auc_score": scores_drop_bn[1][-1],
    },
    {
        "model": "Tuned",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_tuned,
        "train_roc_auc_score": scores_tuned[0][-1],
        "test_roc_auc_score": scores_tuned[1][-1],
    },
]

pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime"], ascending=[False, True]
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3643122313.py in <cell line: 0>()
      3         "model": "Base",
      4         "sample_size": SAMPLE_SIZE,
----> 5         "runtime": runtime_base,
      6         "train_roc_auc_score": scores_base[0][-1],
      7         "test_roc_auc_score": scores_base[1][-1],

NameError: name 'runtime_base' is not defined

## === cell 28
model_tuned_20 = load_model(
    "tuned.keras", custom_objects={"roc_auc_score_": roc_auc_score_}
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4126659689.py in <cell line: 0>()
      1 # Load saved tuned model with custom metric parameter
----> 2 model_tuned_20 = load_model(
      3     "tuned.keras", custom_objects={"roc_auc_score_": roc_auc_score_}
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=tuned.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 29
submis_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".tif")])
submis_data = np.array([os.path.join(test_dir, f) for f in submis_files])

BATCH_SIZE = 64
SUBMIS_BUFFER_SIZE = submis_data.shape[0]

submis_dataset = get_prefetched_data(
    submis_data, BATCH_SIZE, SUBMIS_BUFFER_SIZE, training=False
)



## === cell 30
result_20 = model_tuned_20.predict(submis_dataset, verbose=0).reshape(-1)

result_20 = np.clip(result_20.astype(np.float32), 0.0, 1.0)
result_20[:5], result_20.shape



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1712182194.py in <cell line: 0>()
----> 1 result_20 = model_tuned_20.predict(submis_dataset, verbose=0).reshape(-1)
      2 
      3 result_20 = np.clip(result_20.astype(np.float32), 0.0, 1.0)
      4 result_20[:5], result_20.shape
      5 

NameError: name 'model_tuned_20' is not defined

## === cell 31
id_ = np.char.replace(np.array(submis_files), ".tif", "")
label_ = result_20

table = pd.DataFrame({"id": id_, "label": label_})
table.head()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1632498466.py in <cell line: 0>()
      1 id_ = np.char.replace(np.array(submis_files), ".tif", "")
----> 2 label_ = result_20
      3 
      4 table = pd.DataFrame({"id": id_, "label": label_})
      5 table.head()

NameError: name 'result_20' is not defined

## === cell 32
table



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3406391665.py in <cell line: 0>()
----> 1 table
      2 

NameError: name 'table' is not defined

## === cell 33
table.to_csv("submission_20.csv", index=False)
print("Wrote submission_20.csv with shape:", table.shape)
print("Columns:", list(table.columns))
print("Path exists:", os.path.exists("submission_20.csv"))
print("First rows:\n", table.head())

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2853535095.py in <cell line: 0>()
----> 1 table.to_csv("submission_20.csv", index=False)
      2 print("Wrote submission_20.csv with shape:", table.shape)
      3 print("Columns:", list(table.columns))
      4 print("Path exists:", os.path.exists("submission_20.csv"))
      5 print("First rows:\n", table.head())

NameError: name 'table' is not defined
