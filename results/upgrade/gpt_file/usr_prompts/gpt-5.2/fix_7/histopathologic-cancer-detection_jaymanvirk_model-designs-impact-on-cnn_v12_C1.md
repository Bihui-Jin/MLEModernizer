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

0.90131

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.90131) has done: 'I fix the protobuf crash by not forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides (this error happens before anything runs). Then I fix the tf.data `py_function` path decoding bug by converting the incoming tensor to bytes via `.numpy()` inside the Python callback. Finally, I make the training cell return the trained model (so submission generation doesn’t fail when the `.h5` wasn’t created due to earlier errors) and write `submission.csv` with the required `id,label` columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

tf.random.set_seed(0)
np.random.seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = "/kaggle/input/histopathologic-cancer-detection"

train_labels_path = os.path.join(input_dir, "train_labels.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")

train_dir = os.path.join(input_dir, "train") + os.sep
test_dir = os.path.join(input_dir, "test") + os.sep

assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

sample_data = pd.read_csv(sample_sub_path)
train_data = pd.read_csv(train_labels_path)

sample_data.head(), train_data.head(), train_dir, test_dir




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
print_short_summary("Sample submission", sample_data)



## === cell 4
pass



## === cell 5
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer,
    replace=False,
    n_samples=len(cancer),
    random_state=0,
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)

balanced_train_data["label"].value_counts()



## === cell 6
image_paths = (train_dir + balanced_train_data["id"] + ".tif").values
labels = balanced_train_data["label"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.25,
    shuffle=True,
    random_state=0,
    stratify=labels,
)

len(X_train), len(X_test)



## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_rgba_tf(image_path):
    def _py_decode(path_tensor):
        path_bytes = path_tensor.numpy()
        if isinstance(path_bytes, (np.ndarray,)):
            path_bytes = path_bytes.item()
        if isinstance(path_bytes, str):
            path = path_bytes
        else:
            path = path_bytes.decode("utf-8")

        with Image.open(path) as im:
            im = im.convert("RGBA")
            im = im.resize((32, 32), resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        return arr

    img = tf.py_function(_py_decode, [image_path], Tout=tf.uint8)
    img.set_shape([32, 32, 4])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_decoded_image(image_path, label=None):
    img = _decode_resize_rgba_tf(image_path)
    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size, cache=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    if isinstance(data, (tuple, list)) and len(data) == 2:
        paths, labs = data
        dataset = tf.data.Dataset.from_tensor_slices((paths, labs))
        dataset = dataset.with_options(opts)
        dataset = dataset.map(
            get_decoded_image, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.with_options(opts)
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 8
BATCH_SIZE = 128
train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE, cache=True)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE, cache=True)




## === cell 9
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


def get_model_base_deep():
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


def get_model_base_wide():
    model_drop_bn = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model_drop_bn


def get_model_base_maxpool():
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


def get_model_base_dropout():
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




## === cell 10
def get_compiled_model(func):
    gpus = tf.config.experimental.list_physical_devices("GPU")
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


def plot_model_scores(scores, model_name):
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 6))
    plt.plot(epochs, train_scores, label="Train score")
    plt.plot(epochs, test_scores, label="Test score")
    plt.title("Train and test ROC AUC scores of the {}".format(model_name))
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_model_results(model_name, model_func):
    model = get_compiled_model(model_func)

    st = time.time()
    history = model.fit(
        train_dataset, epochs=5, validation_data=test_dataset, verbose=2
    )
    runtime = time.time() - st

    model.save("{}.h5".format(model_name))

    train_scores = history.history["auc"]
    test_scores = history.history["val_auc"]

    return model, (runtime, (train_scores, test_scores))




## === cell 11
model_base_maxpool, (runtime_base_maxpool, scores_base_maxpool) = get_model_results(
    "model_base_maxpool", get_model_base_maxpool
)
plot_model_scores(scores_base_maxpool, "base + max pooling model")



## === cell 12
results = [
    ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),
]

table = []
for i in range(len(results)):
    tmp = {
        "model": results[i][0],
        "runtime": results[i][1],
        "train_roc_auc_score": results[i][2][0][-1],
        "test_roc_auc_score": results[i][2][1][-1],
    }
    table.append(tmp)

leaderboard = pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime"], ascending=[False, True]
)
leaderboard



## === cell 13
model_path = "model_base_maxpool.h5"
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
else:
    model = model_base_maxpool

submis_paths = (test_dir + sample_data["id"] + ".tif").values
submis_dataset = get_prefetched_data(submis_paths, BATCH_SIZE, cache=True)

pred = model.predict(submis_dataset, verbose=1).reshape(-1).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_data["id"].values, "label": pred})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape
