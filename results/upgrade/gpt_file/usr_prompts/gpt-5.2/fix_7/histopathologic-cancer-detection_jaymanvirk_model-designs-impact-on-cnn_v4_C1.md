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

0.86944

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.88173) has done: 'We fix two blockers that prevent the notebook from running: the protobuf `MessageFactory.GetPrototype` crash (by not forcing the pure-Python protobuf implementation), and the XLA/JIT incompatibility caused by using `tf.py_function`/`tf.numpy_function` inside the ROC-AUC metric (by switching to TensorFlow’s built-in `AUC` metric). Then we ensure the training cells actually produce and save `tuned.keras`, and keep test prediction ordering deterministic so ids align with predictions. Finally, we generate a valid `submission_20.csv` with the required `id,label` columns and the same row count as `sample_submission.csv`.'
- What this solution (achieved 0.86944) has done: 'I fix the protobuf crash that prevents TensorFlow from importing by ensuring we do not force an incompatible protobuf implementation and by importing TensorFlow after a safe environment setup. Because your current score (0.88173) is far above the target (0.4964), I not change the model, training, data pipeline, or prediction logic, so the score should remain essentially unchanged (or only differ by negligible floating-point effects). I also adjust cell numbering to start at 1 (Kaggle-style) while preserving the original cell order/content. Finally, I keep the submission writing intact and ensure the `.csv` file is produced with the required `id,label` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
def _load_tif_with_pil(path_bytes):
    path = path_bytes.decode("utf-8")
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((32, 32), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (32,32,3)
        alpha = np.ones((32, 32, 1), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=-1)  # (32,32,4)
    return arr


def get_decoded_image(image_path, label=None):
    img = tf.numpy_function(_load_tif_with_pil, [image_path], Tout=tf.float32)
    img.set_shape((32, 32, 4))
    return img if label is None else (img, tf.cast(label, tf.float32))


def get_prefetched_data(data, batch_size, buffer_size, training=True):
    """
    Create a TensorFlow dataset from image paths (and optional labels).
    If training=True, shuffle; if not, keep deterministic order for submission alignment.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
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
roc_auc_metric = tf.keras.metrics.AUC(curve="ROC", name="roc_auc_score_")



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
        optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_metric]
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



## === cell 22
plot_model_scores(scores_base)



## === cell 23
runtime_drop_bn, scores_drop_bn = get_model_results("drop_bn", model_drop_bn)



## === cell 24
plot_model_scores(scores_drop_bn)



## === cell 25
runtime_tuned, scores_tuned = get_model_results("tuned", model_tuned)



## === cell 26
plot_model_scores(scores_tuned)



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



## === cell 28
model_tuned_20 = load_model("tuned.keras", compile=False)



## === cell 29
submis_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".tif")])
submis_data = np.array([os.path.join(test_dir, f) for f in submis_files])

BATCH_SIZE = 64
SUBMIS_BUFFER_SIZE = submis_data.shape[0]

submis_dataset = get_prefetched_data(
    submis_data, BATCH_SIZE, SUBMIS_BUFFER_SIZE, training=False
)

(len(submis_files), submis_data.shape)



## === cell 30
result_20 = model_tuned_20.predict(submis_dataset, verbose=0).reshape(-1)
result_20 = np.clip(result_20.astype(np.float32), 0.0, 1.0)
result_20[:5], result_20.shape



## === cell 31
id_ = np.char.replace(np.array(submis_files), ".tif", "")
label_ = result_20

pred_table = pd.DataFrame({"id": id_, "label": label_})

submission = sample_data[["id"]].merge(pred_table, on="id", how="left")

submission["label"] = submission["label"].astype(np.float32)
submission["label"] = submission["label"].fillna(np.float32(0.5))

submission.to_csv("submission_20.csv", index=False)

print("Wrote submission_20.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
print("Path exists:", os.path.exists("submission_20.csv"))
print("First rows:\n", submission.head())
print("Any NaNs in label:", submission["label"].isna().any())
print(
    "Label min/max:", float(submission["label"].min()), float(submission["label"].max())
)
