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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow as tf
    import tensorflow.keras as keras
except Exception as e:
    tf = None
    keras = None
    print("TensorFlow import failed:", e)

if tf is not None:
    try:
        tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
        tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
        tf.config.optimizer.set_jit(True)
        tf.random.set_seed(42)
    except Exception as cfg_err:
        print("TensorFlow config error (ignored):", cfg_err)

seed = 42
np.random.seed(seed)
if tf is not None:
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
random.seed(seed)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)

class_counts = labels.sum(axis=0)
class_priors = (class_counts / len(train)).values  # shape (num_classes,)



## === cell 2
h_target, w_target = 256, 256
batch_size = 32
num_classes = labels.shape[1]
workers = os.cpu_count()  # use all CPUs for I/O



## === cell 3
test_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
test_image_paths = [
    os.path.join("../input/plant-pathology-2021-fgvc8/test_images", fname)
    for fname in test_df["image"]
]


def load_and_preprocess(paths):
    imgs = []
    for p in paths:
        img = keras.preprocessing.image.load_img(p, target_size=(h_target, w_target))
        arr = keras.preprocessing.image.img_to_array(img) / 255.0
        imgs.append(arr)
    return np.stack(imgs)


test_images = load_and_preprocess(test_image_paths)  # shape (num_test, h, w, 3)



## === cell 4
model_path = "./EffnetB4.h5"

if tf is not None:
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
    else:
        base = keras.applications.EfficientNetB4(
            include_top=False, input_shape=(h_target, w_target, 3), weights="imagenet"
        )
        base.trainable = False
        x = keras.layers.GlobalAveragePooling2D()(base.output)
        output = keras.layers.Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs=base.input, outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy")
else:
    model = None



## === cell 5
if tf is not None and model is not None and not os.path.exists(model_path):
    train_df = train.copy()
    for cls in mlb.classes_:
        train_df[cls] = labels[cls]

    train_image_paths = [
        os.path.join("../input/plant-pathology-2021-fgvc8/train_images", fname)
        for fname in train_df["image"]
    ]
    train_images = load_and_preprocess(train_image_paths)  # (N, h, w, 3)

    train_labels = train_df[mlb.classes_].values.astype(np.float32)  # (N, num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(
        buffer_size=len(train_images), seed=seed, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = max(1, len(train_images) // batch_size)

    model.fit(
        train_dataset,
        epochs=3,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )
    try:
        model.save(model_path)
    except Exception:
        pass



## === cell 6
if tf is not None and model is not None:
    preds = model.predict(
        test_images,
        batch_size=batch_size,
        verbose=1,
    )
else:
    preds = np.tile(class_priors, (len(test_df), 1))



## === cell 7
thresh = {
    "complex": 0.37,
    "frog_eye_leaf_spot": 0.41,
    "healthy": 0.25,
    "powdery_mildew": 0.17,
    "rust": 0.36,
    "scab": 0.57,
}
default_thr = 0.5

submission_labels = []
for prob_vec in preds:
    chosen = []
    for idx, cls in enumerate(mlb.classes_):
        thr = thresh.get(cls, default_thr)
        if prob_vec[idx] >= thr:
            chosen.append(cls)
    if not chosen:
        chosen = [mlb.classes_[np.argmax(prob_vec)]]
    submission_labels.append(" ".join(chosen))

test_df["labels"] = submission_labels



## === cell 8
output_path = "submission.csv"
test_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
