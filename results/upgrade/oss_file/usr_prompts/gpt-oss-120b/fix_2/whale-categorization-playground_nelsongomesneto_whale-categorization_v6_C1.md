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
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.00128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from heapq import heappush, heappop
from sklearn.model_selection import train_test_split

folder = "../input/whale-categorization-playground/"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(os.path.join(folder, "train.csv"))

id_dict = {}
for idx, row in train_df.iterrows():
    label = row["Id"]
    if label not in id_dict:
        id_dict[label] = len(id_dict)
    train_df.at[idx, "Image"] = os.path.join(folder, "train", row["Image"])

train_df["LabelIdx"] = train_df["Id"].map(id_dict)

test_files = [
    os.path.join(folder, "test", f)
    for f in sorted(os.listdir(os.path.join(folder, "test")))
]

x_train_paths = train_df["Image"].tolist()
y_train_idxs = train_df["LabelIdx"].tolist()

width, height = 150, 150
batch_size = 256
epochs_per_batch = 5
iterations = 1
num_classes = len(id_dict)



## === cell 2
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(2, 2),
            padding="same",
            activation="relu",
            input_shape=(width, height, 3),
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(5, 5), padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(5, 5)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 3
for it in range(iterations):
    indices = list(range(len(x_train_paths)))
    random.shuffle(indices)
    start_time = time.time()
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        images, labels = [], []
        for idx in batch_idx:
            img_path = x_train_paths[idx]
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(width, height)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)

            one_hot = np.zeros(num_classes)
            one_hot[y_train_idxs[idx]] = 1
            labels.append(one_hot)

        x_batch = np.array(images)
        y_batch = np.array(labels)
        model.fit(
            x_batch, y_batch, epochs=epochs_per_batch, verbose=False, shuffle=True
        )
        print(
            f"\tBatch {start + len(batch_idx)} / {len(x_train_paths)} "
            f"({100 * (start + len(batch_idx)) / len(x_train_paths):.1f}%)"
        )
    print(
        f"Iteration {it + 1}/{iterations} completed in "
        f"{time.time() - start_time:.1f}s"
    )



## === cell 4
y_final = []
for idx, img_path in enumerate(test_files):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(width, height))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    preds = model.predict(np.array([img_arr]), verbose=False)[0]

    top5_idx = np.argsort(-preds)[:5]
    top5_labels = [
        list(id_dict.keys())[list(id_dict.values()).index(i)] for i in top5_idx
    ]
    y_final.append(" ".join(top5_labels))

    if idx % 500 == 0:
        print(f"Processed {idx}/{len(test_files)} images")

image_ids = [os.path.basename(p) for p in test_files]
submission = pd.DataFrame({"Image": image_ids, "Id": y_final})
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_57/3833390466.py in <cell line: 0>()
      2 y_final = []
      3 for idx, img_path in enumerate(test_files):
----> 4     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(width, height))
      5     img_arr = tf.keras.preprocessing.image.img_to_array(img)
      6     preds = model.predict(np.array([img_arr]), verbose=False)[0]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/whale-categorization-playground/test/test'

## === cell 5
submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
