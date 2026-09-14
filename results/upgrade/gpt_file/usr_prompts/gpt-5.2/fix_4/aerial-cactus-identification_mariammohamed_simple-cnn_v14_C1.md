# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

print(os.listdir("../input"))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

import tensorflow as tf
from tensorflow import keras

BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
import random

random.seed(SEED)



## === cell 3
train_data = pd.read_csv(TRAIN_CSV)



## === cell 4
train_data.shape



## === cell 5
train_data.head()



## === cell 6
train_data.has_cactus.unique()



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
positive_examples = train_data[train_data.has_cactus == 1]
negative_examples = train_data[train_data.has_cactus == 0]



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(32, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(64, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(64, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(128, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Conv2D(512, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Dense(1, activation="sigmoid"))



## === cell 16
model.summary()



## === cell 17
opt = keras.optimizers.Adam(0.0001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])



## === cell 18
train_data.shape[0]



## === cell 19
train_ids = train_data["id"].values
y_all = train_data["has_cactus"].values.astype(np.float32).reshape(-1, 1)

X_all = np.empty((len(train_ids), 32, 32, 3), dtype=np.uint8)
for i, fid in enumerate(train_ids):
    X_all[i] = mpimg.imread(os.path.join(TRAIN_DIR, fid))

assert X_all.shape[0] == y_all.shape[0] == train_data.shape[0]




## === cell 20
def image_generator(
    batch_size=64, all_data=True, shuffle=True, train=True, indexes=None
):
    while True:
        if indexes is None:
            if train:
                if all_data:
                    indexes = np.arange(train_data.shape[0])
                else:
                    indexes = np.arange(train_data[:15000].shape[0])
                if shuffle:
                    np.random.shuffle(indexes)
            else:
                indexes = np.arange(15000, train_data.shape[0])
        else:
            indexes = np.array(indexes)

        N = int(len(indexes) / batch_size)
        if N <= 0:
            N = 1

        for i in range(N):
            current_indexes = indexes[i * batch_size : (i + 1) * batch_size]
            if len(current_indexes) == 0:
                continue

            batch_input = X_all[current_indexes]
            batch_output = y_all[current_indexes]
            yield (batch_input, batch_output)




## === cell 21
steps_per_epoch = int(train_data.shape[0] / 64)
model.fit(image_generator(), steps_per_epoch=steps_per_epoch, epochs=25, verbose=2)



## === cell 22
val_steps = int((train_data.shape[0] - 15000) / 64)
if val_steps <= 0:
    val_steps = 1
model.evaluate(image_generator(train=False), steps=val_steps, verbose=2)



## === cell 23
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
len(test_files), test_files[:5]



## === cell 24
batch = 256
all_out = np.empty((len(test_files),), dtype=np.float32)

for start in range(0, len(test_files), batch):
    end = min(start + batch, len(test_files))
    batch_files = test_files[start:end]
    images = np.empty((len(batch_files), 32, 32, 3), dtype=np.uint8)
    for j, fname in enumerate(batch_files):
        images[j] = mpimg.imread(os.path.join(TEST_DIR, fname))
    out = model.predict(images, verbose=0).reshape(-1).astype(np.float32)
    all_out[start:end] = out

assert len(all_out) == len(test_files), (len(all_out), len(test_files))



## === cell 25
sub_file = pd.DataFrame({"id": test_files, "has_cactus": all_out.astype(float)})
sub_file.head()



## === cell 26
sub_path = "submission.csv"
sub_file.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub_file))
print(sub_file.columns.tolist())
