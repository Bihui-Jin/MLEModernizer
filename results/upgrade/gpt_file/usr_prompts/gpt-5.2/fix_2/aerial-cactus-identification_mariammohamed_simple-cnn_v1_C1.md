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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

import tf_keras as keras

BASE_INPUT = "../input/aerial-cactus-identification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input"
print("Using BASE_INPUT:", BASE_INPUT)
print("BASE_INPUT contents:", os.listdir(BASE_INPUT)[:20])



## === cell 1
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_data = pd.read_csv(train_csv_path)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.head()



## === cell 4
train_data.has_cactus.unique()



## === cell 5
train_data.has_cactus.hist()



## === cell 6
train_data.has_cactus.value_counts()



## === cell 7
train_data.has_cactus.plot()



## === cell 8
positive_examples = train_data[train_data.has_cactus == 1]
negative_examples = train_data[train_data.has_cactus == 0]



## === cell 9
train_img_dir = os.path.join(BASE_INPUT, "train", "train")
img = mpimg.imread(os.path.join(train_img_dir, positive_examples.id.tolist()[2]))
plt.imshow(img)
plt.axis("off")



## === cell 10
img = mpimg.imread(os.path.join(train_img_dir, negative_examples.id.tolist()[2]))
plt.imshow(img)
plt.axis("off")



## === cell 11
img.shape



## === cell 12
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(32, (5, 5), activation="relu"))
model.add(keras.layers.Conv2D(64, (5, 5), activation="relu"))
model.add(keras.layers.Conv2D(64, (5, 5), activation="relu"))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.Conv2D(256, (3, 3), activation="relu"))
model.add(keras.layers.Conv2D(256, (3, 3), activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))



## === cell 13
model.summary()



## === cell 14
opt = keras.optimizers.Adam(0.0001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])



## === cell 15
train_data.shape[0]




## === cell 16
def image_generator(batch_size=64, train=True, split_index=15000):
    split_index = int(min(max(split_index, 1), len(train_data) - 1))
    while True:
        if train:
            indexes = np.arange(split_index)
        else:
            indexes = np.arange(split_index, len(train_data))
        np.random.shuffle(indexes)
        N = int(len(indexes) / batch_size)

        for i in range(N):
            current_indexes = indexes[i * batch_size : (i + 1) * batch_size]
            batch_input = []
            batch_output = []
            for idx in current_indexes:
                img_id = train_data.iloc[idx]["id"]
                batch_input.append(mpimg.imread(os.path.join(train_img_dir, img_id)))
                batch_output.append(train_data.iloc[idx]["has_cactus"])

            batch_input = np.asarray(batch_input, dtype=np.float32)
            batch_output = np.asarray(batch_output, dtype=np.float32).reshape(-1, 1)
            yield batch_input, batch_output




## === cell 17
steps_per_epoch = int(train_data.shape[0] / 64)
model.fit(image_generator(), steps_per_epoch=steps_per_epoch, epochs=5)



## === cell 18
val_steps = int(
    max(1, (train_data.shape[0] - min(15000, train_data.shape[0] - 1)) / 64)
)
model.evaluate(image_generator(train=False), steps=val_steps)



## === cell 19
test_img_dir = os.path.join(BASE_INPUT, "test", "test")
print("Test dir:", test_img_dir)
print("Num test entries (including any subdirs):", len(os.listdir(test_img_dir)))



## === cell 20
test_files = sorted(
    [
        f
        for f in os.listdir(test_img_dir)
        if os.path.isfile(os.path.join(test_img_dir, f))
    ]
)
len(test_files), test_files[:5]



## === cell 21
len(test_files)



## === cell 22
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

sample_ids = sample_sub["id"].tolist()
id_set = set(test_files)
ordered_test_ids = [i for i in sample_ids if i in id_set]

batch = 40
preds = []

for start in range(0, len(ordered_test_ids), batch):
    batch_ids = ordered_test_ids[start : start + batch]
    images = [mpimg.imread(os.path.join(test_img_dir, img_id)) for img_id in batch_ids]
    out = model.predict(np.asarray(images, dtype=np.float32), verbose=0)
    preds.append(out.reshape(-1))

all_out = np.concatenate(preds, axis=0).reshape(-1, 1)
all_out.shape



## === cell 23
all_out = np.array(all_out).reshape(-1, 1)



## === cell 24
all_out.shape



## === cell 25
sub_file = pd.DataFrame(
    data={"id": ordered_test_ids, "has_cactus": all_out.reshape(-1).tolist()}
)
sub_file.head()



## === cell 26
sub_file.head()



## === cell 27
sub_path = "submission.csv"
sub_file.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub_file))
print(sub_file.head())
