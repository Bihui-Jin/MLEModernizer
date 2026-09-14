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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import tensorflow as tf
from tensorflow import keras

print(os.listdir("../input"))




## === cell 1
train_data = pd.read_csv("../input/train.csv")




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
train_imgs_path = "../input/train/train/"
train_ids = train_data.id.tolist()
train_images = (
    np.stack(
        [
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(train_imgs_path, img_id), target_size=(32, 32)
                )
            )
            for img_id in train_ids
        ],
        axis=0,
    ).astype("float32")
    / 255.0
)
train_labels = train_data.has_cactus.values.reshape(-1, 1).astype("float32")




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), input_shape=(32, 32, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(32, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(64, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(64, (5, 5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(128, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(128, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Conv2D(512, (3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))




## === cell 14
model.summary()




## === cell 15
opt = keras.optimizers.Adam(1e-4)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])




## === cell 16
def image_generator(batch_size=64, all_data=True, train=True):
    """Yield batches from the pre‑loaded numpy arrays."""
    X = train_images
    y = train_labels
    if not train:
        X = X[15000:]
        y = y[15000:]
    while True:
        indexes = np.arange(X.shape[0])
        if train:
            if all_data:
                np.random.shuffle(indexes)
            else:
                indexes = np.arange(min(15000, X.shape[0]))
                np.random.shuffle(indexes)
        N = int(len(indexes) / batch_size)
        for i in range(N):
            batch_idx = indexes[i * batch_size : (i + 1) * batch_size]
            yield X[batch_idx], y[batch_idx]




## === cell 17
steps = int(train_images.shape[0] / 64)
model.fit(image_generator(), steps_per_epoch=steps, epochs=5, verbose=2)




## === cell 18
val_steps = int(train_images[15000:].shape[0] / 64)
model.evaluate(image_generator(train=False), steps=val_steps, verbose=2)




## === cell 19
test_dir = "../input/test/test/"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
len(test_files)




## === cell 20
test_images = (
    np.stack(
        [
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    os.path.join(test_dir, fname), target_size=(32, 32)
                )
            )
            for fname in test_files
        ],
        axis=0,
    ).astype("float32")
    / 255.0
)

batch = 40
all_preds = []
for start in range(0, test_images.shape[0], batch):
    batch_imgs = test_images[start : start + batch]
    preds = model.predict(batch_imgs, verbose=0)
    all_preds.extend(preds.reshape(-1).tolist())




## === cell 21
assert len(all_preds) == len(test_files), "Prediction length mismatch"




## === cell 22
sub_file = pd.DataFrame({"id": test_files, "has_cactus": all_preds})
sub_file.head()




## === cell 23
sub_file.to_csv("sample_submission.csv", index=False)
