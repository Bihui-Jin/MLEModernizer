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

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 5. Target score

0.9905

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the problematic protobuf import, add a safe‑file check in the `ImageSequence` so missing images are replaced with blank arrays (preventing crashes), and ensure the submission CSV is written to the working directory. These fixes let the notebook run end‑to‑end and produce a valid `submission.csv` while keeping the original model and training logic unchanged.'
- What this solution (achieved 0.5) has done: 'We add a protobuf‑compatibility setting before importing TensorFlow to stop the `MessageFactory` error, and we enable fine‑tuning of the pretrained VGG16 backbone (set `trainable=True`) so the model can learn from the 32×32 cactus images, which should raise the AUC toward the target while preserving the original architecture.'
- What this solution (achieved 0.5) has done: 'The fix restores proper label types, gives the model more training time, and ensures the test sequence is label‑free so predictions are correct. These changes keep the original architecture while improving learning stability and moving the AUC toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import Sequence



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = pathlib.Path("/kaggle/input/aerial-cactus-identification")

train_df = pd.read_csv(BASE_PATH / "train.csv")
sample_submission = pd.read_csv(BASE_PATH / "sample_submission.csv")

train_dir = BASE_PATH / "train"
test_dir = BASE_PATH / "test"

train_df["has_cactus"] = train_df["has_cactus"].astype(int)




## === cell 2
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["has_cactus"],
)





## === cell 3
class ImageSequence(Sequence):
    """Keras Sequence that loads 32x32 JPG images from a directory.
    Missing files are replaced with a zero array to avoid crashes."""

    def __init__(self, df, img_dir, batch_size=64, shuffle=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = pathlib.Path(img_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
        self.datagen = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                horizontal_flip=True,
                shear_range=0.2,
                zoom_range=0.2,
            )
            if augment
            else tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        )

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        imgs = []
        for fname in batch_df["id"]:
            img_path = self.img_dir / fname
            if img_path.is_file():
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=(32, 32)
                )
                img = tf.keras.preprocessing.image.img_to_array(img)
            else:
                img = np.zeros((32, 32, 3), dtype=np.float32)
            imgs.append(img)
        x = np.stack(imgs, axis=0)
        x = self.datagen.standardize(x)

        if "has_cactus" in batch_df.columns:
            y = batch_df["has_cactus"].astype(float).values.reshape(-1, 1)
            return x, y
        else:
            return x

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)




## === cell 4
train_seq = ImageSequence(
    train_split, train_dir, batch_size=64, shuffle=True, augment=True
)
val_seq = ImageSequence(
    val_split, train_dir, batch_size=64, shuffle=False, augment=False
)




## === cell 5
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = True

    model = models.Sequential(
        [
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )




## === cell 6
model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)




## === cell 7
early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=10,  # longer patience to allow more learning
    restore_best_weights=True,
)

history = model.fit(
    train_seq,
    epochs=50,  # more epochs for better convergence
    validation_data=val_seq,
    callbacks=[early_stop],
    verbose=2,
)




## === cell 8
test_df = sample_submission[["id"]].copy()

test_seq = ImageSequence(test_df, test_dir, batch_size=64, shuffle=False, augment=False)

preds = model.predict(test_seq, verbose=2)
sample_submission["has_cactus"] = preds.ravel()

submission_path = "/kaggle/working/submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 9
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["auc"], label="train_auc")
plt.plot(history.history["val_auc"], label="val_auc")
plt.title("AUC")
plt.legend()

plt.show()
