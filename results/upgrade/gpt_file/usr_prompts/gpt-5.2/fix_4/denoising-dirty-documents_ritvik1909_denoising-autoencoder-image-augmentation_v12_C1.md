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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.02688

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I remove the dependency on `imgaug` because it’s crashing at import time due to an incompatible protobuf/TensorFlow environment, and replace it with an equivalent augmentation pipeline using TensorFlow ops (same intent: rotations and flips) so `processed_train` is created and training can run. I also fix the zip extraction paths so they reliably extract and list the correct `train/`, `train_cleaned/`, and `test/` folders in `/kaggle/working/` without double slashes. Finally, I keep your autoencoder architecture and training loop intact, but streamline inference to call `autoencoder(test_batch)` directly (same computation) and ensure the submission is written with the required `id,value` columns and `.csv` suffix.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend before importing TensorFlow (this is a common Kaggle env compatibility issue and is score-neutral). Then I fix the augmentation bug: `tf.image.rot90` swaps height/width, causing concatenation to fail; I rotate and then resize back to the configured `(420,540)` so shapes always match while keeping the same augmentation intent. Finally, I keep your model/training loop and submission logic intact, only unblocking the pipeline so it trains, predicts, and writes a valid `submission.csv` with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I fix the immediate runtime crash in the first cell by removing the protobuf-backend environment forcing that’s incompatible with TF 2.18/protobuf 6 in this Kaggle image, so TensorFlow imports cleanly. Then I fix a key logic issue that severely hurts score: the train and train_cleaned images must be paired by identical image id (filenames), but currently they’re loaded by independent sorted directory listings, which can misalign inputs/targets and train the model to map an image to the wrong “clean” label. Finally, I keep your model, training loop, and submission format the same, only ensuring the pairing is correct and the pipeline runs end-to-end to write `submission.csv`.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

for zname in ["train.zip", "test.zip", "train_cleaned.zip", "sampleSubmission.csv.zip"]:
    zpath = os.path.join(path_zip, zname)
    with zipfile.ZipFile(zpath, "r") as zip_ref:
        zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

if not (
    os.path.isdir(train_dir)
    and os.path.isdir(train_cleaned_dir)
    and os.path.isdir(test_dir)
):
    nested_base = os.path.join(path, "denoising-dirty-documents")
    train_dir = os.path.join(nested_base, "train")
    train_cleaned_dir = os.path.join(nested_base, "train_cleaned")
    test_dir = os.path.join(nested_base, "test")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Found:",
    len(train_img),
    "train,",
    len(train_cleaned_img),
    "train_cleaned,",
    len(test_img),
    "test images",
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(train_dir, f)) for f in sorted(os.listdir(train_dir))[:10]
]
print(
    "Median Dimensions:",
    np.median([img.shape[0] for img in imgs]),
    np.median([img.shape[1] for img in imgs]),
)
del imgs




## === cell 3
def process_image(img_path):
    img = cv2.imread(img_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = set(os.listdir(train_dir))
clean_files = set(os.listdir(train_cleaned_dir))
paired_files = sorted(train_files.intersection(clean_files))

if len(paired_files) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned folders."
    )

missing_in_clean = sorted(train_files - clean_files)
missing_in_train = sorted(clean_files - train_files)
if missing_in_clean:
    print(
        f"Warning: {len(missing_in_clean)} files exist in train but not in train_cleaned. They will be ignored."
    )
if missing_in_train:
    print(
        f"Warning: {len(missing_in_train)} files exist in train_cleaned but not in train. They will be ignored."
    )

train = []
train_cleaned = []
test = []

for f in paired_files:
    train.append(process_image(os.path.join(train_dir, f)))
    train_cleaned.append(process_image(os.path.join(train_cleaned_dir, f)))

for f in sorted(os.listdir(test_dir)):
    test.append(process_image(os.path.join(test_dir, f)))

train = np.asarray(train, dtype=np.float32)
train_cleaned = np.asarray(train_cleaned, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)

train_img = paired_files
train_cleaned_img = paired_files
test_img = sorted(os.listdir(test_dir))

print("Loaded paired train set:", train.shape, train_cleaned.shape)
print("Loaded test set:", test.shape)



## === cell 5
train.shape, train_cleaned.shape, test.shape



## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)
plt.show()




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    """
    Bug fix: tf.image.rot90 swaps H/W, which broke np.concatenate because
    original images are (420,540) but rot90 becomes (540,420).
    Keep same augmentation intent but resize back to config.IMG_SIZE so shapes match.
    """
    tf.random.set_seed(seed)
    out = [images]
    x = tf.convert_to_tensor(images, dtype=tf.float32)

    target_hw = config.IMG_SIZE  # (H, W)

    for step in pipeline:
        if step == "rot90":
            aug = tf.image.rot90(x, k=1)
            aug = tf.image.resize(aug, target_hw, method="bilinear")
        elif step == "rot180":
            aug = tf.image.rot90(x, k=2)
        elif step == "rot270":
            aug = tf.image.rot90(x, k=3)
            aug = tf.image.resize(aug, target_hw, method="bilinear")
        elif step == "hflip":
            aug = tf.image.flip_left_right(x)
        elif step == "vflip":
            aug = tf.image.flip_up_down(x)
        else:
            raise ValueError(f"Unknown augmentation step: {step}")

        out.append(aug.numpy())

    return np.concatenate(out, axis=0).astype(np.float32)




## === cell 8
pipeline = ["rot90", "rot180", "rot270", "hflip", "vflip"]



## === cell 9
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 10
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)



## === cell 11
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=12,
    verbose=2,
)



## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history



## === cell 13
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 14
decoded_imgs = autoencoder(train[:4], training=False).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)
plt.show()

del decoded_imgs



## === cell 15
ids = []
vals = []

for i, f in tqdm(list(enumerate(test_img)), total=len(test_img)):
    file = os.path.join(test_dir, f)
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)  # original size for id layout
    img_shape = img.shape

    decoded_img = np.squeeze(autoencoder(test[i : i + 1], training=False).numpy())
    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    for r in range(img_shape[0]):
        base = f"{imgid}_{r+1}_"
        row = preds_reshaped[r]
        for c in range(img_shape[1]):
            ids.append(base + str(c + 1))
            vals.append(float(row[c]))

print("Length of IDs:", len(ids))

sub = pd.DataFrame({"id": ids, "value": vals})
sub.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
print(sub.head())
