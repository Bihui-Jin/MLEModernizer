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

0.03077

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.47318) has done: 'I fix the immediate import/runtime failure by removing the unused `imgaug` dependency that is incompatible in this environment (it triggers the protobuf `MessageFactory.GetPrototype` error), and replace the augmentation step with a minimal, equivalent geometric augmentation implemented via NumPy (rotations/flips) so the pipeline still expands the dataset without changing the model/training loop. I also ensure image/pair alignment by building train/train_cleaned using the intersection of filenames (sorted) to avoid any accidental mismatch that can severely hurt RMSE. Finally, I keep the autoencoder architecture and training semantics intact, but fix the inference call to avoid mixing `.numpy()` inside Keras calls and guarantee values are clipped to `[0,1]` before writing a valid `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'I fix the two runtime blockers: (1) the protobuf-related import crash by removing the unused plotting/seaborn imports that pull in incompatible extras in this environment, and (2) the augmentation shape-mismatch by ensuring rotations keep the (420,540) shape via resize-back after `np.rot90`. These are execution/correctness fixes and are score-positive because the model actually train on correctly paired and consistently-shaped augmented data instead of failing or learning from mis-shaped arrays. I also make the submission writing robust and fast by vectorizing the pixel “melt” using the provided `sampleSubmission.csv` ids (guaranteeing exact row count/order and correct formatting) while keeping the same prediction values. Core model architecture, loss, and training loop are preserved.'

# 9. Code solution

## === cell 0
import os, zipfile
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks


np.random.seed(19)
tf.random.set_seed(19)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

with zipfile.ZipFile(path_zip + "train.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "test.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "train_cleaned.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
    zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".png")])
train_cleaned_img = sorted(
    [f for f in os.listdir(train_cleaned_dir) if f.lower().endswith(".png")]
)
test_img = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".png")])

common = sorted(set(train_img).intersection(set(train_cleaned_img)))
if len(common) == 0:
    raise RuntimeError(
        "No overlapping filenames between train and train_cleaned. Check extracted paths."
    )
train_img = common
train_cleaned_img = common

print(
    "Train images:",
    len(train_img),
    "Train cleaned images:",
    len(train_cleaned_img),
    "Test images:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(os.path.join(train_dir, f)) for f in train_img[:10]]
print("Sample Dimensions (H,W):", [img.shape[:2] for img in imgs if img is not None])
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path_}")
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train = []
train_cleaned = []
test = []

for f in tqdm(train_img, desc="Loading train"):
    train.append(process_image(os.path.join(train_dir, f)))

for f in tqdm(train_cleaned_img, desc="Loading train_cleaned"):
    train_cleaned.append(process_image(os.path.join(train_cleaned_dir, f)))

for f in tqdm(test_img, desc="Loading test"):
    test.append(process_image(os.path.join(test_dir, f)))

train = np.asarray(train, dtype=np.float32)
train_cleaned = np.asarray(train_cleaned, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)



## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
def _resize_back(batch_hw1):
    out = np.empty(
        (batch_hw1.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1], 1),
        dtype=batch_hw1.dtype,
    )
    for i in range(batch_hw1.shape[0]):
        out[i, ..., 0] = cv2.resize(
            batch_hw1[i, ..., 0],
            (config.IMG_SIZE[1], config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def augment_pipeline(pipeline, images, seed=19):
    np.random.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = step(images)
        processed_images = np.concatenate([processed_images, temp], axis=0)
    return processed_images


def aug_rot90(x):
    y = np.rot90(x, k=1, axes=(1, 2)).copy()
    return _resize_back(y)


def aug_rot180(x):
    y = np.rot90(x, k=2, axes=(1, 2)).copy()
    return y  # same H/W


def aug_rot270(x):
    y = np.rot90(x, k=3, axes=(1, 2)).copy()
    return _resize_back(y)


def aug_hflip(x):
    return np.flip(x, axis=2).copy()


def aug_vflip(x):
    return np.flip(x, axis=1).copy()


pipeline = [aug_rot90, aug_rot180, aug_rot270, aug_hflip, aug_vflip]



## === cell 7
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

print(processed_train.shape, processed_train_cleaned.shape)




## === cell 8
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



## === cell 9
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)
rlp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.8, patience=5, min_lr=1e-6, mode="min", verbose=1
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es, rlp],
    epochs=500,
    batch_size=12,
)



## === cell 10
print("Final training loss:", float(history.history["loss"][-1]))



## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 12
decoded_imgs = autoencoder(train[:4], training=False).numpy()
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)
print(
    "Sanity decoded batch:", decoded_imgs.shape, decoded_imgs.min(), decoded_imgs.max()
)
del decoded_imgs



## === cell 13
sample_path = os.path.join(path, "sampleSubmission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/denoising-dirty-documents/sampleSubmission.csv"

sample_sub = pd.read_csv(sample_path)
ids = sample_sub["id"].values

img_ids = np.array([int(s.split("_", 1)[0]) for s in ids], dtype=np.int32)

pred_by_imgid = {}
for i, f in tqdm(
    list(enumerate(test_img)), total=len(test_img), desc="Predicting test"
):
    file = os.path.join(test_dir, f)
    imgid = int(f[:-4])

    raw = cv2.imread(file, 0)
    if raw is None:
        raise RuntimeError(f"Failed to read test image: {file}")
    h, w = raw.shape

    decoded = autoencoder(test[i : i + 1], training=False).numpy()
    decoded = np.squeeze(decoded, axis=0)  # (420,540,1)
    decoded = np.squeeze(decoded, axis=-1)  # (420,540)
    decoded = np.clip(decoded, 0.0, 1.0)

    resized = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_by_imgid[imgid] = resized.reshape(-1).astype(np.float32)

values = np.empty(len(ids), dtype=np.float32)
for imgid in sorted(pred_by_imgid.keys()):
    mask = img_ids == imgid
    values[mask] = pred_by_imgid[imgid]

sub = pd.DataFrame({"id": ids, "value": values.astype(np.float32)})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.head())
