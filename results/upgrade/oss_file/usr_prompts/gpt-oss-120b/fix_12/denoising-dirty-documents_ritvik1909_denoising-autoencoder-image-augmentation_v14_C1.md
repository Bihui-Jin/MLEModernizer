# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

try:
    import seaborn as sns

    sns.set_style("darkgrid")
except Exception:
    pass

np.random.seed(42)
tf.random.set_seed(42)




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

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)  # (height, width)




## === cell 3
imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 4
def process_image(p):
    img = cv2.imread(p)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])  # width, height
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 5
from concurrent.futures import ThreadPoolExecutor


def load_folder(folder):
    files = sorted(os.listdir(folder))
    paths = [folder + f for f in files]
    with ThreadPoolExecutor() as ex:
        imgs = list(
            tqdm(
                ex.map(process_image, paths), total=len(paths), desc=f"Loading {folder}"
            )
        )
    return np.asarray(imgs), files


train, _ = load_folder(path + "train/")
train_cleaned, _ = load_folder(path + "train_cleaned/")
test, _ = load_folder(path + "test/")




## === cell 6
def _tf_resize(imgs):
    return tf.image.resize(imgs, config.IMG_SIZE, method="bilinear")


def _make_tf_step(func):
    def step(imgs):
        transformed = func(imgs)
        resized = _tf_resize(transformed)
        return resized

    return step


pipeline = [
    _make_tf_step(lambda x: tf.image.rot90(x, k=1)),
    _make_tf_step(lambda x: tf.image.rot90(x, k=2)),
    _make_tf_step(lambda x: tf.image.rot90(x, k=3)),
    _make_tf_step(tf.image.flip_left_right),
    _make_tf_step(tf.image.flip_up_down),
]


def augment_pair(pipeline, noisy_imgs, clean_imgs, seed=19):
    tf.random.set_seed(seed)
    aug_noisy = [noisy_imgs]
    aug_clean = [clean_imgs]
    for step in pipeline:
        aug_noisy.append(step(noisy_imgs))
        aug_clean.append(step(clean_imgs))
    return tf.concat(aug_noisy, axis=0).numpy(), tf.concat(aug_clean, axis=0).numpy()


processed_train, processed_train_cleaned = augment_pair(
    pipeline, tf.convert_to_tensor(train), tf.convert_to_tensor(train_cleaned)
)

print(
    "Shapes after augmentation:", processed_train.shape, processed_train_cleaned.shape
)




## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=32, activation="relu", padding="same"
                ),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=2, activation="relu", padding="same"
                ),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.DepthwiseConv2D(
                    (3, 3), depth_multiplier=1, activation="relu", padding="same"
                ),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
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




## === cell 8
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
)

rlp = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.8, patience=5, min_lr=1e-15, mode="min", verbose=1
)

total_samples = processed_train.shape[0]
val_size = int(total_samples * 0.1)

BATCH_SIZE = 256

train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (processed_train[:-val_size], processed_train_cleaned[:-val_size])
    )
    .cache()
    .shuffle(1000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices(
        (processed_train[-val_size:], processed_train_cleaned[-val_size:])
    )
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=[es, rlp],
    verbose=0,  # suppress per‑batch logging for speed
)




## === cell 9
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
plt.show()
del history




## === cell 10
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 11
decoded_imgs = autoencoder(train[:4]).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title(f"Clean image: {train_img[i]}")
    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title(f"Predicted image: {train_img[i]}")
    ax[i][0].axis("off")
    ax[i][1].axis("off")
del decoded_imgs




## === cell 12
orig_shapes = []
for f in test_img:
    img_path = path + "test/" + f
    img = cv2.imread(img_path, 0)
    orig_shapes.append(img.shape)  # (height, width)

decoded_batch = autoencoder.predict(test, batch_size=64, verbose=0)  # faster batch size

ids = []
vals = []

for f, (h_orig, w_orig), decoded_img in zip(test_img, orig_shapes, decoded_batch):
    imgid = int(f[:-4])
    resized = cv2.resize(decoded_img.squeeze(), (w_orig, h_orig))

    rows = np.arange(1, h_orig + 1, dtype=np.int32)
    cols = np.arange(1, w_orig + 1, dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr_flat = rr.ravel()
    cc_flat = cc.ravel()

    ids.extend([f"{imgid}_{r}_{c}" for r, c in zip(rr_flat, cc_flat)])
    vals.extend(resized.ravel().tolist())

print("Length of IDs:", len(ids))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
