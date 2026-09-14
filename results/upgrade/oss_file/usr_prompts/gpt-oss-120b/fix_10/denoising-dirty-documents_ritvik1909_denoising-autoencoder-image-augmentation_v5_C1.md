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
import seaborn as sns

import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

tf.config.optimizer.set_jit(True)

import random
import concurrent.futures  # added for parallel image loading

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception as e:
    ia = None
    iaa = None

sns.set_style("darkgrid")




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def extract_if_missing(zip_name, target_dir):
    if not os.path.isdir(target_dir):
        with zipfile.ZipFile(path_zip + zip_name, "r") as zip_ref:
            zip_ref.extractall(path)


extract_if_missing("train.zip", os.path.join(path, "train"))
extract_if_missing("test.zip", os.path.join(path, "test"))
extract_if_missing("train_cleaned.zip", os.path.join(path, "train_cleaned"))
extract_if_missing("sampleSubmission.csv.zip", path)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)




## === cell 3
imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 4
def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 5
def load_images(file_list, base_dir):
    """Load and preprocess images using a process pool (CPU‑bound work)."""
    paths = [os.path.join(base_dir, f) for f in file_list]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        images = list(executor.map(process_image, paths))
    return np.asarray(images)


train = load_images(sorted(os.listdir(path + "train/")), path + "train/")
train_cleaned = load_images(
    sorted(os.listdir(path + "train_cleaned/")), path + "train_cleaned/"
)
test = load_images(sorted(os.listdir(path + "test/")), path + "test/")

test_original_shapes = []
for f in test_img:
    orig = cv2.imread(os.path.join(path, "test", f), cv2.IMREAD_GRAYSCALE)
    test_original_shapes.append(orig.shape)  # (h, w)




## === cell 6
train.shape, train_cleaned.shape, test.shape




## === cell 7
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




## === cell 8
def augment_pipeline(pipeline, images, seed=19):
    """
    Apply an imgaug pipeline to a set of images.
    If the pipeline is empty we simply return the original array
    (no unnecessary copy). When a pipeline is provided we build the
    augmented set with np.concatenate to avoid repeated np.append copies.
    """
    if not pipeline:
        return images  # no copy, saves time and memory
    ia.seed(seed)
    augmented = [images]
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        augmented.append(temp)
    return np.concatenate(augmented, axis=0)




## === cell 9
pipeline = []




## === cell 10
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.2),  # reduced dropout to improve learning
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
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




## === cell 12
def _cast_to_float16(x, y):
    x = tf.cast(x, tf.float16)
    y = tf.cast(y, tf.float16)
    return x, y


train_ds = tf.data.Dataset.from_tensor_slices(
    (processed_train, processed_train_cleaned)
)
train_ds = train_ds.map(_cast_to_float16, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

es = callbacks.EarlyStopping(
    monitor="loss", patience=5000, verbose=0, restore_best_weights=False
)

history = autoencoder.fit(
    train_ds,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    verbose=0,
)




## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
del history




## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()




## === cell 15
decoded_imgs = autoencoder(train[:4]).numpy()

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

del decoded_imgs




## === cell 16
decoded_all = autoencoder(test).numpy()  # shape (N, H, W, 1)

ids = []
vals = []

for i, f in enumerate(test_img):
    imgid = int(f[:-4])
    h, w = test_original_shapes[i]  # use cached original dimensions

    decoded = np.squeeze(decoded_all[i])  # (H, W) in model size
    pred_resized = cv2.resize(decoded, (w, h))  # back to original size

    vals.extend(pred_resized.ravel())

    rows = np.arange(1, h + 1, dtype=str)[:, None]  # (h,1)
    cols = np.arange(1, w + 1, dtype=str)[None, :]  # (1,w)
    ids_img = np.char.add(
        np.char.add(np.char.add(str(imgid) + "_", rows), "_"), cols
    ).ravel()
    ids.extend(ids_img.tolist())

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
