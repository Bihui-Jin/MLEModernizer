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
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")

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



## === cell 1
path_zip_candidates = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents",
]
path_zip = None
for cand in path_zip_candidates:
    if os.path.exists(os.path.join(cand, "train.zip")):
        path_zip = cand
        break
if path_zip is None:
    path_zip = "/kaggle/input/denoising-dirty-documents/"

path = "/kaggle/working/"

for zname, out_dir in [
    ("train.zip", "train"),
    ("test.zip", "test"),
    ("train_cleaned.zip", "train_cleaned"),
    ("sampleSubmission.csv.zip", None),
]:
    target_dir = os.path.join(path, out_dir) if out_dir else None
    if (
        (target_dir is None)
        or (not os.path.exists(target_dir))
        or (len(os.listdir(target_dir)) == 0)
    ):
        with zipfile.ZipFile(os.path.join(path_zip, zname), "r") as zip_ref:
            zip_ref.extractall(path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Train images:",
    len(train_img),
    "Train_cleaned images:",
    len(train_cleaned_img),
    "Test images:",
    len(test_img),
)




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(train_dir, f)) for f in sorted(os.listdir(train_dir))[:10]
]
print("Example Dimensions (first 10):", [(img.shape[0], img.shape[1]) for img in imgs])
print(
    "Median Dimensions (first 10):",
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
def _stem(fn: str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]


train_files = sorted(os.listdir(train_dir))
clean_files = sorted(os.listdir(train_cleaned_dir))

clean_map = {_stem(f): f for f in clean_files}

paired_train_files = []
paired_clean_files = []
missing = []
for f in train_files:
    s = _stem(f)
    if s in clean_map:
        paired_train_files.append(f)
        paired_clean_files.append(clean_map[s])
    else:
        missing.append(f)

if missing:
    print(
        "Warning: missing cleaned counterparts for",
        len(missing),
        "train files. Example:",
        missing[:5],
    )

print("Paired train/clean count:", len(paired_train_files))

train = []
train_cleaned = []
test = []

for f in paired_train_files:
    train.append(process_image(os.path.join(train_dir, f)))

for f in paired_clean_files:
    train_cleaned.append(process_image(os.path.join(train_cleaned_dir, f)))

for f in sorted(os.listdir(test_dir)):
    test.append(process_image(os.path.join(test_dir, f)))

train = np.asarray(train, dtype=np.float32)
train_cleaned = np.asarray(train_cleaned, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)

train_img = paired_train_files



## === cell 5
print(train.shape, train_cleaned.shape, test.shape)



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




## === cell 7
def augment_pipeline(images):
    """
    np.rot90 swaps (H,W) when rotating by 90/270 degrees.
    To preserve identical tensor shapes for concatenation (and consistent model input),
    rotate each image then resize back to config.IMG_SIZE, keeping the same augmentation idea.
    """
    h, w = config.IMG_SIZE
    out_list = [images.astype(np.float32)]

    def rot_and_resize(batch, k):
        rotated = np.rot90(batch, k=k, axes=(1, 2))  # rotate over (H,W)
        resized = np.empty((batch.shape[0], h, w, 1), dtype=np.float32)
        for i in range(batch.shape[0]):
            resized[i, :, :, 0] = cv2.resize(
                rotated[i, :, :, 0], (w, h), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        return resized

    out_list.append(rot_and_resize(images, 1))
    out_list.append(rot_and_resize(images, 2))
    out_list.append(rot_and_resize(images, 3))

    out_list.append(images[:, :, ::-1, :].astype(np.float32))  # horizontal flip (L-R)
    out_list.append(images[:, ::-1, :, :].astype(np.float32))  # vertical flip (U-D)

    out = np.concatenate(out_list, axis=0).astype(np.float32)
    assert (
        out.shape[1:] == images.shape[1:]
    ), f"Aug shape mismatch: {out.shape} vs {images.shape}"
    return out


processed_train = augment_pipeline(train)
processed_train_cleaned = augment_pipeline(train_cleaned)

print(processed_train.shape, processed_train_cleaned.shape)




## === cell 8
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (5, 5), activation="relu", padding="same"),
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
                layers.Conv2D(48, (5, 5), activation="relu", padding="same"),
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
history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    epochs=500,
    batch_size=12,
)



## === cell 10
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
plt.show()
del history



## === cell 11
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 12
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

plt.show()
del decoded_imgs



## === cell 13
sample_path_candidates = [
    os.path.join(path, "sampleSubmission.csv"),
    os.path.join(path_zip, "sampleSubmission.csv"),
    os.path.join("/kaggle/input", "sampleSubmission.csv"),
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sampleSubmission.csv. Tried: {sample_path_candidates}"
    )

sample = pd.read_csv(sample_path)
n_rows = len(sample)
print("Using sample submission at:", sample_path)
print("Sample submission rows:", n_rows)

ids_out = np.empty(n_rows, dtype=object)
vals_out = np.empty(n_rows, dtype=np.float32)

k = 0
for i, f in tqdm(list(enumerate(test_img)), total=len(test_img)):
    file = os.path.join(test_dir, f)
    imgid = int(os.path.splitext(f)[0])

    img0 = cv2.imread(file, 0)
    img_shape = img0.shape  # (H,W)

    decoded_img = np.squeeze(autoencoder(test[i : i + 1]).numpy())  # (420,540)
    preds_reshaped = cv2.resize(
        decoded_img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    for r in range(img_shape[0]):
        base = f"{imgid}_{r+1}_"
        row_vals = preds_reshaped[r]
        for c in range(img_shape[1]):
            ids_out[k] = base + str(c + 1)
            vals_out[k] = float(row_vals[c])
            k += 1

print("Generated rows:", k)
assert (
    k == n_rows
), f"Row count mismatch vs sampleSubmission: generated {k}, expected {n_rows}"

sub_path = os.path.join(path, "submission.csv")
pd.DataFrame({"id": ids_out, "value": vals_out}).to_csv(sub_path, index=False)
print(f"Results saved to {sub_path}!")
print(pd.read_csv(sub_path, nrows=5))
