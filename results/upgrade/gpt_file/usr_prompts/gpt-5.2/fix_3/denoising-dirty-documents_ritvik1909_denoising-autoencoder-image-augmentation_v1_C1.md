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

0.02953

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fix the runtime crash caused by `imgaug`/protobuf incompatibility by removing that dependency (the augmentation pipeline isn’t used for training anyway), which unblocks execution. I also make the dataset path handling robust to Kaggle’s directory layout so the zips are found and extracted correctly. Finally, I correct the submission generation to match the required `id,value` order and pixel intensity scaling by writing predictions in exactly the same order as `sampleSubmission.csv` (avoids subtle ordering mismatches that can severely hurt RMSE). These changes keep the core model/training logic intact while addressing the main source of the poor score and the existing runtime errors.'
- What this solution (achieved 0.28616) has done: 'I fix the protobuf/TensorFlow crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing it (this resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment without changing model logic). I also make image reading safer by explicitly using grayscale reads and ensuring `cv2.imread` failures are caught early, preventing silent shape/type issues that can corrupt training/predictions. Finally, I keep your submission-generation logic but ensure it always writes a valid `submission.csv` with the correct `id,value` columns and clipped `[0,1]` predictions in the exact `sampleSubmission.csv` order (score-critical alignment). No changes to the model architecture, loss, or training loop semantics are introduced.'

# 9. Code solution

## === cell 0
import os, zipfile

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
CANDIDATE_INPUT_DIRS = [
    "/kaggle/input/denoising-dirty-documents/",
    "/kaggle/input/",
    "../input/denoising-dirty-documents/",
    "../input/",
]
path_zip = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.isdir(d) and (
        os.path.exists(os.path.join(d, "train.zip"))
        or os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip"))
    ):
        if os.path.exists(os.path.join(d, "denoising-dirty-documents", "train.zip")):
            path_zip = os.path.join(d, "denoising-dirty-documents") + "/"
        else:
            path_zip = d if d.endswith("/") else d + "/"
        break

if path_zip is None:
    raise FileNotFoundError(
        "Could not find denoising-dirty-documents zips under expected /kaggle/input paths."
    )

path = "/kaggle/working/"


def maybe_extract(zip_name, target_dir):
    zip_path = os.path.join(path_zip, zip_name)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing {zip_path}")
    if zip_name.endswith(".csv.zip"):
        out_csv = os.path.join(path, "sampleSubmission.csv")
        if os.path.exists(out_csv):
            return
    else:
        folder = zip_name.replace(".zip", "")
        out_dir = os.path.join(path, folder)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)


maybe_extract("train.zip", path)
maybe_extract("test.zip", path)
maybe_extract("train_cleaned.zip", path)
maybe_extract("sampleSubmission.csv.zip", path)

train_dir = os.path.join(path, "train")
train_cleaned_dir = os.path.join(path, "train_cleaned")
test_dir = os.path.join(path, "test")
sample_sub_path = os.path.join(path, "sampleSubmission.csv")

train_img = sorted(os.listdir(train_dir))
train_cleaned_img = sorted(os.listdir(train_cleaned_dir))
test_img = sorted(os.listdir(test_dir))

print(
    "Num train:",
    len(train_img),
    "Num train_cleaned:",
    len(train_cleaned_img),
    "Num test:",
    len(test_img),
)
print("Sample submission exists:", os.path.exists(sample_sub_path))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = []
for f in sorted(os.listdir(train_dir))[:10]:
    im = cv2.imread(os.path.join(train_dir, f), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Could not read train image: {f}")
    imgs.append(im)
print(
    "Median Dimensions:",
    int(np.median([img.shape[0] for img in imgs])),
    int(np.median([img.shape[1] for img in imgs])),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(train_dir)):
    train.append(process_image(os.path.join(train_dir, f)))

for f in sorted(os.listdir(train_cleaned_dir)):
    train_cleaned.append(process_image(os.path.join(train_cleaned_dir, f)))

for f in sorted(os.listdir(test_dir)):
    test.append(process_image(os.path.join(test_dir, f)))

train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)



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




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    return images




## === cell 8
pipeline = []  # not used




## === cell 9
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
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



## === cell 10
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    train,
    train_cleaned,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=24,
    verbose=1,
)



## === cell 11
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
del history



## === cell 12
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 13
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



## === cell 14
sample = pd.read_csv(sample_sub_path)
ids = sample["id"].values

parts = np.char.split(ids.astype(str), "_")
img_ids = np.array([int(p[0]) for p in parts], dtype=np.int32)
rows_1b = np.array([int(p[1]) for p in parts], dtype=np.int32)
cols_1b = np.array([int(p[2]) for p in parts], dtype=np.int32)

test_id_to_index = {int(fn[:-4]): i for i, fn in enumerate(test_img)}

pred_cache = {}
vals = np.empty(len(ids), dtype=np.float32)

for k in tqdm(range(len(ids))):
    imgid = int(img_ids[k])
    if imgid not in pred_cache:
        if imgid not in test_id_to_index:
            raise KeyError(
                f"Image id {imgid} from sampleSubmission not found in test files."
            )
        idx = test_id_to_index[imgid]

        file_path = os.path.join(test_dir, f"{imgid}.png")
        orig = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            raise FileNotFoundError(f"Could not read {file_path}")
        h, w = orig.shape

        decoded_img = np.squeeze(autoencoder(test[idx : idx + 1]).numpy())  # (420,540)
        decoded_img = np.clip(decoded_img, 0.0, 1.0).astype(np.float32)
        preds_reshaped = cv2.resize(
            decoded_img, (w, h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        pred_cache[imgid] = preds_reshaped

    r = rows_1b[k] - 1
    c = cols_1b[k] - 1
    vals[k] = pred_cache[imgid][r, c]

vals = np.clip(vals, 0.0, 1.0).astype(np.float32)

sub = pd.DataFrame({"id": ids, "value": vals})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with rows:", len(sub))
print(sub.head())
