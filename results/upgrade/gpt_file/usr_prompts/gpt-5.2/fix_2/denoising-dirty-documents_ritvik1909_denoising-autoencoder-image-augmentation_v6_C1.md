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

0.02843

# 6. Current score

0.47036

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.47036) has done: 'I fix the TensorFlow/protobuf crash caused by importing `imgaug` (it’s incompatible with the Kaggle TF/protobuf stack here) by removing `imgaug` usage and replacing the augmentation with safe, equivalent numpy-based flips/rotations. This unblocks the pipeline so `processed_train` exists and training runs, which should also substantially improve score versus the current broken/no-augmentation path. I also make the data path robust to both `/kaggle/input/...` and the provided relative path, and keep the model architecture/training loop/loss intact. Finally, I keep the submission format identical but make prediction code slightly safer (clipping to [0,1]) without changing semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, os, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip_candidates = [
    "/kaggle/input/denoising-dirty-documents/",
    "../input/denoising-dirty-documents/",
    "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents/",
]
path_zip = None
for p in path_zip_candidates:
    if os.path.exists(p) and (
        os.path.exists(os.path.join(p, "train.zip"))
        or os.path.exists(os.path.join(p, "train/"))
    ):
        path_zip = p
        break
if path_zip is None:
    raise FileNotFoundError(
        "Could not find denoising-dirty-documents input folder in expected locations."
    )

path = "/kaggle/working/"


def _safe_extract(zip_path, out_dir):
    if not os.path.exists(zip_path):
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


if os.path.exists(os.path.join(path_zip, "train.zip")):
    _safe_extract(os.path.join(path_zip, "train.zip"), path)
    _safe_extract(os.path.join(path_zip, "test.zip"), path)
    _safe_extract(os.path.join(path_zip, "train_cleaned.zip"), path)
    _safe_extract(os.path.join(path_zip, "sampleSubmission.csv.zip"), path)
else:
    pass

if (
    os.path.exists(os.path.join(path, "train"))
    and len(os.listdir(os.path.join(path, "train"))) > 0
):
    base_img_path = path
else:
    base_img_path = path_zip if path_zip.endswith("/") else path_zip + "/"

train_img = sorted(os.listdir(os.path.join(base_img_path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(base_img_path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(base_img_path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(base_img_path, "train", f))
    for f in sorted(os.listdir(os.path.join(base_img_path, "train")))
]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 3
def process_image(p):
    img = cv2.imread(p)
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

for f in sorted(os.listdir(os.path.join(base_img_path, "train"))):
    train.append(process_image(os.path.join(base_img_path, "train", f)))

for f in sorted(os.listdir(os.path.join(base_img_path, "train_cleaned"))):
    train_cleaned.append(process_image(os.path.join(base_img_path, "train_cleaned", f)))

for f in sorted(os.listdir(os.path.join(base_img_path, "test"))):
    test.append(process_image(os.path.join(base_img_path, "test", f)))

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
    rng = np.random.default_rng(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = step(images, rng)
        processed_images = np.append(processed_images, temp, axis=0)
    return processed_images


def aug_rot90_k(k):
    def _fn(images, rng):
        return np.rot90(images, k=k, axes=(1, 2)).copy()

    return _fn


def aug_hflip(images, rng):
    return images[:, :, ::-1, :].copy()


def aug_vflip(images, rng):
    return images[:, ::-1, :, :].copy()




## === cell 8
pipeline = [aug_rot90_k(1), aug_rot90_k(2), aug_rot90_k(3), aug_hflip, aug_vflip]



## === cell 9
processed_train = augment_pipeline(pipeline, train, seed=19)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=19)

processed_train.shape, processed_train_cleaned.shape




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3718931983.py in <cell line: 0>()
----> 1 processed_train = augment_pipeline(pipeline, train, seed=19)
      2 processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=19)
      3 
      4 processed_train.shape, processed_train_cleaned.shape
      5 

/tmp/ipykernel_11/3573652310.py in augment_pipeline(pipeline, images, seed)
      7         # Step can be a callable(images)->aug_images
      8         temp = step(images, rng)
----> 9         processed_images = np.append(processed_images, temp, axis=0)
     10     return processed_images
     11 

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in append(arr, values, axis)
   5616         values = ravel(values)
   5617         axis = arr.ndim-1
-> 5618     return concatenate((arr, values), axis=axis)
   5619 
   5620 

ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 420 and the array at index 1 has size 540

## === cell 10
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
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
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3245784305.py in <cell line: 0>()
      4 
      5 history = autoencoder.fit(
----> 6     processed_train,
      7     processed_train_cleaned,
      8     shuffle=True,

NameError: name 'processed_train' is not defined

## === cell 12
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
del history



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2278981582.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).plot(ax=ax)
      3 del history
      4 

NameError: name 'history' is not defined

## === cell 13
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## === cell 14
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



## === cell 15
ids = []
vals = []
for i, f in tqdm(list(enumerate(test_img))):
    file = os.path.join(base_img_path, "test", f)
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape

    decoded_img = np.squeeze(autoencoder(test[i : i + 1]).numpy())
    preds_reshaped = cv2.resize(decoded_img, (img_shape[1], img_shape[0]))
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(f"{imgid}_{r+1}_{c+1}")
            vals.append(float(preds_reshaped[r, c]))

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id", "value"]
assert len(sub) == len(ids)
sub.head()
