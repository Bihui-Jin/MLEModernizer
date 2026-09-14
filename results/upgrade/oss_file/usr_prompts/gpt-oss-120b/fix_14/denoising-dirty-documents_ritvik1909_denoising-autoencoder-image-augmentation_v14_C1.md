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

0.0429

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.05714) has done: 'I wrap the imgaug import in a safe try/except and provide dummy augmenters when the library cannot be loaded, so the pipeline can still run without errors. I also make the augmentation pipeline optional (empty if augmenters are unavailable) and ensure the `augment_pipeline` function simply returns the original images in that case. Finally, I add a small validation split to the model fitting to help early‑stopping without altering the core architecture. These fixes remove the runtime errors, allow the script to produce a valid `submission.csv`, and modestly improve training stability, moving the score toward the target.'
- What this solution (achieved 0.4726) has done: 'I fix the import error handling by providing simple NumPy‑based augmentations when `imgaug` cannot be loaded, and I make the early‑stopping and learning‑rate callbacks monitor validation loss instead of training loss. These minimal changes keep the original architecture intact while improving model generalisation and ensuring the script always runs, producing a proper `submission.csv`.'
- What this solution (achieved 0.47193) has done: 'I fixed the import error by skipping imgaug entirely, rewrote the augmentation logic so that noisy and clean images are augmented together and resized back to the original size (preventing shape mismatches), and updated the cells to use this new paired augmentation. These changes eliminate runtime failures, keep the image‑pair alignment during training (which should improve the denoising performance and move the RMSE closer to the target), and still produce a valid `submission.csv` file.'
- What this solution (achieved 0.28616) has done: 'Implemented fixes to resolve runtime errors and improve model performance:

- Wrapped seaborn import in a safe try/except to avoid protobuf‑related crashes.
- Simplified the augmentation pipeline: removed unnecessary resizing that caused dimension mismatches and directly concatenated augmented images, keeping all tensors 4‑D.
- Enhanced the autoencoder slightly by adding extra convolutional layers to boost capacity while preserving the original architecture style.
- Updated comments for clarity.'
- What this solution (achieved 0.28616) has done: 'Implemented a robust augmentation pipeline that resizes every augmented image back to the original dimensions, fixing the shape mismatch error that halted training. Adjusted the inference step in the submission generation to avoid unnecessary `.numpy()` conversions, ensuring smooth decoding. These fixes allow the autoencoder to train properly and produce a valid `submission.csv`, moving the RMSE score toward the target.'
- What this solution (achieved 0.28616) has done: 'The changes focus on the inference step (cell 12), replacing the per‑pixel Python loops with batch prediction and vectorized ID generation. By predicting the whole test set at once and resizing each decoded image only once, we eliminate millions of inner‑loop iterations, drastically reducing runtime while keeping the model and preprocessing identical. The rest of the notebook remains unchanged, preserving exact training behavior and output format.'
- What this solution (achieved 0.28616) has done: 'The changes speed up image loading by parallelizing disk reads, replace the slow NumPy‑based augmentation with a fully‑vectorized TensorFlow version (removing per‑image Python loops and cv2 resizing), and add deterministic seeds. These adjustments keep the exact same preprocessing, model architecture, training loop, and evaluation logic, so the results remain unchanged while the total runtime falls well under the 600‑second limit.'
- What this solution (achieved 0.28616) has done: 'We replace the eager‑concatenation in the augmentation step with a single batched concat (removing many costly copies) and enable mixed‑precision training to speed up the heavy CNN on large images. Both changes keep the exact same augmentations, model architecture and training semantics while giving a large runtime reduction.'
- What this solution (achieved 0.28616) has done: 'The changes focus on speeding up the training loop, which is the main source of the timeout. By increasing the batch size from 12 to 32 (and keeping the same early‑stopping logic) each epoch processes far fewer steps, reducing overall runtime without altering the model architecture, loss, or augmentation pipeline. The rest of the code remains unchanged to preserve exact prediction semantics.'
- What this solution (achieved 0.28616) has done: 'The changes introduce an efficient tf.data pipeline that batches, shuffles, and prefetches the augmented images, and increase the batch size from 32 to 64, which reduces overhead per epoch. The training split is performed manually so the same validation proportion is kept while using the dataset API. In the prediction step the batch size is also increased to 64 to speed up inference. These adjustments keep the model architecture, augmentation, loss, and early‑stopping logic unchanged, so the results remain identical while fitting comfortably within the 600 s limit.'
- What this solution (achieved 0.28616) has done: 'The changes increase the data pipeline efficiency by caching the datasets in memory and enlarge the batch size, which reduces the number of gradient updates per epoch and thus cuts total training time while preserving the exact model architecture, loss, and augmentation logic. The caching is safe because the data already resides in RAM, and the larger batch does not alter the learning algorithm, only its execution speed.'
- What this solution (achieved 0.28616) has done: 'To meet the 600‑second limit we keep the model and augmentation unchanged but speed up the most costly parts: we lower the early‑stopping patience (still using the same callback logic), drop the non‑essential training metric, and parallelise the large‑scale ID/value construction for the submission file. These tweaks shave off many seconds while preserving the exact algorithmic behaviour.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zipfile, os, cv2, csv
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
def _make_tf_step(func):
    def step(imgs):
        return func(imgs)

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
    return tf.concat(aug_noisy, axis=0), tf.concat(aug_clean, axis=0)


processed_train, processed_train_cleaned = augment_pair(
    pipeline, tf.convert_to_tensor(train), tf.convert_to_tensor(train_cleaned)
)

print(
    "Shapes after augmentation:", processed_train.shape, processed_train_cleaned.shape
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/3813563685.py in <cell line: 0>()
     27 
     28 
---> 29 processed_train, processed_train_cleaned = augment_pair(
     30     pipeline, tf.convert_to_tensor(train), tf.convert_to_tensor(train_cleaned)
     31 )

/tmp/ipykernel_55/3813563685.py in augment_pair(pipeline, noisy_imgs, clean_imgs, seed)
     24         aug_clean.append(step(clean_imgs))
     25     # Return tensors directly; avoid .numpy() conversion.
---> 26     return tf.concat(aug_noisy, axis=0), tf.concat(aug_clean, axis=0)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__ConcatV2_N_6_device_/job:localhost/replica:0/task:0/device:GPU:0}} ConcatOp : Dimension 1 in both shapes must be equal: shape[0] = [115,420,540,1] vs. shape[1] = [115,540,420,1] [Op:ConcatV2] name: concat

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
autoencoder.compile(optimizer="adam", loss="mean_squared_error")




## === cell 8
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2798095711.py in <cell line: 0>()
      7 )
      8 
----> 9 total_samples = processed_train.shape[0]
     10 val_size = int(total_samples * 0.1)
     11 

NameError: name 'processed_train' is not defined

## === cell 9
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
plt.show()
del history




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2176889759.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)
      3 plt.show()
      4 del history
      5 

NameError: name 'history' is not defined

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

with open("submission.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "value"])

    for f_name, (h_orig, w_orig), decoded_img in zip(
        test_img, orig_shapes, decoded_batch
    ):
        imgid = int(f_name[:-4])
        resized = cv2.resize(decoded_img.squeeze(), (w_orig, h_orig))

        rows = np.arange(1, h_orig + 1, dtype=np.int32)
        cols = np.arange(1, w_orig + 1, dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()
        values = resized.ravel()

        for r, c, v in zip(rr_flat, cc_flat, values):
            writer.writerow([f"{imgid}_{r}_{c}", float(v)])

print("Results saved to submission.csv!")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/1271782576.py in <cell line: 0>()
     16     ):
     17         imgid = int(f_name[:-4])
---> 18         resized = cv2.resize(decoded_img.squeeze(), (w_orig, h_orig))
     19 
     20         rows = np.arange(1, h_orig + 1, dtype=np.int32)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4086: error: (-215:Assertion failed) func != 0 in function 'resize'


## --- ERROR in outputing the csv:
Invalid submission: Expected the submission to have 5789880 rows, but got 3628800.
