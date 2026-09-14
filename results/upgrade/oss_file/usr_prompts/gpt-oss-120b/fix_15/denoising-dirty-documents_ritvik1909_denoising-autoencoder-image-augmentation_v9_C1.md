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

0.03088

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fixed the import error from **imgaug** by wrapping it in a safe try/except and falling back to a no‑op augmentation pipeline. The pipeline is now an empty list, so `augment_pipeline` simply returns the original image arrays, allowing the autoencoder to train on real data instead of failing. I also reduced the training epochs to a reasonable number (150) to stay within runtime limits while still improving performance. Finally, minor clean‑ups ensure the script runs end‑to‑end and creates a valid `submission.csv`.'
- What this solution (achieved 0.28616) has done: 'The changes set the protobuf implementation flag before importing TensorFlow to avoid the `MessageFactory` error, fix the input data path, and adjust early‑stopping to monitor validation loss with a small validation split and a larger batch size. These fixes let the notebook run end‑to‑end and should improve the denoising performance, moving the RMSE closer to the target while keeping the original model architecture unchanged.'
- What this solution (achieved 0.28616) has done: 'I moved the protobuf‐implementation flag to the very top so it is applied before any library (including TensorFlow) loads protobuf, fixed the early‑stopping patience to allow longer training, and increased the maximum epochs to give the auto‑encoder more opportunity to converge. These changes resolve the import error and let the model train longer, which should lower the RMSE toward the target while preserving the original architecture.'
- What this solution (achieved 0.28616) has done: 'Implemented fixes:
- Set the protobuf implementation flag **before any imports** to prevent the “MessageFactory” error.
- Increased training capacity: extended `EarlyStopping` patience to 100 and max epochs to 800 so the auto‑encoder can converge further.
- Minor clean‑ups keep the original architecture untouched while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.28616) has done: 'I move the protobuf‑implementation flag to the very top of the script (setting both related env variables) to stop the import‑time `MessageFactory` error, and I remove the early‑stopping callback so the auto‑encoder can train for the full number of epochs (1500). This keeps the original model architecture unchanged while allowing the network to converge better, which should lower the RMSE toward the target. Additionally, I increase the patience value (unused after removal) for safety and keep all other logic identical.'
- What this solution (achieved 0.28616) has done: 'We speed up the notebook by (1) fixing the early‑stopping callback (which was defined but never used) and moving the validation split out of the `.fit` loop, so TensorFlow doesn’t recompute the split each epoch; (2) adding deterministic seeds for reproducibility; (3) removing the costly per‑epoch `np.append`‑based augmentation by building the augmented array once with a list‑concatenation (functionally identical); and (4) keeping all model architecture and training hyper‑parameters unchanged. These changes cut the per‑epoch overhead dramatically while preserving exact model logic and results.'
- What this solution (achieved 0.28616) has done: 'The update enables mixed‑precision training to halve the compute time per epoch and switches the training loop to a cached, pre‑batched `tf.data.Dataset`, which removes Python overhead while keeping the exact model, loss, and early‑stopping logic unchanged. These changes speed up both data feeding and GPU utilization without altering any architectural or training semantics.'
- What this solution (achieved 0.28616) has done: 'I reordered the imports so TensorFlow loads before the optional imgaug package (preventing the protobuf error), fixed the wrong keyword argument when shuffling the training dataset, and ensured the decoded predictions are converted to float32 before OpenCV resizing. The optional image‑augmentation pipeline is now active when imgaug is available, which can improve model performance while keeping the original architecture unchanged. These minimal changes unblock the script, produce a valid `submission.csv`, and move the RMSE closer to the target.'
- What this solution (achieved 0.28616) has done: 'The changes speed up the pipeline by (1) increasing the training batch size to cut the number of gradient steps per epoch, and (2) replacing the Python‑level loops that build millions of string IDs with fully vectorized NumPy string operations, dramatically reducing the time to construct the submission file while keeping the exact same contents and model logic.'
- What this solution (achieved 0.28616) has done: 'The changes replace the slow imgaug augmentation with fast NumPy rotations/flips, add dataset caching to avoid repeated data copying each epoch, and keep the same augmentation outputs, model, and training logic. These tweaks drastically cut preprocessing and per‑epoch overhead while preserving exact functionality and results.'
- What this solution (achieved 0.28616) has done: 'The update speeds up training by making early stopping trigger much sooner (reducing its patience) and by processing larger batches, which cuts the number of optimizer steps per epoch. These changes keep the model architecture, loss, and overall training logic unchanged, so the final learned weights and predictions remain equivalent.'
- What this solution (achieved 0.28616) has done: 'Increasing the batch size drastically reduces the number of training steps per epoch, cutting the overall training time while keeping the exact model, data, epochs, and early‑stopping logic unchanged. The change is limited to the dataset pipeline (cell 11) and retains the same random seeds and shuffling behavior, so the learned model and final predictions remain functionally identical.'
- What this solution (achieved 0.28616) has done: 'The changes speed up the heavyweight I/O and inference steps: image loading/processing is parallelized with a thread pool, and test‑set predictions are done in a single batch instead of one‑by‑one, cutting down Python‑level loops while keeping the exact same preprocessing, model architecture, and training logic.'
- What this solution (achieved 0.28616) has done: 'The changes reduce the expensive full‑batch training by using a modest mini‑batch size (32) while keeping the same model, loss, optimizer, early‑stopping logic, and data augmentations. Mini‑batch training speeds up each epoch dramatically and, because early stopping still monitors validation loss, the number of epochs run be similar or lower, keeping the final accuracy unchanged. No other logic or I/O paths are altered.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

try:
    import imgaug as ia
    from imgaug import augmenters as iaa

    IMG_AUG_AVAILABLE = True
except Exception as e:
    print("imgaug import failed, proceeding without augmentations:", e)
    IMG_AUG_AVAILABLE = False

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

np.random.seed(42)
tf.random.set_seed(42)

sns.set_style("darkgrid")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

for zip_name in [
    "train.zip",
    "test.zip",
    "train_cleaned.zip",
    "sampleSubmission.csv.zip",
]:
    with zipfile.ZipFile(os.path.join(path_zip, zip_name), "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)  # (height, width)




## === cell 3
imgs = [cv2.imread(os.path.join(path, "train", f)) for f in train_img]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 4
def process_image(img_path):
    """Read an image, resize, convert to grayscale, and scale to [0,1]."""
    img = cv2.imread(img_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 5
from concurrent.futures import ThreadPoolExecutor


def load_batch(file_list, folder):
    with ThreadPoolExecutor() as executor:
        paths = [os.path.join(path, folder, f) for f in file_list]
        return list(executor.map(process_image, paths))


train = np.asarray(load_batch(train_img, "train"), dtype="float32")
train_cleaned = np.asarray(
    load_batch(train_cleaned_img, "train_cleaned"), dtype="float32"
)
test = np.asarray(load_batch(test_img, "test"), dtype="float32")




## === cell 6
def augment_pipeline(pipeline, images, seed=19):
    """Apply the same augmentations using fast NumPy ops."""
    if not pipeline:
        return images.copy()
    np.random.seed(seed)
    aug_images = [images]  # original batch
    aug_images.append(np.rot90(images, k=1, axes=(1, 2)))
    aug_images.append(np.rot90(images, k=2, axes=(1, 2)))
    aug_images.append(np.rot90(images, k=3, axes=(1, 2)))
    aug_images.append(np.flip(images, axis=2))
    aug_images.append(np.flip(images, axis=1))
    return np.concatenate(aug_images, axis=0)




## === cell 7
if IMG_AUG_AVAILABLE:
    rotate90 = iaa.Rot90(1)
    rotate180 = iaa.Rot90(2)
    rotate270 = iaa.Rot90(3)
    hflip = iaa.Fliplr(1)
    vflip = iaa.Flipud(1)
    pipeline = [rotate90, rotate180, rotate270, hflip, vflip]
else:
    pipeline = []




## === cell 8
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

print("Processed shapes:", processed_train.shape, processed_train_cleaned.shape)




## === cell 9
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




## === cell 10
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=100, verbose=1, restore_best_weights=True
)




## === cell 11
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    processed_train,
    processed_train_cleaned,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

BATCH_SIZE = 32

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .cache()
    .shuffle(len(X_train), seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1500,
    callbacks=[es],
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
decoded_imgs = autoencoder(train[:4]).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title(f"Denoised image: {train_img[i]}")
    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title(f"Predicted image: {train_img[i]}")
    ax[i][0].axis("off")
    ax[i][1].axis("off")
del decoded_imgs




## === cell 15
decoded_batch = autoencoder(test).numpy()  # shape (N, H, W, 1)

ids_parts = []
vals_parts = []

for i, f in tqdm(enumerate(test_img), total=len(test_img)):
    img_id = int(f[:-4])  # numeric part of filename
    orig_img = cv2.imread(os.path.join(path, "test", f), 0)
    orig_shape = orig_img.shape  # (height, width)

    decoded = np.squeeze(decoded_batch[i]).astype(np.float32)

    decoded_resized = cv2.resize(decoded, (orig_shape[1], orig_shape[0]))

    rows, cols = orig_shape
    rr, cc = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
    img_ids_arr = np.full(rr.size, img_id, dtype=np.int32)
    ids_img = np.char.mod(
        "%d_%d_%d", np.stack([img_ids_arr, rr.ravel(), cc.ravel()], axis=1)
    )
    ids_parts.append(ids_img)
    vals_parts.append(decoded_resized.ravel())

all_ids = np.concatenate(ids_parts)
all_vals = np.concatenate(vals_parts)

print("Length of IDs:", len(all_ids))
pd.DataFrame({"id": all_ids, "value": all_vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/605900799.py in <cell line: 0>()
     16     rr, cc = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
     17     img_ids_arr = np.full(rr.size, img_id, dtype=np.int32)
---> 18     ids_img = np.char.mod(
     19         "%d_%d_%d", np.stack([img_ids_arr, rr.ravel(), cc.ravel()], axis=1)
     20     )

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in mod(a, values)
    417     """
    418     return _to_bytes_or_str_array(
--> 419         _vec_string(a, object_, '__mod__', (values,)), a)
    420 
    421 

TypeError: not enough arguments for format string
