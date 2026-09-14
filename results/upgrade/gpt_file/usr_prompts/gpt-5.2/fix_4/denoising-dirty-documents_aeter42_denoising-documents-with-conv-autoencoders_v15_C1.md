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

3.7

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
PyYAML==6.0.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.19042

# 6. Current score

0.12785

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.48372) has done: 'I fix the environment-breaking import issue by switching the Keras import path to `tf_keras` (compatible with the Kaggle TensorFlow/Keras stack) and remove notebook-only magics. Then I fix the core data loading bug (images have varying shapes) by avoiding `np.array(...)` on variable-sized images and instead grouping indices by shape while keeping the generator-based training logic intact. Finally, I make inference and submission writing robust: read test images as a list, apply the same padding/normalization used in training, clip outputs to [0,1], and write `submission.csv` with exactly `id,value` as required.'
- What this solution (achieved 0.1111) has done: 'I fix the runtime failure that happens before any training by removing the problematic protobuf-dependent import chain (matplotlib/animation/IPython) from the early cells, since it is not required for training or submission generation. Then I fix the `model.fit(...)` crash by omitting the invalid `workers=0` argument (tf_keras expects `workers>=1` if provided), which allow training to run and produce `log` for the later plotting cell. Finally, I keep the existing autoencoder/generator core logic intact and ensure the pipeline always writes a valid `submission.csv` with `id,value` and clipped predictions in `[0,1]`.'
- What this solution (achieved 0.12785) has done: 'I fix the environment-breaking `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by preventing optional visualization imports (matplotlib/IPython) from being imported at module import time, since that import chain is what triggers the protobuf incompatibility in this Kaggle image. I also ensure we always use the correct competition input paths (`/kaggle/input/denoising-dirty-documents/...`) deterministically, and keep the training/inference/submission core logic unchanged. Finally, I keep submission generation stable (correct `id,value`, clipping to `[0,1]`) and write to `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os, glob, csv
import numpy as np
import cv2

from tf_keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Conv2DTranspose
from tf_keras.models import Model

np.random.seed(42)

BASE_INPUT = "/kaggle/input"
COMP_DIR = os.path.join(BASE_INPUT, "denoising-dirty-documents")

if os.path.isdir(COMP_DIR):
    TRAIN_DIR = os.path.join(COMP_DIR, "train")
    TRAIN_CLEAN_DIR = os.path.join(COMP_DIR, "train_cleaned")
    TEST_DIR = os.path.join(COMP_DIR, "test")
else:
    TRAIN_DIR = os.path.join(BASE_INPUT, "train")
    TRAIN_CLEAN_DIR = os.path.join(BASE_INPUT, "train_cleaned")
    TEST_DIR = os.path.join(BASE_INPUT, "test")

print("Using paths:")
print("TRAIN_DIR      :", TRAIN_DIR)
print("TRAIN_CLEAN_DIR:", TRAIN_CLEAN_DIR)
print("TEST_DIR       :", TEST_DIR)


def imread_rgb(path):
    """Read image as RGB float32 in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
clean_files = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

train_map = {os.path.basename(p): p for p in train_files}
clean_map = {os.path.basename(p): p for p in clean_files}
common_names = sorted(set(train_map.keys()).intersection(clean_map.keys()))
if len(common_names) == 0:
    raise RuntimeError("No matching train/train_cleaned filenames found. Check paths.")

X = [imread_rgb(train_map[n]) for n in common_names[:4]]
y = [imread_rgb(clean_map[n]) for n in common_names[:4]]

print("Loaded sample shapes (X, y):")
for i in range(len(X)):
    print(i, X[i].shape, y[i].shape)



## === cell 2
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 8))
    for i in range(min(4, len(X))):
        plt.subplot(241 + i)
        fig = plt.imshow(X[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(245 + i)
        fig = plt.imshow(y[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.show()
except Exception as e:
    print("Skipping visualization due to:", repr(e))



## === cell 3
shapes = np.unique([imread_rgb(each).shape for each in train_files], axis=0)
print("Unique shapes in train:", shapes)




## === cell 4
class Autoencoder:
    def __init__(
        self,
        dimensions_factor=2,
        layers=2,
        k=3,
        filter_size=None,
        pooling_factor=None,
        only_decoder=False,
        only_encoder=False,
        loss="mean_squared_error",
        channels=3,
    ):

        if not (only_decoder or only_encoder):
            self.channels = channels
            self.layers = layers
            self.k = k
            self.filter_size = (
                [(k, k)] * (layers * 2 + 1) if (filter_size is None) else filter_size
            )
            self.pooling_factor = (
                [2] * (layers * 2) if (pooling_factor is None) else pooling_factor
            )
            self.dimensions_factor = dimensions_factor

        if not only_decoder:
            input_img = Input(shape=(None, None, self.channels))
            x = input_img
            for i in range(self.layers):
                filters = int(
                    self.pooling_factor[i] * int(x.shape[-1]) * self.dimensions_factor
                )
                x = Conv2D(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="encoding_conv_" + str(i),
                )(x)
                x = MaxPooling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    padding="valid",
                    name="encoding_pool_" + str(i),
                )(x)
            self.code_filters = filters
        else:
            input_img = Input(shape=(None, None, self.code_filters))
            x = input_img

        if not only_encoder:
            for i in range(self.layers, 2 * self.layers):
                filters = int(
                    int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
                )
                x = Conv2DTranspose(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="decoding_conv_" + str(i),
                )(x)
                x = UpSampling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    name="decoding_pool_" + str(i),
                )(x)

            x = Conv2D(
                self.channels,
                (self.filter_size[-1]),
                activation="sigmoid",
                padding="same",
                name="decoded",
            )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])

        if not (only_decoder or only_encoder):
            self.model = autoencoder
        else:
            return autoencoder

    def get_encoder(self):
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not created")
        encoder = self.__init__(only_encoder=True)
        for layer in encoder.layers[1:]:
            weights = self.model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)
        return encoder

    def get_decoder(self):
        if getattr(self, "model", None) is None:
            raise RuntimeError("Model is not created")
        decoder = self.__init__(only_decoder=True)
        for layer in decoder.layers[1:]:
            weights = self.model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)
        return decoder




## === cell 5
class Image_generator:
    """
    Fix: don't force variable-sized images into a single np.array (causes ValueError).
    Keep the same overall batching idea: group by shape, yield fixed-shape batches.
    """

    def __init__(self, path, val_percentage=0.115, batch_size=4, reduction_factor=4):
        self.base_path = path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
        clean_files = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

        train_map = {os.path.basename(p): p for p in train_files}
        clean_map = {os.path.basename(p): p for p in clean_files}
        self.names = sorted(set(train_map.keys()).intersection(clean_map.keys()))
        if len(self.names) == 0:
            raise RuntimeError("No matching train/train_cleaned filenames found.")

        self.train_paths = [train_map[n] for n in self.names]
        self.clean_paths = [clean_map[n] for n in self.names]

        self.X = [imread_rgb(p) for p in self.train_paths]
        self.y = [imread_rgb(p) for p in self.clean_paths]

        self.idx_train, self.idx_val = self.split_batches(val_percentage)

        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

    def split_batches(self, val_percentage):
        idx_batches = []
        shapes = {}
        for i, img in enumerate(self.X):
            shapes.setdefault(img.shape, []).append(i)

        self.shapes = np.array(list(shapes.keys()), dtype=object)

        for shape, shape_idx in shapes.items():
            shape_idx = np.array(shape_idx, dtype=int)
            np.random.shuffle(shape_idx)
            num_full = (len(shape_idx) // self.batch_size) * self.batch_size
            shape_idx = shape_idx[:num_full]
            if num_full == 0:
                continue
            batches = np.array(shape_idx).reshape(-1, self.batch_size)
            idx_batches.extend(list(batches))

        idx_batches = np.array(idx_batches, dtype=int)
        np.random.shuffle(idx_batches)
        samples = len(idx_batches)
        val_samples = int(samples * val_percentage)
        if val_samples < 1 and samples > 1:
            val_samples = 1

        return idx_batches[:-val_samples], idx_batches[-val_samples:]

    def check_size(self, batch_x, batch_y, axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis] % self.reduction_factor) != 0:
                batch_x = np.insert(
                    batch_x, batch_x.shape[each_axis], 1.0, axis=each_axis
                )
                batch_y = np.insert(
                    batch_y, batch_y.shape[each_axis], 1.0, axis=each_axis
                )
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_ids = self.idx_train[0]
            batch_x = np.stack([self.X[i] for i in batch_ids], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_ids], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            self.idx_train = np.roll(self.idx_train, 1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_ids = self.idx_val[0]
            batch_x = np.stack([self.X[i] for i in batch_ids], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_ids], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            self.idx_val = np.roll(self.idx_val, 1, axis=0)
            yield batch_x, batch_y




## === cell 6
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")

print(
    "Train steps:", data_generator.train_steps, "Val steps:", data_generator.val_steps
)

log = autoencoder.model.fit(
    x=data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=20,
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    verbose=1,
)



## === cell 7
try:
    import matplotlib.pyplot as plt

    plt.figure()
    for k, v in log.history.items():
        plt.plot(v, label=str(k))
    plt.legend()
    plt.show()
except Exception as e:
    print("Skipping training plot due to:", repr(e))




## === cell 8
def to_csv(pred_list, ids, out_path="submission.csv"):
    """
    Write valid submission with id,value; values clipped to [0,1].
    pred_list: list of (H,W,C) float arrays in [0,1]
    ids: list of image ids (filenames without .png)
    """
    with open(out_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(("id", "value"))
        for i, each in enumerate(pred_list):
            each = np.clip(each, 0.0, 1.0)
            rows, cols, _ = each.shape
            img_id = str(ids[i])
            for row in range(rows):
                for col in range(cols):
                    id_pixel = img_id + "_" + str(row + 1) + "_" + str(col + 1)
                    value_pixel = float(np.mean(each[row, col, :]))
                    csvwriter.writerow([id_pixel, value_pixel])


test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
if len(test_files) == 0:
    raise RuntimeError(f"No test images found in {TEST_DIR}")

ids = [os.path.basename(p)[:-4] for p in test_files]

predictions = []
for i, p in enumerate(test_files, start=1):
    x = imread_rgb(p)  # normalize to [0,1] as during training
    original_shape = x.shape

    while (x.shape[0] % 4) != 0:
        x = np.insert(x, x.shape[0], 1.0, axis=0)
    while (x.shape[1] % 4) != 0:
        x = np.insert(x, x.shape[1], 1.0, axis=1)

    x_in = x.reshape((1,) + x.shape)
    prediction = autoencoder.model.predict(x_in, verbose=0)[0]
    prediction = prediction[: original_shape[0], : original_shape[1], :]
    predictions.append(prediction)

    if i % 1 == 0:
        print(f"{i} of {len(test_files)} predictions calculated", end="\r")

print("\nSaving...")
to_csv(predictions, ids, out_path="submission.csv")
print("Saved submission.csv")



## === cell 9
try:
    encoder = autoencoder.get_encoder()
    example = imread_rgb(test_files[min(10, len(test_files) - 1)])
    ex_in = example.reshape((1,) + example.shape)
    filters = encoder.predict(ex_in, verbose=0)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(example)
    plt.axis("off")
    plt.show()

    print("Encoder output shape:", filters.shape)
except Exception as e:
    filters = None
    print("Skipping encoder visualization due to:", repr(e))



## === cell 10
try:
    if filters is None:
        raise RuntimeError("No filters to animate.")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(4, 4))
    ims = []
    max_frames = min(filters.shape[-1], 32)
    for i in range(max_frames):
        im = plt.imshow(filters[0, :, :, i], animated=True, cmap="viridis")
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=0)
except Exception as e:
    ani = None
    print("Skipping animation due to:", repr(e))



## === cell 11
try:
    if ani is not None:
        from IPython.display import HTML

        HTML(ani.to_jshtml())
except Exception as e:
    print("Animation display not available:", repr(e))
