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

0.28311

# 6. Current score

0.23316

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.48325) has done: 'I fix the environment-breaking imports/magics by removing notebook-only directives and using `tf_keras` (which is installed) to avoid the protobuf `MessageFactory.GetPrototype` crash. Then I fix the inhomogeneous image array errors by keeping images as lists (variable sizes) and letting the generator `np.stack` only within same-shape batches, while also ensuring train/cleaned images are paired by filename. Finally, I make inference scale inputs to `[0,1]`, pad to the required multiple, clip outputs to `[0,1]`, and write a valid `submission.csv` with the exact `id,value` format to the working directory.'
- What this solution (achieved 0.23316) has done: 'I fix the two runtime blockers so training actually runs: the protobuf `MessageFactory.GetPrototype` crash during `tf_keras` import (by forcing the pure-Python protobuf implementation before any TF/Keras import) and the `workers=0` error in `model.fit` (by removing `workers/use_multiprocessing`, letting defaults apply). Because your current score (0.48325, lower-is-better) is far from the target 0.28311 (>30% gap), I make one minimal, core-logic-preserving change that should improve denoising quality without changing the architecture: train for a few epochs instead of 1 so the same model can converge. I also make the later visualization cells robust so they don’t crash if earlier steps fail, while keeping the submission-writing logic and format unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob, csv
import numpy as np
import matplotlib.pyplot as plt

import tf_keras as keras
from tf_keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Conv2DTranspose
from tf_keras.models import Model

np.random.seed(42)

INPUT_ROOT = "/kaggle/input"
WORK_ROOT = "/kaggle/working"


def _resolve_competition_dir():
    cand = os.path.join(INPUT_ROOT, "denoising-dirty-documents")
    return cand if os.path.isdir(cand) else INPUT_ROOT


COMP_DIR = _resolve_competition_dir()
TRAIN_DIR = os.path.join(COMP_DIR, "train")
TRAIN_CLEAN_DIR = os.path.join(COMP_DIR, "train_cleaned")
TEST_DIR = os.path.join(COMP_DIR, "test")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import cv2


def read_img_float(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype("float32") / 255.0


train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
clean_files = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))

if len(train_files) == 0 or len(clean_files) == 0 or len(test_files) == 0:
    raise RuntimeError(
        "No images found. "
        f"train={TRAIN_DIR} ({len(train_files)}), "
        f"train_cleaned={TRAIN_CLEAN_DIR} ({len(clean_files)}), "
        f"test={TEST_DIR} ({len(test_files)})"
    )



## === cell 2
X_vis = [read_img_float(p) for p in train_files[:4]]
y_vis = [read_img_float(p) for p in clean_files[:4]]

plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_vis[i])
    plt.axis("off")
    plt.subplot(2, 4, i + 5)
    plt.imshow(y_vis[i])
    plt.axis("off")
plt.show()



## === cell 3
shapes = np.unique([read_img_float(p).shape for p in train_files], axis=0)
print(shapes)




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
    def __init__(self, path, val_percentage=0.115, batch_size=4, reduction_factor=4):
        self.base_path = path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        x_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
        y_paths = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

        x_map = {os.path.splitext(os.path.basename(p))[0]: p for p in x_paths}
        y_map = {os.path.splitext(os.path.basename(p))[0]: p for p in y_paths}
        keys = sorted(set(x_map.keys()) & set(y_map.keys()))
        if len(keys) == 0:
            raise RuntimeError("No matching train/train_cleaned filenames found.")

        self.X = [read_img_float(x_map[k]) for k in keys]
        self.y = [read_img_float(y_map[k]) for k in keys]

        self.idx_train, self.idx_val = self.split_batches(val_percentage)
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

        if self.train_steps == 0 or self.val_steps == 0:
            raise RuntimeError(
                f"Not enough batches. train_steps={self.train_steps}, val_steps={self.val_steps}. "
                "Consider reducing batch_size or val_percentage."
            )

    def split_batches(self, val_percentage):
        idx = []
        self.shapes = np.unique([img.shape for img in self.X], axis=0)

        for shape in self.shapes:
            shape_idx = np.argwhere(
                [all(img.shape == shape) for img in self.X]
            ).flatten()
            np.random.shuffle(shape_idx)
            n_batches = int(len(shape_idx) / self.batch_size)
            if n_batches <= 0:
                continue
            idx.extend(np.array_split(shape_idx, n_batches))

        idx = np.array([x for x in idx if x.size == self.batch_size], dtype=object)
        np.random.shuffle(idx)
        samples = len(idx)
        val_samples = int(samples * val_percentage)
        val_samples = max(1, val_samples) if samples > 1 else 0

        if val_samples == 0:
            return idx, idx[:0]
        return idx[:-val_samples], idx[-val_samples:]

    def check_size(self, batch_x, batch_y, axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis] % self.reduction_factor) != 0:
                batch_x = np.insert(
                    batch_x, batch_x.shape[each_axis], 1, axis=each_axis
                )
                batch_y = np.insert(
                    batch_y, batch_y.shape[each_axis], 1, axis=each_axis
                )
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_indices = self.idx_train[0]
            batch_x = np.stack([self.X[i] for i in batch_indices], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_indices], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            self.idx_train = np.roll(self.idx_train, 1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_indices = self.idx_val[0]
            batch_x = np.stack([self.X[i] for i in batch_indices], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_indices], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            self.idx_val = np.roll(self.idx_val, 1, axis=0)
            yield batch_x, batch_y




## === cell 6
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")

log = autoencoder.model.fit(
    x=data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=5,
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    verbose=1,
)



## === cell 7
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()




## === cell 8
def to_csv(pred_list, ids, out_path):
    with open(out_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(("id", "value"))
        for i, each in enumerate(pred_list):
            rows, cols, _ = each.shape
            base_id = str(ids[i])
            for row in range(rows):
                for col in range(cols):
                    id_pixel = base_id + "_" + str(row + 1) + "_" + str(col + 1)
                    value_pixel = float(np.mean(each[row, col, :]))
                    csvwriter.writerow([id_pixel, value_pixel])


test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
X_test_list = [read_img_float(p) for p in test_paths]

predictions = []
for i, x in enumerate(X_test_list):
    original_shape = x.shape

    while (x.shape[0] % 4) != 0:
        x = np.insert(x, x.shape[0], 1, axis=0)
    while (x.shape[1] % 4) != 0:
        x = np.insert(x, x.shape[1], 1, axis=1)

    x_in = x.reshape((1,) + x.shape)

    prediction = autoencoder.model.predict(x_in, verbose=0)
    prediction = prediction.reshape(prediction.shape[1:])
    prediction = prediction[: original_shape[0], : original_shape[1], :]
    prediction = np.clip(prediction, 0.0, 1.0)

    predictions.append(prediction)
    print(f"{i+1} of {len(X_test_list)} predictions calculated", end="\r")

print("\nSaving...")
sub_path = os.path.join(WORK_ROOT, "submission.csv")
to_csv(predictions, ids, sub_path)
print(f"Saved to {sub_path}")



## === cell 9
filters = None
try:
    encoder = autoencoder.get_encoder()
    idx = min(10, len(X_test_list) - 1)
    example = X_test_list[idx].reshape((1,) + X_test_list[idx].shape)
    filters = encoder.predict(example, verbose=0)

    plt.figure()
    plt.imshow(example.reshape(example.shape[1:4]))
    plt.axis("off")
    plt.show()
except Exception as e:
    print("Skipping encoder visualization due to error:", repr(e))



## === cell 10
import matplotlib.animation as animation

ani = None
if filters is not None:
    fig = plt.figure()
    ims = []
    for i in range(filters.shape[-1]):
        im = plt.imshow(
            filters[:, :, :, i].reshape(filters.shape[1:3]), animated=True, cmap="gray"
        )
        plt.axis("off")
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=0)



## === cell 11
print(
    "Animation object created:",
    isinstance(ani, animation.ArtistAnimation) if ani is not None else False,
)
