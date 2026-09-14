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

0.14197

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import glob, os, csv
import numpy as np
from PIL import Image


def imread(path):
    """Read an image file as an RGB uint8 NumPy array (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
    Conv2DTranspose,
)
from tensorflow.keras import Model
import matplotlib.pyplot as plt




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
X = [
    imread(each)
    for each in glob.glob(os.path.join(os.getcwd(), "../input/train/*.png"))[:4]
]
y = [
    imread(each)
    for each in glob.glob(os.path.join(os.getcwd(), "../input/train_cleaned/*.png"))[:4]
]

plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(241 + i)
    plt.imshow(X[i])
    plt.axis("off")
    plt.subplot(245 + i)
    plt.imshow(y[i])
    plt.axis("off")
plt.show()




## === cell 2
shapes = np.unique(
    [
        imread(each).shape
        for each in glob.glob(os.path.join(os.getcwd(), "../input/train/*.png"))
    ],
    axis=0,
)
print("Unique image shapes in training set:", shapes)




## === cell 3
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
                [(k, k)] * (layers * 2 + 1) if filter_size is None else filter_size
            )
            self.pooling_factor = (
                [2] * (layers * 2) if pooling_factor is None else pooling_factor
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
                self.filter_size[-1],
                activation="sigmoid",
                padding="same",
                name="decoded",
            )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])

        self.model = autoencoder

    def get_encoder(self):
        encoder_input = self.model.input
        encoder_output = self.model.get_layer(
            name="encoding_pool_" + str(self.layers - 1)
        ).output
        return Model(encoder_input, encoder_output)

    def get_decoder(self):
        decoder_input = Input(shape=(None, None, self.code_filters))
        x = decoder_input
        for i in range(self.layers, 2 * self.layers):
            filters = int(
                int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
            )
            x = Conv2DTranspose(
                filters, self.filter_size[i], activation="relu", padding="same"
            )(x)
            x = UpSampling2D((self.pooling_factor[i], self.pooling_factor[i]))(x)
        x = Conv2D(
            self.channels, self.filter_size[-1], activation="sigmoid", padding="same"
        )(x)
        return Model(decoder_input, x)




## === cell 4
np.random.seed(42)
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")

log = autoencoder.model.fit(
    data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=10,  # reduced for faster turnaround
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3462580761.py in <cell line: 0>()
      1 np.random.seed(42)
----> 2 data_generator = Image_generator(os.getcwd())
      3 autoencoder = Autoencoder(loss="mse")
      4 
      5 log = autoencoder.model.fit(

NameError: name 'Image_generator' is not defined

## === cell 5
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3891366262.py in <cell line: 0>()
      1 plt.figure()
----> 2 for k, v in log.history.items():
      3     plt.plot(v, label=str(k))
      4 plt.legend()
      5 plt.show()

NameError: name 'log' is not defined

## === cell 6
def to_csv(npdata, ids):
    with open("submission.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(("id", "value"))
        for i, each in enumerate(npdata):
            rows, cols, _ = each.shape
            for row in range(rows):
                for col in range(cols):
                    id_pixel = f"{ids[i]}_{row+1}_{col+1}"
                    value_pixel = np.mean(each[row, col, :])
                    csvwriter.writerow([id_pixel, value_pixel])


test_paths = sorted(glob.glob(os.path.join(os.getcwd(), "../input/test/*.png")))
X_test = [imread(p) for p in test_paths]
ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]

predictions = []
for x in X_test:
    original_shape = x.shape
    while x.shape[0] % 4 != 0:
        x = np.insert(x, x.shape[0], 1, axis=0)
    while x.shape[1] % 4 != 0:
        x = np.insert(x, x.shape[1], 1, axis=1)
    x = x.astype("float32") / 255.0
    x = x.reshape((1,) + x.shape)  # add batch dim

    pred = autoencoder.model.predict(x, verbose=0)
    pred = pred.reshape(pred.shape[1:])  # drop batch dim
    pred = pred[: original_shape[0], : original_shape[1], :]  # crop back
    predictions.append(pred)

print("\nSaving submission...")
to_csv(predictions, ids)
print("Saved submission.csv")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2455967479.py in <cell line: 0>()
     27     x = x.reshape((1,) + x.shape)  # add batch dim
     28 
---> 29     pred = autoencoder.model.predict(x, verbose=0)
     30     pred = pred.reshape(pred.shape[1:])  # drop batch dim
     31     pred = pred[: original_shape[0], : original_shape[1], :]  # crop back

NameError: name 'autoencoder' is not defined

## === cell 7
try:
    encoder = autoencoder.get_encoder()
    example = X_test[0]
    while example.shape[0] % 4 != 0:
        example = np.insert(example, example.shape[0], 1, axis=0)
    while example.shape[1] % 4 != 0:
        example = np.insert(example, example.shape[1], 1, axis=1)
    example = example.astype("float32") / 255.0
    example = example.reshape((1,) + example.shape)
    filters = encoder.predict(example)
    plt.figure()
    plt.imshow(example[0])
    plt.title("Encoded representation shape: " + str(filters.shape))
except Exception as e:
    print("Encoder visualization skipped:", e)




## === cell 8
try:
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure()
    ims = []
    for i in range(filters.shape[-1]):
        im = plt.imshow(filters[0, :, :, i], animated=True, cmap="viridis")
        ims.append([im])
    ani = animation.ArtistAnimation(
        fig, ims, interval=500, blit=True, repeat_delay=1000
    )
except Exception as e:
    print("Animation creation skipped:", e)




## === cell 9
try:
    HTML(ani.to_jshtml())
except Exception:
    pass
