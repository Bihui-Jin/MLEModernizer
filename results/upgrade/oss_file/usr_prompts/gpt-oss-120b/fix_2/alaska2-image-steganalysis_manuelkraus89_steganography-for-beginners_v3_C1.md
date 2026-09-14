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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.65

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os



## === cell 1
image_cover = color.rgb2ycbcr(
    io.imread("/kaggle/input/alaska2-image-steganalysis/Cover/00001.jpg")
)
image_jmipod = color.rgb2ycbcr(
    io.imread("/kaggle/input/alaska2-image-steganalysis/JMiPOD/00001.jpg")
)
image_juniward = color.rgb2ycbcr(
    io.imread("/kaggle/input/alaska2-image-steganalysis/JUNIWARD/00001.jpg")
)
image_uerd = color.rgb2ycbcr(
    io.imread("/kaggle/input/alaska2-image-steganalysis/UERD/00001.jpg")
)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(io.imread("/kaggle/input/alaska2-image-steganalysis/Cover/00001.jpg"))
ax[1].imshow(image_cover[:, :, 0])
plt.show()




## === cell 2
def calc_block_std(image_channel):
    std_image = np.zeros((64, 64))
    for i in range(63):
        for j in range(63):
            std_image[i, j] = np.std(
                image_channel[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            )
    return std_image


cover_y_std = calc_block_std(image_cover[:, :, 0])
plt.imshow(cover_y_std)




## === cell 3
def block_changed(image_cover, image_hidden):
    changed_image = np.ones((64, 64))
    for i in range(63):
        for j in range(63):
            changed_image[i, j] = not np.allclose(
                image_cover[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8],
                image_hidden[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8],
            )
    return changed_image.astype(bool)




## === cell 4
jmipod_diff = block_changed(image_cover[:, :, 0], image_jmipod[:, :, 0])
juniward_diff = block_changed(image_cover[:, :, 0], image_juniward[:, :, 0])
uerd_diff = block_changed(image_cover[:, :, 0], image_uerd[:, :, 0])
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(cover_y_std)
ax[1].imshow(jmipod_diff)
ax[2].imshow(juniward_diff)
ax[3].imshow(uerd_diff)
plt.show()



## === cell 5
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from math import ceil
from skimage.io import imread
from skimage import color, transform
import numpy as np
from scipy import ndimage, misc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Activation,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
cover_path = os.path.join(
    os.getcwd(), "/kaggle/input/alaska2-image-steganalysis/Cover/"
)
cover_paths_labels = [(os.path.join(cover_path, f), 0) for f in os.listdir(cover_path)]

jmipod_path = os.path.join(
    os.getcwd(), "/kaggle/input/alaska2-image-steganalysis/JMiPOD/"
)
jmipod_paths_labels = [
    (os.path.join(jmipod_path, f), 1) for f in os.listdir(jmipod_path)
]

juniward_path = os.path.join(
    os.getcwd(), "/kaggle/input/alaska2-image-steganalysis/JUNIWARD/"
)
juniward_paths_labels = [
    (os.path.join(juniward_path, f), 2) for f in os.listdir(juniward_path)
]

uerd_path = os.path.join(os.getcwd(), "/kaggle/input/alaska2-image-steganalysis/UERD/")
uerd_paths_labels = [(os.path.join(uerd_path, f), 3) for f in os.listdir(uerd_path)]



## === cell 7
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
NUM_CLASSES = 4


def _parse_function(image_path, label):
    bits = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(bits, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [512, 512, 1])
    label = tf.one_hot(label, NUM_CLASSES)
    label = tf.reshape(label, (4,))
    return image, label


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    return image, label


def create_dataset(image_paths_labels, augment=True, shuffle=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_paths_labels,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    dataset = dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE)
    if augment:
        dataset = dataset.map(
            _augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset




## === cell 8
data_cover = create_dataset(cover_paths_labels, augment=False, shuffle=False).take(1)
data_jmipod = create_dataset(jmipod_paths_labels, augment=False, shuffle=False).take(1)
data_juniward = create_dataset(
    juniward_paths_labels, augment=False, shuffle=False
).take(1)
data_uerd = create_dataset(uerd_paths_labels, augment=False, shuffle=False).take(1)

for cov, jmi, jun, uerd in zip(data_cover, data_jmipod, data_juniward, data_uerd):
    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    ax[0, 0].imshow(np.reshape(cov[0][1], (512, 512)))
    ax[0, 1].imshow(np.reshape(jmi[0][1], (512, 512)))
    ax[0, 2].imshow(np.reshape(jun[0][1], (512, 512)))
    ax[0, 3].imshow(np.reshape(uerd[0][1], (512, 512)))
    ax[1, 0].imshow(np.abs(np.reshape(jmi[0][1] - cov[0][1], (512, 512))))
    ax[1, 1].imshow(np.abs(np.reshape(jun[0][1] - cov[0][1], (512, 512))))
    ax[1, 2].imshow(np.abs(np.reshape(uerd[0][1] - cov[0][1], (512, 512))))
    ax[1, 3].imshow(np.abs(np.reshape(uerd[0][1] - jmi[0][1], (512, 512))))
    plt.show()




## === cell 9
def weight_init(shape, dtype=None):
    kernel_sum = tf.stack(
        [
            kernel_first_order_1,
            kernel_first_order_2,
            kernel_first_order_3,
            kernel_first_order_4,
            kernel_second_order_1,
            kernel_second_order_2,
            kernel_second_order_3,
            kernel_second_order_4,
            kernel_third_order_1,
            kernel_third_order_2,
            kernel_third_order_3,
            kernel_third_order_4,
            kernel_edge_three_1,
            kernel_edge_three_2,
            kernel_edge_three_3,
            kernel_edge_three_4,
            square_kernel_1,
            square_kernel_2,
        ],
        axis=3,
    )
    kernel_collection = tf.reshape(kernel_sum, [5, 5, 1, 18])
    return kernel_collection




## === cell 10
paths_labels = (
    cover_paths_labels + jmipod_paths_labels + juniward_paths_labels + uerd_paths_labels
)
paths_labels_train, paths_labels_valid = train_test_split(
    paths_labels, shuffle=True, test_size=0.20
)
data_train = create_dataset(paths_labels_train)
data_valid = create_dataset(paths_labels_valid)



## === cell 11
model_dummy = Sequential()
model_dummy.add(
    Conv2D(
        18,
        kernel_initializer=weight_init,
        kernel_size=5,
        padding="valid",
        input_shape=(512, 512, 1),
    )
)
data_dummy = create_dataset(paths_labels_train).take(1)
dummy_images = model_dummy.predict(data_dummy)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(np.reshape(dummy_images[0, :, :, 13] * 255, (508 * 508,)))
ax[1].imshow(dummy_images[0, :, :, 13])



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_58/3786792693.py in <cell line: 0>()
      1 model_dummy = Sequential()
----> 2 model_dummy.add(
      3     Conv2D(
      4         18,
      5         kernel_initializer=weight_init,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_58/1699455035.py in weight_init(shape, dtype)
      4     kernel_sum = tf.stack(
      5         [
----> 6             kernel_first_order_1,
      7             kernel_first_order_2,
      8             kernel_first_order_3,

NameError: name 'kernel_first_order_1' is not defined

## === cell 12
from tensorflow.keras import layers


class AbsLayer(layers.Layer):
    def get_config(self):
        base_config = super(AbsLayer, self).get_config()
        return dict(list(base_config.items()))

    def __init__(self, **kwargs):
        super(AbsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AbsLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.abs(inputs)




## === cell 13
model_dummy_abs = Sequential()
model_dummy_abs.add(
    Conv2D(
        18,
        kernel_initializer=weight_init,
        kernel_size=5,
        padding="valid",
        input_shape=(512, 512, 1),
    )
)
model_dummy_abs.add(AbsLayer())
dummy_images_abs = model_dummy_abs.predict(data_dummy)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(np.reshape(dummy_images_abs[0, :, :, 1] * 255, (508 * 508,)))
ax[1].imshow(dummy_images_abs[0, :, :, 1])




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_58/447851677.py in <cell line: 0>()
      1 model_dummy_abs = Sequential()
----> 2 model_dummy_abs.add(
      3     Conv2D(
      4         18,
      5         kernel_initializer=weight_init,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_58/1699455035.py in weight_init(shape, dtype)
      4     kernel_sum = tf.stack(
      5         [
----> 6             kernel_first_order_1,
      7             kernel_first_order_2,
      8             kernel_first_order_3,

NameError: name 'kernel_first_order_1' is not defined

## === cell 14
def act_trunc(inputs):
    TRUNCATION_VAL = tf.constant(10.0 / 255.0, dtype=tf.float32)
    return tf.clip_by_value(inputs, -TRUNCATION_VAL, TRUNCATION_VAL)




## === cell 15
model_dummy_trunc = Sequential()
model_dummy_trunc.add(
    Conv2D(
        18,
        activation=act_trunc,
        kernel_initializer=weight_init,
        kernel_size=5,
        padding="valid",
        input_shape=(512, 512, 1),
    )
)
model_dummy_trunc.add(AbsLayer())
dummy_images_trunc = model_dummy_trunc.predict(data_dummy)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(np.reshape(dummy_images_trunc[0, :, :, 1] * 255, (508 * 508,)))
ax[1].imshow(dummy_images_trunc[0, :, :, 1])



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_58/2254188350.py in <cell line: 0>()
      1 model_dummy_trunc = Sequential()
----> 2 model_dummy_trunc.add(
      3     Conv2D(
      4         18,
      5         activation=act_trunc,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_58/1699455035.py in weight_init(shape, dtype)
      4     kernel_sum = tf.stack(
      5         [
----> 6             kernel_first_order_1,
      7             kernel_first_order_2,
      8             kernel_first_order_3,

NameError: name 'kernel_first_order_1' is not defined

## === cell 16
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
)

model = Sequential()
model.add(
    Conv2D(
        18,
        strides=(2, 2),
        kernel_initializer=weight_init,
        kernel_size=5,
        padding="valid",
        input_shape=(512, 512, 1),
    )
)
model.add(AbsLayer())
model.add(
    Conv2D(
        32,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(
    Conv2D(
        64,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=2, padding="same", kernel_initializer=he_normal()))
model.add(Activation("relu"))
model.add(
    Conv2D(
        128,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(
    Conv2D(
        256,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(
    Conv2D(
        512,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(
    Conv2D(
        1024,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_normal(),
    )
)
model.add(Activation("relu"))
model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation="softmax"))
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_58/2527941674.py in <cell line: 0>()
      4 
      5 model = Sequential()
----> 6 model.add(
      7     Conv2D(
      8         18,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
    120         self._layers.append(layer)
    121         if rebuild:
--> 122             self._maybe_rebuild()
    123         else:
    124             self.built = False

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in _maybe_rebuild(self)
    139         if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
    140             input_shape = self._layers[0].batch_shape
--> 141             self.build(input_shape)
    142         elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
    143             # We can build the Sequential model if the first layer has the

/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py in build_wrapper(*args, **kwargs)
    226             with obj._open_name_scope():
    227                 obj._path = current_path()
--> 228                 original_build_method(*args, **kwargs)
    229             # Record build config.
    230             signature = inspect.signature(original_build_method)

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    185         for layer in self._layers[1:]:
    186             try:
--> 187                 x = layer(x)
    188             except NotImplementedError:
    189                 # Can happen if shape inference is not implemented.

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_58/1699455035.py in weight_init(shape, dtype)
      4     kernel_sum = tf.stack(
      5         [
----> 6             kernel_first_order_1,
      7             kernel_first_order_2,
      8             kernel_first_order_3,

NameError: name 'kernel_first_order_1' is not defined

## === cell 17
model.fit(
    x=data_train,
    validation_data=data_valid,
    steps_per_epoch=1000,
    validation_steps=10,
    callbacks=[reduce_lr],
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_58/1378911474.py in <cell line: 0>()
----> 1 model.fit(
      2     x=data_train,
      3     validation_data=data_valid,
      4     steps_per_epoch=1000,
      5     validation_steps=10,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 18
test_path = os.path.join(os.getcwd(), "/kaggle/input/alaska2-image-steganalysis/Test/")
test_labels = [(os.path.join(test_path, f), -1) for f in os.listdir(test_path)]
data_test = create_dataset(test_labels, shuffle=False, augment=False)



## === cell 19
custom_obj = {"weight_init": weight_init, "AbsLayer": AbsLayer}
predictions = model.predict(data_test, verbose=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_58/2995767831.py in <cell line: 0>()
      1 custom_obj = {"weight_init": weight_init, "AbsLayer": AbsLayer}
----> 2 predictions = model.predict(data_test, verbose=1)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    168         if isinstance(self._layers[0], InputLayer):
    169             if self._layers[0].batch_shape != input_shape:
--> 170                 raise ValueError(
    171                     f"Sequential model '{self.name}' has already been "
    172                     "configured to use input shape "

ValueError: Sequential model 'sequential_3' has already been configured to use input shape (None, 512, 512, 1). You cannot build it with input_shape (20, 512, 512, 1)

## === cell 20
prediction_stego = [np.sum(pred[1:4]) for pred in predictions]

import pandas as pd

sub = pd.DataFrame({"Id": os.listdir(test_path), "Label": prediction_stego})
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_58/1479789966.py in <cell line: 0>()
----> 1 prediction_stego = [np.sum(pred[1:4]) for pred in predictions]
      2 
      3 import pandas as pd
      4 
      5 sub = pd.DataFrame({"Id": os.listdir(test_path), "Label": prediction_stego})

NameError: name 'predictions' is not defined
