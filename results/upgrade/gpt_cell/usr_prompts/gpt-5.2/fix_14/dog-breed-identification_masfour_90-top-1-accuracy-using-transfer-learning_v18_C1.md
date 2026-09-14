# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import sys
import subprocess

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
except Exception as e:
    print(
        "pip install protobuf skipped/failed (expected in Kaggle no-internet). Continuing.",
        repr(e),
    )

import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

from keras import backend as K
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Lambda,
    UpSampling2D,
    Concatenate,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.initializers import he_normal

from tf_keras.preprocessing.image import ImageDataGenerator

import shutil
import tensorflow as tf

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)


def fast_link_or_copy(src, dst):
    try:
        os.link(src, dst)  # hardlink if possible (fast, no extra space)
    except Exception:
        try:
            os.symlink(src, dst)  # fallback (also fast)
        except Exception:
            shutil.copy2(src, dst)  # final fallback (correct but slower)




## === cell 1
train_preview_dir = "../input/dog-breed-identification/train/"
print("Preview train files:", sorted(os.listdir(train_preview_dir))[:5])




## === cell 2
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels.head(5)




## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num




## === cell 4
train_dir = "../input/dog-breed-identification/train"  # the images directory
images_names = os.listdir(train_dir)  # names of the files in the directory
images_num = len(images_names)
print(f"Number of images: {images_num}")  # Number of training images




## === cell 5
new_train_dir = "/kaggle/working/new_train/"  # parent directoiry of the training set
new_test_dir = "/kaggle/working/new_test/"  # parent directory of the validation set (your internal "test")
new_valid_dir = "/kaggle/working/new_valid/"  # parent directory of the test set (your internal "valid")

for d in [new_train_dir, new_test_dir, new_valid_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print("Created:", new_train_dir, new_test_dir, new_valid_dir)




## === cell 6
for sub_dir in classes:
    os.mkdir(new_train_dir + sub_dir)
    os.mkdir(new_test_dir + sub_dir)
    os.mkdir(new_valid_dir + sub_dir)
print("Example class dirs:", sorted(os.listdir(new_train_dir))[:5])




## === cell 7
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] += ".jpg"  # add .jpg to each image id to get its filename

grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:5])




## === cell 8
test_split = 0.1
valid_split = 0.2




## === cell 9
train_size = 0
valid_size = 0
test_size = 0

rng = np.random.RandomState(SEED)

for breed_idx, (breed, breed_images) in enumerate(grouped_ids.items()):
    for img in breed_images:
        rnd_prob = rng.rand()
        src = os.path.join(train_dir, img)
        if rnd_prob <= test_split:
            dst = os.path.join(new_test_dir, breed, img)
            fast_link_or_copy(src, dst)
            test_size += 1
        elif rnd_prob <= (test_split + valid_split):
            dst = os.path.join(new_valid_dir, breed, img)
            fast_link_or_copy(src, dst)
            valid_size += 1
        else:
            dst = os.path.join(new_train_dir, breed, img)
            fast_link_or_copy(src, dst)
            train_size += 1

    if (breed_idx + 1) % 20 == 0 or (breed_idx + 1) == classes_num:
        print(f"Organized {breed_idx+1} out of {classes_num} breeds: {breed}")




## === cell 10
print("Split sizes:", train_size, valid_size, test_size)




## === cell 11
test_breed = classes[0]
print(
    "Example files in new_train for first class:",
    sorted(os.listdir(os.path.join(new_train_dir, test_breed)))[:5],
)




## === cell 12
width, height, channels = 512, 512, 3




## === cell 13
images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(train_dir + "/" + img_filename)
    images_samples[i] = (
        cv2.resize(src=img_bgr[:, :, [2, 1, 0]], dsize=(width, height)) / 255
    )
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)




## === cell 14
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=15)




## === cell 15
norm_factor = 1 / 255

transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": norm_factor,
}

img_gen = ImageDataGenerator(**transform_params)




## === cell 16
img_feed = ImageDataGenerator(rescale=1 / 255)




## === cell 17
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Augmentation Results", size=32)

for axs_col, img in enumerate(images_samples):
    viz_transoform_params = {
        "theta": np.random.randint(
            -transform_params["rotation_range"], transform_params["rotation_range"]
        ),
        "tx": np.random.uniform(0, transform_params["width_shift_range"]),
        "ty": np.random.uniform(0, transform_params["height_shift_range"]),
        "flip_horizontal": np.random.choice([True, False], p=[0.5, 0.5]),
    }

    aug_img = img_gen.apply_transform(img, viz_transoform_params)

    axs[0, axs_col].imshow(img)
    axs[0, axs_col].axis("off")
    axs[0, axs_col].set_title("Original Image", size=15)

    axs[1, axs_col].imshow(aug_img)
    axs[1, axs_col].axis("off")
    axs[1, axs_col].set_title("Augmented Image", size=15)




## === cell 18
class Plotter(Callback):
    def __init__(self, plot_every_n_epochs=5):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs

    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")

        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")

        ax1.set_title(f"Loss vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.set_title(f"Accuracy vs Epochs")
        ax2.set_xlabel("Epoches")
        ax2.set_ylabel("Accuracy")

        ax1.legend()
        ax2.legend()
        plt.show()

        print(
            f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
        )
        print(
            f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
        )

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.batch_no = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)
        if (epoch + 1) % self.plot_every_n_epochs == 0 or (epoch + 1) == 1:
            self.plot()

    def on_train_end(self, logs=None):
        self.plot()

    def load_plot_data(self, data):
        (
            self.losses,
            self.val_losses,
            self.epochs,
            self.batch_no,
            self.acc,
            self.val_acc,
        ) = data

    def get_plot_data(self):
        return [
            self.losses,
            self.val_losses,
            self.epochs,
            self.batch_no,
            self.acc,
            self.val_acc,
        ]


plotter = Plotter(plot_every_n_epochs=5)




## === cell 19
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)




## === cell 20
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)




## === cell 21
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 22
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 23
def create_model(shape):
    input_layer = Input(shape, name="input_layer")

    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 24
height, width, channels_num = 512, 512, 3
learning_rate = 0.004
epochs = 15
batch_size = 32




## === cell 25
model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
model.summary()




## === cell 26
train_gen = img_gen.flow_from_directory(
    directory=new_train_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
    seed=SEED,  # stability
)

valid_gen = img_feed.flow_from_directory(
    directory=new_valid_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
    seed=SEED,  # stability
)




## === cell 27
output_signature = (
    tf.TensorSpec(shape=(None, height, width, channels_num), dtype=tf.float32),
    tf.TensorSpec(shape=(None, classes_num), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_gen, output_signature=output_signature
)
valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_gen, output_signature=output_signature
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)

steps_per_epoch = int(np.ceil(train_gen.samples / batch_size))
val_steps = int(np.ceil(valid_gen.samples / batch_size))

history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)




## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mUnknownError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2351826410.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     24[0m [0mval_steps[0m [0;34m=[0m [0mint[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mceil[0m[0;34m([0m[0mvalid_gen[0m[0;34m.[0m[0msamples[0m [0;34m/[0m [0mbatch_size[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m [0;34m[0m[0m
[0;32m---> 26[0;31m history = model.fit(
[0m[1;32m     27[0m     [0mtrain_ds[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m     [0mepochs[0m[0;34m=[0m[0mepochs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py[0m in [0;36mquick_execute[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)[0m
[1;32m     57[0m       [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m" name: "[0m [0;34m+[0m [0mname[0m[0;34m[0m[0;34m[0m[0m
[1;32m     58[0m     [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 59[0;31m   [0;32mexcept[0m [0mTypeError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     60[0m     [0mkeras_symbolic_tensors[0m [0;34m=[0m [0;34m[[0m[0mx[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0minputs[0m [0;32mif[0m [0m_is_keras_symbolic_tensor[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m     [0;32mif[0m [0mkeras_symbolic_tensors[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mUnknownError[0m: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) UNKNOWN:  FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/new_train/curly-coated_retriever/3ad193d212c34fb9e5c77a1dfc99efe1.jpg'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 156, in __next__
    return self.next(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 168, in next
    return self._get_batches_of_transformed_samples(index_array)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 370, in _get_batches_of_transformed_samples
    img = image_utils.load_img(
          ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py", line 422, in load_img
    with open(path, "rb") as f:
         ^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/new_train/curly-coated_retriever/3ad193d212c34fb9e5c77a1dfc99efe1.jpg'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_4]]
  (1) UNKNOWN:  FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/new_train/curly-coated_retriever/3ad193d212c34fb9e5c77a1dfc99efe1.jpg'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 156, in __next__
    return self.next(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 168, in next
    return self._get_batches_of_transformed_samples(index_array)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/preprocessing/image.py", line 370, in _get_batches_of_transformed_samples
    img = image_utils.load_img(
          ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py", line 422, in load_img
    with open(path, "rb") as f:
         ^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/new_train/curly-coated_retriever/3ad193d212c34fb9e5c77a1dfc99efe1.jpg'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_33744]

## === cell 28
test_gen_internal = ImageDataGenerator(rescale=1 / 255)
test_flow_internal = test_gen_internal.flow_from_directory(
    new_test_dir,
    target_size=(512, 512),
    batch_size=1,
    shuffle=False,
    classes=list(classes),
)

internal_steps = test_flow_internal.samples
metrics = model.evaluate(test_flow_internal, steps=internal_steps, verbose=1)
m_names = model.metrics_names
print(f"{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}")
