# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import shutil
import random
import hashlib
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import clear_output

import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
except Exception:
    tf = None

print("OK imports. Using tf_keras:", keras.__version__)
print("KERAS_BACKEND:", os.environ.get("KERAS_BACKEND"))
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)



## === cell 1
input_dir = "../input/dog-breed-identification"
train_dir = os.path.join(input_dir, "train")
test_dir = os.path.join(input_dir, "test")

labels_path = os.path.join(input_dir, "labels.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")

print(train_dir, test_dir, labels_path, sample_sub_path)



## === cell 2
labels = pd.read_csv(labels_path)
labels.head()



## === cell 3
classes = np.unique(labels.breed.values)
classes_num = classes.size
print("Num classes:", classes_num)



## === cell 4
images_names = os.listdir(train_dir)
images_num = len(images_names)
print(f"Number of training images: {images_num}")



## === cell 5
work_base = "/kaggle/working/dog_split"
new_train_dir = os.path.join(work_base, "train")
new_valid_dir = os.path.join(work_base, "valid")
new_test_dir = os.path.join(work_base, "test")

print(
    "Skipping on-disk split folder creation to eliminate heavy I/O. Using in-memory split instead."
)



## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["filename"] = labels_jpg["id"].astype(str) + ".jpg"
print(
    classes[0],
    labels_jpg[labels_jpg["breed"] == classes[0]]["filename"].head().tolist(),
)



## === cell 7
test_split = 0.1
valid_split = 0.2




## === cell 8
def deterministic_bucket_vec(filenames, seed=SEED):
    vals = []
    prefix = str(seed) + "_"
    for fn in filenames:
        h = hashlib.md5((prefix + fn).encode("utf-8")).hexdigest()
        vals.append(int(h[:8], 16) / float(16**8))
    return np.asarray(vals, dtype=np.float64)


buckets = deterministic_bucket_vec(labels_jpg["filename"].tolist(), seed=SEED)
labels_jpg = labels_jpg.assign(_bucket=buckets)

labels_jpg["_split"] = np.where(
    labels_jpg["_bucket"] <= test_split,
    "test",
    np.where(labels_jpg["_bucket"] <= (test_split + valid_split), "valid", "train"),
)

train_df = labels_jpg[labels_jpg["_split"] == "train"][
    ["filename", "breed"]
].reset_index(drop=True)
valid_df = labels_jpg[labels_jpg["_split"] == "valid"][
    ["filename", "breed"]
].reset_index(drop=True)
test_df_internal = labels_jpg[labels_jpg["_split"] == "test"][
    ["filename", "breed"]
].reset_index(drop=True)

print(
    "Split sizes:",
    len(train_df),
    len(valid_df),
    len(test_df_internal),
    "missing/skipped:",
    0,
)



## === cell 9
test_breed = classes[0]
example_files = (
    train_df.loc[train_df["breed"] == test_breed, "filename"].head(5).tolist()
)
print("Example files:", example_files)



## === cell 10
width, height, channels_num = 512, 512, 3



## === cell 11
images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(os.path.join(train_dir, img_filename))
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=15)
plt.show()



## === cell 12
norm_factor = 1 / 255.0
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
img_feed = ImageDataGenerator(rescale=1 / 255.0)




## === cell 13
class Plotter(Callback):
    def __init__(self, plot_every=5):
        super().__init__()
        self.plot_every = int(plot_every)

    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")

        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")

        ax1.set_title("Loss vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.set_title("Accuracy vs Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        ax1.legend()
        ax2.legend()
        plt.show()

        if self.epochs:
            print(
                f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
            )
            print(
                f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
            )

    def on_train_begin(self, logs=None):
        self.losses, self.val_losses, self.epochs = [], [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)

        if (
            (epoch == 0)
            or ((epoch + 1) % self.plot_every == 0)
            or (epoch + 1 == getattr(self.params, "epochs", epoch + 1))
        ):
            self.plot()


plotter = Plotter(plot_every=5)



## === cell 14
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 15
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x


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




## === cell 16
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
model.summary()



## === cell 17
train_gen = img_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="filename",
    y_col="breed",
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
    seed=SEED,
)

valid_gen = img_feed.flow_from_dataframe(
    dataframe=valid_df,
    directory=train_dir,
    x_col="filename",
    y_col="breed",
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False,
    interpolation="nearest",
    seed=SEED,
)



## === cell 18
steps_per_epoch = int(np.ceil(train_gen.samples / batch_size))
validation_steps = int(np.ceil(valid_gen.samples / batch_size))

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
    workers=max(1, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=16,
)



## === cell 19
backbone = None
for lyr in model.layers:
    if isinstance(lyr, InceptionResNetV2):
        backbone = lyr
        break

unfreeze_last_n = 60  # small, conservative fine-tune to improve score materially
trainable_layers = 0
for lyr in model.layers[::-1]:
    if "inception_resnet_v2" in lyr.name.lower():
        if ("batch_normalization" in lyr.name.lower()) or ("bn" in lyr.name.lower()):
            lyr.trainable = False
            continue
        if trainable_layers < unfreeze_last_n:
            lyr.trainable = True
            trainable_layers += 1
        else:
            lyr.trainable = False

print("Unfroze backbone layers:", trainable_layers)

fine_tune_lr = learning_rate * 0.1
model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

history_ft = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=max(1, epochs // 2),
    validation_data=valid_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
    workers=max(1, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=16,
)



## === cell 20
one_hot_map = train_gen.class_indices
print("First 10 class indices:", list(one_hot_map.items())[:10])



## === cell 21
sample_sub = pd.read_csv(sample_sub_path)
test_ids = (
    sample_sub["id"].astype(str).tolist()
)  # required length/order for valid submission

test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"filename": test_files})
test_gen = ImageDataGenerator(rescale=1 / 255.0)
test_flow = test_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="filename",
    y_col=None,
    target_size=(height, width),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False,
    interpolation="nearest",
)

flow_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_flow.filenames]

print("Test flow samples:", test_flow.samples)
print("Sample submission rows:", len(test_ids))



## === cell 22
preds = model.predict(
    test_flow,
    steps=test_flow.samples,
    verbose=1,
    workers=max(1, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=16,
)
if preds.shape[1] != (sample_sub.shape[1] - 1):
    raise ValueError(
        f"Pred dim {preds.shape[1]} != submission classes {sample_sub.shape[1]-1}"
    )

eps = 1e-7
preds = np.clip(preds, eps, 1.0 - eps)
preds = preds / preds.sum(axis=1, keepdims=True)

pred_df = pd.DataFrame(preds, columns=sample_sub.columns[1:])
pred_df.insert(0, "id", flow_ids)

pred_df = pred_df.set_index("id").reindex(test_ids)
if pred_df.isnull().any().any():
    n_classes = sample_sub.shape[1] - 1
    pred_df = pred_df.fillna(1.0 / n_classes)

submission = pred_df.reset_index()
submission.columns = sample_sub.columns  # enforce exact header order

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())



## === cell 23
assert out_path.endswith(".csv")
assert submission.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert list(submission.columns) == list(
    sample_sub.columns
), "Submission columns mismatch"
print("Submission validated.")
