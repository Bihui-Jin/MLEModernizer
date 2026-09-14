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
import shutil
import random
import numpy as np
import pandas as pd
import cv2

import seaborn as sns
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
os.environ["PYTHONHASHSEED"] = str(SEED)

INPUT_DIR = "../input/dog-breed-identification"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

WORK_DIR = "/kaggle/working"
NEW_TRAIN_DIR = os.path.join(WORK_DIR, "new_train")
NEW_VALID_DIR = os.path.join(WORK_DIR, "new_valid")
NEW_TEST_DIR = os.path.join(WORK_DIR, "new_test")



## === cell 1
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_CSV))
print("Example train files:", sorted(os.listdir(TRAIN_DIR))[:5])



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels.head(5)



## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
print("Num classes:", classes_num)
print("First 5 classes:", classes[:5])



## === cell 4
images_names = os.listdir(TRAIN_DIR)
images_num = len(images_names)
print(f"Number of train images: {images_num}")



## === cell 5
for d in [NEW_TRAIN_DIR, NEW_VALID_DIR, NEW_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

for sub_dir in classes:
    os.makedirs(os.path.join(NEW_TRAIN_DIR, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(NEW_VALID_DIR, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(NEW_TEST_DIR, sub_dir), exist_ok=True)

print("Created/verified split directories under:", WORK_DIR)



## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] = labels_jpg["id"].astype(str) + ".jpg"
grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:5])



## === cell 7
test_split = 0.1
valid_split = 0.2




## === cell 8
def file_count(root_dir):
    total = 0
    for c in classes:
        p = os.path.join(root_dir, c)
        if os.path.isdir(p):
            total += len(os.listdir(p))
    return total


already_split = (
    file_count(NEW_TRAIN_DIR) + file_count(NEW_VALID_DIR) + file_count(NEW_TEST_DIR)
) > 0

train_size = valid_size = test_size = 0

if not already_split:
    for breed_idx, (breed, breed_images) in enumerate(grouped_ids.items()):
        for img in breed_images:
            src = os.path.join(TRAIN_DIR, img)
            if not os.path.isfile(src):
                continue
            rnd_prob = np.random.rand()
            if rnd_prob <= test_split:
                dst = os.path.join(NEW_TEST_DIR, breed, img)
                test_size += 1
            elif rnd_prob <= (test_split + valid_split):
                dst = os.path.join(NEW_VALID_DIR, breed, img)
                valid_size += 1
            else:
                dst = os.path.join(NEW_TRAIN_DIR, breed, img)
                train_size += 1

            if not os.path.isfile(dst):
                shutil.copy(src, dst)

        clear_output(wait=True)
        print(f"Organized {breed_idx+1}/{classes_num} breeds: {breed}")

else:
    train_size = file_count(NEW_TRAIN_DIR)
    valid_size = file_count(NEW_VALID_DIR)
    test_size = file_count(NEW_TEST_DIR)
    print("Split directories already populated; skipping copy step.")

print("Split sizes:", train_size, valid_size, test_size)



## === cell 9
width, height, channels_num = 512, 512, 3

images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(os.path.join(TRAIN_DIR, img_filename))
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=12)
plt.show()



## === cell 10
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": 1 / 255.0,
}

img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(rescale=1 / 255.0)




## === cell 11
class Plotter(Callback):
    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")
        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")
        ax1.set_title("Loss vs Epochs")
        ax2.set_title("Accuracy vs Epochs")
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
        self.losses, self.val_losses, self.epochs = [], [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)
        self.plot()


plotter = Plotter()



## === cell 12
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 13
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(seed=layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 14
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




## === cell 15
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
model.summary()



## === cell 16
train_gen = img_gen.flow_from_directory(
    directory=NEW_TRAIN_DIR,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
    seed=SEED,
)

valid_gen = img_feed.flow_from_directory(
    directory=NEW_VALID_DIR,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
    seed=SEED,
)

steps_per_epoch = int(np.ceil(train_gen.samples / batch_size))
val_steps = int(np.ceil(valid_gen.samples / batch_size))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 17
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)



## === cell 18
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
breed_cols = [c for c in sample_sub.columns if c != "id"]

test_gen = ImageDataGenerator(rescale=1 / 255.0)
test_flow = test_gen.flow_from_directory(
    INPUT_DIR,
    target_size=(height, width),
    batch_size=1,
    shuffle=False,
    classes=["test"],  # tells generator to read from ../input/.../test
)

test_filenames = test_flow.filenames
test_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_filenames]
print("Num test images:", len(test_ids), "Expected:", len(sample_sub))



## === cell 19
preds = model.predict(test_flow, steps=len(test_ids), verbose=1)
preds = np.asarray(preds, dtype=np.float64)

sub = pd.DataFrame(preds, columns=list(classes))
sub.insert(0, "id", test_ids)

sub = sample_sub[["id"]].merge(sub, on="id", how="left")

for c in breed_cols:
    if c not in sub.columns:
        sub[c] = 1.0 / len(breed_cols)

sub = sub[["id"] + breed_cols]

prob = sub[breed_cols].to_numpy(dtype=np.float64)
nan_mask = ~np.isfinite(prob)
if nan_mask.any():
    prob[nan_mask] = 1.0 / len(breed_cols)

row_sums = prob.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
prob = prob / row_sums
sub.loc[:, breed_cols] = prob

out_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape, "Expected:", sample_sub.shape)
