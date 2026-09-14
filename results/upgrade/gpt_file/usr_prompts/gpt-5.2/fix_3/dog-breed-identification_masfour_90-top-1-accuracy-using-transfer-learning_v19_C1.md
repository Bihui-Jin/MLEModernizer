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
from tf_keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
INPUT_DIR = "../input/dog-breed-identification"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/dog-breed-identification"

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

print("INPUT_DIR:", INPUT_DIR)
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels.head(5)



## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num



## === cell 4
train_dir = TRAIN_DIR  # the images directory
images_names = os.listdir(train_dir)
images_num = len(images_names)
print(f"Number of images: {images_num}")



## === cell 5
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
new_train_dir = os.path.join(WORK_DIR, "new_train")
new_test_dir = os.path.join(WORK_DIR, "new_test")
new_valid_dir = os.path.join(WORK_DIR, "new_valid")

for d in [new_train_dir, new_test_dir, new_valid_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print(new_train_dir, new_test_dir, new_valid_dir)



## === cell 6
for sub_dir in classes:
    os.makedirs(os.path.join(new_train_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_valid_dir, sub_dir), exist_ok=True)

len(os.listdir(new_train_dir)), "breed subdirs created"



## === cell 7
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] += ".jpg"
grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:3])



## === cell 8
test_split = 0.1
valid_split = 0.2



## === cell 9
rng = np.random.RandomState(SEED)

train_size = 0
valid_size = 0
holdout_size = 0

for breed_idx, (breed, breed_images) in enumerate(grouped_ids.items()):
    for img in breed_images:
        rnd_prob = rng.rand()
        src_path = os.path.join(train_dir, img)
        if rnd_prob <= test_split:
            shutil.copy(src_path, os.path.join(new_test_dir, breed))
            holdout_size += 1
        elif rnd_prob <= (test_split + valid_split):
            shutil.copy(src_path, os.path.join(new_valid_dir, breed))
            valid_size += 1
        else:
            shutil.copy(src_path, os.path.join(new_train_dir, breed))
            train_size += 1

    clear_output(wait=True)
    print(f"Organized {breed_idx+1} out of {classes_num} breeds: {breed}")

print("Split sizes:", train_size, valid_size, holdout_size)



## === cell 10
width, height, channels = 512, 512, 3

images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = rng.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(os.path.join(train_dir, img_filename))
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)



## === cell 11
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=15)
plt.show()



## === cell 12
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "preprocessing_function": preprocess_input,
}
img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(preprocessing_function=preprocess_input)



## === cell 13
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Augmentation Results", size=24)

for axs_col, img in enumerate(images_samples):
    viz_transoform_params = {
        "theta": rng.randint(
            -transform_params["rotation_range"], transform_params["rotation_range"] + 1
        ),
        "tx": rng.uniform(
            -transform_params["width_shift_range"],
            transform_params["width_shift_range"],
        ),
        "ty": rng.uniform(
            -transform_params["height_shift_range"],
            transform_params["height_shift_range"],
        ),
        "flip_horizontal": rng.choice([True, False]),
    }
    aug_img = img_gen.apply_transform(img, viz_transoform_params)

    axs[0, axs_col].imshow(img)
    axs[0, axs_col].axis("off")
    axs[0, axs_col].set_title("Original", size=12)

    axs[1, axs_col].imshow(aug_img)
    axs[1, axs_col].axis("off")
    axs[1, axs_col].set_title("Augmented", size=12)

plt.show()




## === cell 14
class Plotter(Callback):
    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)
        self.plot()


plotter = Plotter()



## === cell 15
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 16
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(seed=layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 17
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




## === cell 18
height, width, channels_num = 512, 512, 3
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
model.summary()



## === cell 19
train_gen = img_gen.flow_from_directory(
    directory=new_train_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
    interpolation="nearest",
)

valid_gen = img_feed.flow_from_directory(
    directory=new_valid_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
    interpolation="nearest",
)



## === cell 20
history = model.fit(
    train_gen, epochs=epochs, validation_data=valid_gen, callbacks=callbacks, verbose=1
)



## === cell 21
holdout_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
holdout_flow = holdout_gen.flow_from_directory(
    new_test_dir,
    target_size=(height, width),
    batch_size=1,
    shuffle=False,
    classes=list(classes),
    class_mode="categorical",
)

metrics = model.evaluate(holdout_flow, steps=holdout_size, verbose=1)
m_names = model.metrics_names
print(f"{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}")



## === cell 22
one_hot_map = train_gen.class_indices
list(one_hot_map.items())[:5], len(one_hot_map)



## === cell 23
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
breed_cols = [c for c in sample_sub.columns if c != "id"]

test_ids = sample_sub["id"].astype(str).tolist()
test_df = pd.DataFrame({"filename": [f"{tid}.jpg" for tid in test_ids]})

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_flow = test_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col="filename",
    y_col=None,
    target_size=(height, width),
    color_mode="rgb",
    class_mode=None,
    batch_size=32,
    shuffle=False,
    seed=SEED,
)

pred = model.predict(
    test_flow, steps=int(np.ceil(test_flow.samples / test_flow.batch_size)), verbose=1
)
pred = pred[: len(sample_sub)]  # safety

idx_to_class = {v: k for k, v in one_hot_map.items()}
model_class_order = [idx_to_class[i] for i in range(classes_num)]

pred_df = pd.DataFrame(pred, columns=model_class_order)

pred_df = pred_df.reindex(columns=breed_cols).fillna(1.0 / len(breed_cols))

eps = 1e-7
pred_df = pred_df.clip(lower=eps, upper=1.0 - eps)
pred_df = pred_df.div(pred_df.sum(axis=1), axis=0)

submission = pd.concat([sample_sub[["id"]].copy(), pred_df], axis=1)
assert submission.shape == sample_sub.shape, (submission.shape, sample_sub.shape)
assert list(submission.columns) == list(sample_sub.columns)

out_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
