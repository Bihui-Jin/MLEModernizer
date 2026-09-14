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

# 5. Target score

0.28106

# 6. Current score

5.08168

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.08168) has done: 'I fix the environment-breaking imports by switching the code to use `tf_keras` (which avoids the protobuf `MessageFactory.GetPrototype` crash seen with `keras==3.x` here) and ensure `os/shutil` are imported before use. I also remove the internet `wget` dependencies (no external downloads on Kaggle) and make the pipeline train the existing model end-to-end instead of trying to load remote weights/curves. To avoid huge filesystem copies and permission issues, I keep the same `flow_from_directory` approach but create a small symlinked split inside `/kaggle/working/` rather than copying all images. Finally, I generate `submission.csv` using `sample_submission.csv` to guarantee correct column order/length and `model.predict(...)` (replacing deprecated `evaluate_generator`), ensuring IDs align with filenames.'

# 9. Code solution

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
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

print("OK imports. Using tf_keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _reset_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


_reset_dir(work_base)
_reset_dir(new_train_dir)
_reset_dir(new_valid_dir)
_reset_dir(new_test_dir)

for sub_dir in classes:
    os.makedirs(os.path.join(new_train_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_valid_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, sub_dir), exist_ok=True)

print("Created split folders in:", work_base)



## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] = labels_jpg["id"].astype(str) + ".jpg"
grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:5])



## === cell 7
test_split = 0.1
valid_split = 0.2




## === cell 8
def safe_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


train_size = 0
valid_size = 0
test_size = 0

rng = np.random.RandomState(SEED)

for breed_idx, (breed, breed_images) in enumerate(grouped_ids.items()):
    for img in breed_images:
        rnd_prob = rng.rand()
        src_path = os.path.join(train_dir, img)

        if rnd_prob <= test_split:
            dst_path = os.path.join(new_test_dir, breed, img)
            safe_link(src_path, dst_path)
            test_size += 1
        elif rnd_prob <= (test_split + valid_split):
            dst_path = os.path.join(new_valid_dir, breed, img)
            safe_link(src_path, dst_path)
            valid_size += 1
        else:
            dst_path = os.path.join(new_train_dir, breed, img)
            safe_link(src_path, dst_path)
            train_size += 1

    clear_output(wait=True)
    print(f"Organized {breed_idx+1} out of {classes_num} breeds: {breed}")

print("Split sizes:", train_size, valid_size, test_size)



## === cell 9
test_breed = classes[0]
print("Example files:", os.listdir(os.path.join(new_train_dir, test_breed))[:5])



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
        self.plot()


plotter = Plotter()



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
train_gen = img_gen.flow_from_directory(
    directory=new_train_dir,
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
    directory=new_valid_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
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
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3975947943.py in <cell line: 0>()
      4 validation_steps = int(np.ceil(valid_gen.samples / batch_size))
      5 
----> 6 history = model.fit(
      7     train_gen,
      8     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py in load_img(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)
    420         if isinstance(path, pathlib.Path):
    421             path = str(path.resolve())
--> 422         with open(path, "rb") as f:
    423             img = pil_image.open(io.BytesIO(f.read()))
    424     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/dog_split/train/curly-coated_retriever/3ad193d212c34fb9e5c77a1dfc99efe1.jpg'

## === cell 19
test_gen_local = ImageDataGenerator(rescale=1 / 255.0)
test_flow_local = test_gen_local.flow_from_directory(
    new_test_dir, target_size=(height, width), batch_size=1, shuffle=False
)
metrics = model.evaluate(test_flow_local, steps=test_flow_local.samples, verbose=1)
m_names = model.metrics_names
print(f"{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2023824668.py in <cell line: 0>()
      5     new_test_dir, target_size=(height, width), batch_size=1, shuffle=False
      6 )
----> 7 metrics = model.evaluate(test_flow_local, steps=test_flow_local.samples, verbose=1)
      8 m_names = model.metrics_names
      9 print(f"{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}")

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py in load_img(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)
    420         if isinstance(path, pathlib.Path):
    421             path = str(path.resolve())
--> 422         with open(path, "rb") as f:
    423             img = pil_image.open(io.BytesIO(f.read()))
    424     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/dog_split/test/affenpinscher/100f347ec78a42a9e7c2418e4beb3f6a.jpg'

## === cell 20
one_hot_map = train_gen.class_indices
print("First 10 class indices:", list(one_hot_map.items())[:10])



## === cell 21
sample_sub = pd.read_csv(sample_sub_path)
test_ids = (
    sample_sub["id"].astype(str).tolist()
)  # required length/order for valid submission

test_gen = ImageDataGenerator(rescale=1 / 255.0)
test_flow = test_gen.flow_from_directory(
    input_dir,
    target_size=(height, width),
    batch_size=1,
    shuffle=False,
    classes=["test"],
)

flow_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_flow.filenames]

print("Test flow samples:", test_flow.samples)
print("Sample submission rows:", len(test_ids))



## === cell 22
preds = model.predict(test_flow, steps=test_flow.samples, verbose=1)
if preds.shape[1] != (sample_sub.shape[1] - 1):
    raise ValueError(
        f"Pred dim {preds.shape[1]} != submission classes {sample_sub.shape[1]-1}"
    )

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
