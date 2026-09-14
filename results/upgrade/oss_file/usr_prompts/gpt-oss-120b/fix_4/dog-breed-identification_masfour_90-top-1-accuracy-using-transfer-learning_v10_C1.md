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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import shutil
import numpy as np
import pandas as pd
import cv2
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
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.applications import InceptionResNetV2
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "../input/dog-breed-identification/train"
labels_path = "../input/dog-breed-identification/labels.csv"

labels = pd.read_csv(labels_path)
labels.head(5)



## === cell 2
classes = np.unique(labels.breed)
classes_num = classes.size
print("Number of breeds:", classes_num)



## === cell 3
base_dir = "./dog_breed_split"
new_train_dir = os.path.join(base_dir, "train")
new_valid_dir = os.path.join(base_dir, "valid")
new_test_dir = os.path.join(base_dir, "test")

for d in [new_train_dir, new_valid_dir, new_test_dir]:
    os.makedirs(d, exist_ok=True)

for breed in classes:
    os.makedirs(os.path.join(new_train_dir, breed), exist_ok=True)
    os.makedirs(os.path.join(new_valid_dir, breed), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, breed), exist_ok=True)



## === cell 4
test_split = 0.1
valid_split = 0.2

train_size = valid_size = test_size = 0
labels_jpg = labels.copy()
labels_jpg["filename"] = labels_jpg["id"] + ".jpg"

grouped = labels_jpg.groupby("breed")["filename"].apply(list).to_dict()

for breed, files in grouped.items():
    for f in files:
        rnd = np.random.rand()
        src = os.path.join(train_dir, f)
        if rnd < test_split:
            dst = os.path.join(new_test_dir, breed, f)
            test_size += 1
        elif rnd < test_split + valid_split:
            dst = os.path.join(new_valid_dir, breed, f)
            valid_size += 1
        else:
            dst = os.path.join(new_train_dir, breed, f)
            train_size += 1
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)
    clear_output(wait=True)
    print(f"Split done for breed: {breed}")

print(f"Train: {train_size}, Valid: {valid_size}, Test: {test_size}")



## === cell 5
height, width, channels = 512, 512, 3



## === cell 6
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
img_gen = ImageDataGenerator(**transform_params)  # training (augmented)
img_feed = ImageDataGenerator(rescale=norm_factor)  # validation / test (no aug)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1712503814.py in <cell line: 0>()
     11     "rescale": norm_factor,
     12 }
---> 13 img_gen = ImageDataGenerator(**transform_params)  # training (augmented)
     14 img_feed = ImageDataGenerator(rescale=norm_factor)  # validation / test (no aug)
     15 

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
class Plotter(Callback):
    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
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
            f"Epoch {self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}% "
            f"train_loss={self.losses[-1]:.5f}"
        )
        print(
            f"Epoch {self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}% "
            f"val_loss={self.val_losses[-1]:.5f}"
        )

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))
        self.epochs.append(epoch)
        self.plot()


plotter = Plotter()




## === cell 8
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
    base = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in base.layers:
        layer.trainable = False
    x = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(base.output)
    x = Flatten(name="Flatten1")(x)
    x = BatchNormalization()(x)
    x = dense_block(x, 512, 1)
    x = dense_block(x, 512, 2)
    x = dense_block(x, 1024, 3)
    x = Dense(classes_num, name="Dense_Final")(x)
    output = Activation("softmax")(x)
    model = Model(inputs=input_layer, outputs=output)
    return model




## === cell 9
learning_rate = 0.001
epochs = 5  # reduced for quick run
batch_size = 32

model = create_model((height, width, channels))
model.compile(
    optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 10
train_gen = img_gen.flow_from_directory(
    directory=new_train_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    interpolation="nearest",
)

valid_gen = img_feed.flow_from_directory(
    directory=new_valid_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False,
    interpolation="nearest",
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/243997552.py in <cell line: 0>()
----> 1 train_gen = img_gen.flow_from_directory(
      2     directory=new_train_dir,
      3     target_size=(height, width),
      4     color_mode="rgb",
      5     classes=list(classes),

NameError: name 'img_gen' is not defined

## === cell 11
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
)

e_stop = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", restore_best_weights=True
)

callbacks = [plotter, plateau_reduce, e_stop]



## === cell 12
model.fit(
    train_gen,
    steps_per_epoch=train_size // batch_size,
    validation_data=valid_gen,
    validation_steps=valid_size // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=0,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2247561226.py in <cell line: 0>()
      1 model.fit(
----> 2     train_gen,
      3     steps_per_epoch=train_size // batch_size,
      4     validation_data=valid_gen,
      5     validation_steps=valid_size // batch_size,

NameError: name 'train_gen' is not defined

## === cell 13
test_gen = ImageDataGenerator(rescale=norm_factor)
test_flow = test_gen.flow_from_directory(
    directory=new_test_dir,
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),  # sub‑folders exist but labels are ignored
    class_mode=None,
    batch_size=1,
    shuffle=False,
    interpolation="nearest",
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1383159664.py in <cell line: 0>()
----> 1 test_gen = ImageDataGenerator(rescale=norm_factor)
      2 test_flow = test_gen.flow_from_directory(
      3     directory=new_test_dir,
      4     target_size=(height, width),
      5     color_mode="rgb",

NameError: name 'ImageDataGenerator' is not defined

## === cell 14
preds = model.predict(test_flow, verbose=0)  # predict all samples automatically

filenames = test_flow.filenames
ids = [os.path.basename(f).split(".")[0] for f in filenames]



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3186341029.py in <cell line: 0>()
----> 1 preds = model.predict(test_flow, verbose=0)  # predict all samples automatically
      2 
      3 filenames = test_flow.filenames
      4 ids = [os.path.basename(f).split(".")[0] for f in filenames]
      5 

NameError: name 'test_flow' is not defined

## === cell 15
submission = pd.DataFrame(preds, columns=classes)
submission.insert(0, "id", ids)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3135137852.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(preds, columns=classes)
      2 submission.insert(0, "id", ids)
      3 submission_path = "submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}, shape: {submission.shape}")

NameError: name 'preds' is not defined
