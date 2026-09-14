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

3.7

# 3. Installed packages



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

4.88533

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import time
import datetime
import random
import re
import gc
from glob import glob

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # safe in headless Kaggle runs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import class_weight as cw

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Conv2D,
    MaxPool2D,
    GlobalAveragePooling2D,
)
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
from keras.optimizers import Adam

from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

for p in ("../input", "/kaggle/input"):
    if os.path.exists(p):
        print("Listing", p, "->", os.listdir(p)[:10])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def date_time(x):
    if x == 1:
        return "Timestamp: {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
    if x == 2:
        return "Timestamp: {:%Y-%b-%d %H:%M:%S}".format(datetime.datetime.now())
    if x == 3:
        return "Date now: %s" % datetime.datetime.now()
    if x == 4:
        return "Date today: %s" % datetime.date.today()




## === cell 2
input_directory = r"../input/dog-breed-identification/"
if not os.path.exists(input_directory):
    input_directory = r"/kaggle/input/dog-breed-identification/"

output_directory = r"../output/"
if not os.path.exists(output_directory):
    output_directory = r"/kaggle/working/"

training_dir = os.path.join(input_directory, "train")
testing_dir = os.path.join(input_directory, "test")

os.makedirs(output_directory, exist_ok=True)

figure_directory = os.path.join(output_directory, "figures")
os.makedirs(figure_directory, exist_ok=True)

file_name_pred_batch = os.path.join(figure_directory, "result")
file_name_pred_sample = os.path.join(figure_directory, "sample")

print("input_directory:", input_directory)
print("output_directory:", output_directory)
print("training_dir exists:", os.path.exists(training_dir))
print("testing_dir exists:", os.path.exists(testing_dir))



## === cell 3
train_df = pd.read_csv(os.path.join(input_directory, "labels.csv"))
train_df.rename(columns={"breed": "label"}, inplace=True)
train_df["id"] = train_df["id"].apply(lambda x: x + "." + "jpg")
train_df.head()



## === cell 4
classes = list(train_df["label"].unique())
classes.sort()
len(classes), classes[:5]



## === cell 5
test_files = os.listdir(testing_dir)
test_df = pd.DataFrame({"id": test_files, "label": "boston_bull"})
test_df.head()



## === cell 6
len(train_df), len(test_df)




## === cell 7
def plot_image(file, directory=None, sub=False, aspect=None, title=False):
    path = directory + "/" + file
    img = plt.imread(path)
    plt.imshow(img, aspect=aspect)
    if title:
        plt.title(file)
    plt.xticks([])
    plt.yticks([])
    if sub:
        plt.show()


def plot_img_dir(directory=training_dir, count=5):
    selected_files = random.sample(os.listdir(directory), count)
    ncols = 5
    nrows = count // ncols if count % ncols == 0 else count // ncols + 1
    figsize = (20, ncols * nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    params = {
        "figure.figsize": figsize,
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
    }
    plt.rcParams.update(params)

    for i, file in enumerate(selected_files):
        plt.subplot(nrows, ncols, i + 1)
        plot_image(file, directory, aspect=None)

    plt.tight_layout()
    plt.show()


def plot_img_df(
    directory=None, df=None, filename="id", label="label", count=5, num_cat=-1
):
    label_map = {}
    classes_local = list(set(df[label]))

    for l in classes_local:
        label_map[l] = df[df[label] == l][filename]
        label_map[l] = label_map[l].sample(count, replace=True)

    ncols = 5
    nrows = count // ncols if count % ncols == 0 else count // ncols + 1
    figsize = (20, ncols * nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    params = {
        "figure.figsize": figsize,
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
    }
    plt.rcParams.update(params)

    i = 0
    if num_cat == -1:
        print("Showing {} classes...".format(len(label_map)))
    else:
        print("Showing {} classes...".format(num_cat))

    for lab in label_map:
        if num_cat == i:
            break
        label2 = re.sub("_", " ", lab).title()
        print(str(i + 1) + ". " + label2)

        for j, (_, file) in enumerate(label_map[lab].items()):
            plt.subplot(nrows, 5, j + 1)
            plot_image(file, directory, aspect="auto")

        plt.tight_layout()
        plt.show()
        i += 1




## === cell 8
try:
    plot_img_df(
        directory=training_dir,
        df=train_df,
        filename="id",
        label="label",
        count=5,
        num_cat=2,
    )
except Exception as e:
    print("EDA plotting skipped:", repr(e))



## === cell 9
train_df2 = train_df.copy()
train_df2["label"] = (
    train_df["label"].apply(lambda x: re.sub("_", " ", x)).apply(lambda x: x.title())
)
classes2 = np.array(sorted(train_df2["label"].unique()))
rows = max(5, int(np.ceil((len(classes2) - 1) / 4.0)))



## === cell 10
try:
    plt.figure(figsize=(18, rows))
    _ = sns.countplot(
        y="label", data=train_df2, order=train_df2["label"].value_counts().index
    )
    plt.title("Countplot sorted by Value count of Categories")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, "countplot_valuecount.png"))
    plt.close()
except Exception as e:
    print("Countplot skipped:", repr(e))



## === cell 11
try:
    plt.figure(figsize=(18, rows))
    _ = sns.countplot(y="label", data=train_df2, order=classes2)
    plt.title("Countplot Sorted by Alphabetical Order of Category Names")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, "countplot_alpha.png"))
    plt.close()
except Exception as e:
    print("Countplot alpha skipped:", repr(e))




## === cell 12
def get_weight(y, binary=True, n_samples=-1):
    if binary is False:
        return cw.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
    else:
        d = {x: y.count(x) for x in set(y)}
        num_class_temp = 2
        class_weights = {}
        for cls in d:
            count = d.get(cls)
            class_weights[cls] = n_samples / (num_class_temp * count)
        return class_weights


def get_data(
    batch_size=32,
    target_size=(299, 299),
    class_mode="categorical",
    training_dir=training_dir,
    testing_dir=testing_dir,
    x_col="id",
    y_col="label",
):
    print("Preprocessing and Generating Data Batches.......\n")

    rescale = 1.0 / 255.0

    validation_batch_size = batch_size * 5
    test_batch_size = batch_size * 5

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=45,
        shear_range=15,
        rescale=rescale,
        validation_split=0.25,
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        training_dir,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset="training",
    )

    validation_generator = train_datagen.flow_from_dataframe(
        train_df,
        training_dir,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=validation_batch_size,
        shuffle=True,
        seed=42,
        subset="validation",
    )

    test_datagen = ImageDataGenerator(rescale=rescale)

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        testing_dir,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=test_batch_size,
        shuffle=False,
        seed=42,
    )

    class_weights = get_weight(train_generator.classes, binary=False)

    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    print("\nPreprocessing and Data Batch Generation Completed.\n")

    return (
        train_generator,
        validation_generator,
        test_generator,
        class_weights,
        steps_per_epoch,
        validation_steps,
    )




## === cell 13
def get_model(
    model_name,
    input_shape=(96, 96, 3),
    num_class=2,
    weights="imagenet",
    dense_units=1024,
):
    inputs = Input(input_shape)

    if model_name == "Xception":
        base_model = Xception(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "ResNet50":
        base_model = ResNet50(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "InceptionV3":
        base_model = InceptionV3(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "DenseNet201":
        base_model = DenseNet201(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(
            include_top=False, weights=weights, input_shape=input_shape
        )
    else:
        raise ValueError("Unsupported model_name: {}".format(model_name))

    x = base_model(inputs)
    x = Dropout(0.8)(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)

    if num_class > 1:
        outputs = Dense(num_class, activation="softmax")(x)
    else:
        outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model




## === cell 14
def get_conv_model(num_class=2, input_shape=(150, 150, 3)):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            input_shape=(32, 32, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(1e-4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4))
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(
        Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4))
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4))
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    if num_class > 1:
        model.add(Dense(num_class, activation="softmax"))
    else:
        model.add(Dense(num_class, activation="sigmoid"))

    return model




## === cell 15
main_model_dir = os.path.join(output_directory, "models")
main_log_dir = os.path.join(output_directory, "logs")

os.makedirs(main_model_dir, exist_ok=True)
os.makedirs(main_log_dir, exist_ok=True)

model_dir = os.path.join(main_model_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
log_dir = os.path.join(main_log_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

model_file = os.path.join(model_dir, "{epoch:02d}-val_loss-{val_loss:.4f}.hdf5")
print("model_dir:", model_dir)
print("log_dir:", log_dir)



## === cell 16
print("Settting Callbacks")


def step_decay(epoch, lr):
    lrate = lr
    if epoch == 2:
        lrate = 0.0001
    return lrate


checkpoint = ModelCheckpoint(
    model_file, monitor="val_loss", save_best_only=True, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=1, min_lr=0.0000001, verbose=1
)

learning_rate_scheduler = LearningRateScheduler(step_decay, verbose=1)

callbacks = [reduce_lr, early_stopping, checkpoint]
print("Set Callbacks at ", date_time(1))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1614910702.py in <cell line: 0>()
      9 
     10 
---> 11 checkpoint = ModelCheckpoint(
     12     model_file, monitor="val_loss", save_best_only=True, verbose=1
     13 )

NameError: name 'ModelCheckpoint' is not defined

## === cell 17
dim = 299
input_shape = (dim, dim, 3)
num_class = len(classes)
weights = "imagenet"
dense_units = 256



## === cell 18
print("Getting Base Model", date_time(1))
model = get_model(
    model_name="InceptionV3",
    input_shape=input_shape,
    num_class=num_class,
    weights=weights,
    dense_units=dense_units,
)
print("Loaded Base Model", date_time(1))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/792874311.py in <cell line: 0>()
      1 print("Getting Base Model", date_time(1))
----> 2 model = get_model(
      3     model_name="InceptionV3",
      4     input_shape=input_shape,
      5     num_class=num_class,

/tmp/ipykernel_11/1156628288.py in get_model(model_name, input_shape, num_class, weights, dense_units)
     17         )
     18     elif model_name == "InceptionV3":
---> 19         base_model = InceptionV3(
     20             include_top=False, weights=weights, input_shape=input_shape
     21         )

NameError: name 'InceptionV3' is not defined

## === cell 19
batch_size = 64
class_mode = "categorical"
target_size = (dim, dim)
y_col = "label"

(
    train_generator,
    validation_generator,
    test_generator,
    class_weights,
    steps_per_epoch,
    validation_steps,
) = get_data(
    batch_size=batch_size, target_size=target_size, class_mode=class_mode, y_col=y_col
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/999696309.py in <cell line: 0>()
     11     steps_per_epoch,
     12     validation_steps,
---> 13 ) = get_data(
     14     batch_size=batch_size, target_size=target_size, class_mode=class_mode, y_col=y_col
     15 )

/tmp/ipykernel_11/3428405337.py in get_data(batch_size, target_size, class_mode, training_dir, testing_dir, x_col, y_col)
     31     test_batch_size = batch_size * 5
     32 
---> 33     train_datagen = ImageDataGenerator(
     34         horizontal_flip=True,
     35         rotation_range=45,

NameError: name 'ImageDataGenerator' is not defined

## === cell 20
print("Compliling Model ...")

learning_rate = 0.0001
optimizer = Adam(learning_rate)

loss = "categorical_crossentropy"
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Completed Model Compilation.\n")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3859454225.py in <cell line: 0>()
      2 
      3 learning_rate = 0.0001
----> 4 optimizer = Adam(learning_rate)
      5 
      6 loss = "categorical_crossentropy"

NameError: name 'Adam' is not defined

## === cell 21
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

verbose = 1
epochs = 1  # kept identical to provided code (do not change training approach)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3894032956.py in <cell line: 0>()
----> 1 steps_per_epoch = len(train_generator)
      2 validation_steps = len(validation_generator)
      3 
      4 verbose = 1
      5 epochs = 1  # kept identical to provided code (do not change training approach)

NameError: name 'train_generator' is not defined

## === cell 22
print("Training started at", date_time(1))
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=verbose,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weights,
)
print("Training finished at", date_time(1))




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3400038240.py in <cell line: 0>()
      1 # --- Core training loop (was missing in provided cells; required to produce non-random predictions)
      2 print("Training started at", date_time(1))
----> 3 history = model.fit(
      4     train_generator,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'model' is not defined

## === cell 23
def plot_performance(history=None, figure_directory=None):
    xlabel = "Epoch"
    legends = ["Training", "Validation"]
    ylim_pad = [0, 0]

    plt.figure(figsize=(20, 5))

    acc_key = "acc" if "acc" in history.history else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"

    y1 = history.history[acc_key]
    y2 = history.history[val_acc_key]

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)
    plt.plot(y1)
    plt.plot(y2)
    plt.title("Model Accuracy\n" + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()

    y1 = history.history["loss"]
    y2 = history.history["val_loss"]

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]

    plt.subplot(122)
    plt.plot(y1)
    plt.plot(y2)
    plt.title("Model Loss\n" + date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()

    if figure_directory:
        plt.savefig(os.path.join(figure_directory, "history.png"))
        plt.close()
    else:
        plt.show()


try:
    plot_performance(history=history, figure_directory=figure_directory)
except Exception as e:
    print("Performance plot skipped:", repr(e))



## === cell 24
label_map = train_generator.class_indices
label_map_inv = {v: k for k, v in label_map.items()}
len(label_map), list(label_map.items())[:5]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2903082243.py in <cell line: 0>()
----> 1 label_map = train_generator.class_indices
      2 label_map_inv = {v: k for k, v in label_map.items()}
      3 len(label_map), list(label_map.items())[:5]
      4 

NameError: name 'train_generator' is not defined

## === cell 25
ypreds = model.predict(test_generator, steps=len(test_generator), verbose=1)
ypreds.shape



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4080625050.py in <cell line: 0>()
      1 # --- Bugfix: predict_generator is deprecated; use predict for compatibility
      2 # Keep semantics identical (predict on generator in order).
----> 3 ypreds = model.predict(test_generator, steps=len(test_generator), verbose=1)
      4 ypreds.shape
      5 

NameError: name 'model' is not defined

## === cell 26
n_test = test_generator.n
ypreds = ypreds[:n_test]
ypred = ypreds.argmax(axis=-1)
n_test, ypreds.shape, ypred.shape




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4248009624.py in <cell line: 0>()
      1 # --- Bugfix: ensure predictions count matches number of files (generator may pad last batch)
----> 2 n_test = test_generator.n
      3 ypreds = ypreds[:n_test]
      4 ypred = ypreds.argmax(axis=-1)
      5 n_test, ypreds.shape, ypred.shape

NameError: name 'test_generator' is not defined

## === cell 27
def get_rand_test_img(test_generator=None, labels_test=None, count=5):
    filepaths = test_generator.filepaths
    file_names = test_generator.filenames

    selected_filepaths = []
    selected_file_names = []
    selected_labels = []

    mem = set()
    i = count

    while i > 0:
        rnd = random.randint(0, test_generator.n - 1)
        while rnd in mem:
            rnd = random.randint(0, test_generator.n - 1)
        mem.add(rnd)

        selected_filepaths.append(filepaths[rnd])
        selected_file_names.append(file_names[rnd])

        lbl = label_map_inv[int(labels_test[rnd])]
        lbl = re.sub("_", " ", lbl).title()
        selected_labels.append(lbl)

        i -= 1

    return selected_filepaths, selected_file_names, selected_labels




## === cell 28
try:
    test_img_count = 6
    labels_test = ypred
    selected_filepaths, selected_file_names, selected_labels = get_rand_test_img(
        test_generator=test_generator, labels_test=labels_test, count=test_img_count
    )

    count = test_img_count
    ncols = 3
    nrows = count // ncols if count % ncols == 0 else count // ncols + 1
    figsize = (15, 5 * nrows)

    plt.figure(figsize=figsize)
    for i in range(count):
        plt.subplot(nrows, ncols, i + 1)
        plot_image(os.path.basename(selected_file_names[i]), testing_dir, aspect="auto")
        plt.title(selected_labels[i])
    plt.tight_layout()
    plt.savefig(os.path.join(figure_directory, "test_samples.png"))
    plt.close()
except Exception as e:
    print("Test sample visualization skipped:", repr(e))



## === cell 29
sample_submission = pd.read_csv(os.path.join(input_directory, "sample_submission.csv"))
sample_submission.head()



## === cell 30
submission_ids = sample_submission["id"].values
breed_columns = [c for c in sample_submission.columns if c != "id"]

test_gen_files = [
    os.path.basename(f) for f in test_generator.filenames
]  # ensures just "<id>.jpg"
pred_map = {fname: ypreds[i] for i, fname in enumerate(test_gen_files)}

ypreds_sync = np.zeros((len(submission_ids), len(breed_columns)), dtype=np.float32)
for idx, fid in enumerate(submission_ids):
    fname = fid + ".jpg"
    if fname in pred_map:
        ypreds_sync[idx] = pred_map[fname]
    else:
        ypreds_sync[idx] = 1.0 / len(breed_columns)

model_class_to_index = train_generator.class_indices  # breed -> index in ypreds
reorder_idx = [model_class_to_index[c] for c in breed_columns]
ypreds_sync = ypreds_sync[:, reorder_idx]

row_sums = ypreds_sync.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
ypreds_sync = ypreds_sync / row_sums

sub_df = pd.DataFrame(ypreds_sync, columns=breed_columns)
sub_df.insert(0, "id", submission_ids)

out_path = os.path.join(output_directory, "submission.csv")
sub_df.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
sub_df.head()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3509349471.py in <cell line: 0>()
      6 # Map test filename -> prediction vector
      7 test_gen_files = [
----> 8     os.path.basename(f) for f in test_generator.filenames
      9 ]  # ensures just "<id>.jpg"
     10 pred_map = {fname: ypreds[i] for i, fname in enumerate(test_gen_files)}

NameError: name 'test_generator' is not defined

## === cell 31
assert sub_df.shape == sample_submission.shape, (sub_df.shape, sample_submission.shape)
assert list(sub_df.columns) == list(
    sample_submission.columns
), "Submission columns must match sample_submission"
print("Submission format validated:", sub_df.shape)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1394383477.py in <cell line: 0>()
      1 # Final validation: shape and columns
----> 2 assert sub_df.shape == sample_submission.shape, (sub_df.shape, sample_submission.shape)
      3 assert list(sub_df.columns) == list(
      4     sample_submission.columns
      5 ), "Submission columns must match sample_submission"

NameError: name 'sub_df' is not defined
