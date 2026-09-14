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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass



## === cell 1
train_dir = "../input/dog-breed-identification/train"
test_dir = "../input/dog-breed-identification/test"
labels_path = "../input/dog-breed-identification/labels.csv"

labels = pd.read_csv(labels_path)
labels.head(5)



## === cell 2
train_files = [
    os.path.join(train_dir, f)
    for f in sorted(os.listdir(train_dir))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
breed_map = dict(zip(labels["id"], labels["breed"]))
train_breeds = [breed_map.get(i, None) for i in train_ids]

train_df = pd.DataFrame({"filepath": train_files, "breed": train_breeds})
train_df = train_df.dropna().reset_index(drop=True)



## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
print("Number of breeds:", classes_num)



## === cell 4
test_split = 0.1  # not used for filesystem copy any more
valid_split = 0.2  # will be passed to ImageDataGenerator



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
    "validation_split": valid_split,  # enable Keras split
}
img_gen = ImageDataGenerator(**transform_params)  # training (augmented)
img_feed = ImageDataGenerator(rescale=norm_factor)  # validation / test (no aug)




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
    x = Dense(neurons, kernel_initializer=he_normal(), name=f"topDense{layer_no}")(x)
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
epochs = 5  # quick run; increase if needed
batch_size = 32

model = create_model((height, width, channels))
model.compile(
    optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 10
train_gen = img_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="breed",
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    interpolation="nearest",
    seed=42,
)

valid_gen = img_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="breed",
    target_size=(height, width),
    color_mode="rgb",
    classes=list(classes),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False,
    subset="validation",
    interpolation="nearest",
    seed=42,
)



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
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=valid_gen,
    validation_steps=valid_gen.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=0,
)



## === cell 13
test_gen = ImageDataGenerator(rescale=norm_factor)

test_files = [
    os.path.join(test_dir, f)
    for f in sorted(os.listdir(test_dir))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

test_df = pd.DataFrame({"filepath": test_files})

test_flow = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col=None,
    class_mode=None,
    target_size=(height, width),
    batch_size=1,
    shuffle=False,
    interpolation="nearest",
)



## === cell 14
preds = model.predict(test_flow, verbose=0)  # predict all samples automatically

filenames = test_flow.filenames
ids = [os.path.basename(f).split(".")[0] for f in filenames]



## === cell 15
row_sums = preds.sum(axis=1, keepdims=True)
preds = np.divide(preds, np.clip(row_sums, 1e-12, None), where=row_sums != 0)

submission = pd.DataFrame(preds, columns=classes)
submission.insert(0, "id", ids)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
