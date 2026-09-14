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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9969

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will fix the protobuf import error, correct the optimizer argument, replace deprecated fit_generator with model.fit, make the train/test directory paths robust, and remove the cell that deletes all files. These changes resolve the runtime errors and ensure a proper submission.csv is created while keeping the original model logic unchanged.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/3237712591.py", line 1
    I will fix the protobuf import error, correct the optimizer argument, replace deprecated fit_generator with model.fit, make the train/test directory paths robust, and remove the cell that deletes all files. These changes resolve the runtime errors and ensure a proper submission.csv is created while keeping the original model logic unchanged.
                                                                                            ^
SyntaxError: invalid non-printable character U+202F


## === cell 1
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as pimg
import seaborn as sns
import math
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Dropout,
                                     GlobalMaxPooling2D)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
print(sys.version)
print('tensorflow -> ', tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3590638906.py in <cell line: 0>()
----> 1 print(sys.version)
      2 print('tensorflow -> ', tf.__version__)
      3 

NameError: name 'sys' is not defined

## === cell 3
np.random.seed(12)
tf.random.set_seed(12)



## === cell 4
main_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
sub_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')

def resolve_dir(base, folder_name):
    possible = [
        os.path.join(base, folder_name),
        os.path.join(base, 'aerial-cactus-identification', folder_name),
        os.path.join(base, 'input', 'aerial-cactus-identification', folder_name)
    ]
    for p in possible:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Directory {folder_name} not found under {base}")

train_dir = resolve_dir('/kaggle/working', 'train')
test_dir = resolve_dir('/kaggle/working', 'test')



## === cell 5
main_df.head()



## === cell 6
print('shape: ', main_df.shape)
print('===================================')
print(main_df['has_cactus'].value_counts())



## === cell 7
plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Pastel2')

x = ['has cactus', "hasn't cactus"]
y = main_df.groupby('has_cactus').size()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pie(y, labels=x, autopct="%1.1f%%")
plt.show()



## === cell 8
fig, ax = plt.subplots(2, 5, figsize = (12,6))

def img_path(dir_base, filename):
    path = os.path.join(dir_base, filename)
    if os.path.exists(path):
        return path
    alt_path = os.path.join(dir_base, 'train', filename)
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"Image {filename} not found in {dir_base}")

for i, idx in enumerate(main_df[main_df['has_cactus'] == 1]['id'].tail(5)):
    path = img_path(train_dir, idx)
    img = load_img(path)
    ax[0, i].axis('off')
    ax[0, i].set_title('has cactus')
    ax[0, i].imshow(img)

for i, idx in enumerate(main_df[main_df['has_cactus'] == 0]['id'].tail(5)):
    path = img_path(train_dir, idx)
    img = load_img(path)
    ax[1, i].axis('off')
    ax[1, i].set_title("hasn't cactus")
    ax[1, i].imshow(img)



## === cell 9
train_df, val_df = train_test_split(
    main_df,
    test_size=0.25,
    stratify=main_df['has_cactus'],
    shuffle=True,
    random_state=12
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_val = val_df.shape[0]

print(f'total_train: {total_train}, total_val: {total_val}')



## === cell 10
img_width, img_height = 32, 32
target_size = (img_width, img_height)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_df['has_cactus'] = train_df['has_cactus'].astype(str)
val_df['has_cactus'] = val_df['has_cactus'].astype(str)



## === cell 11
batch_size = 32
x_col, y_col = 'id', 'has_cactus'
class_mode = 'binary'

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False
)



## === cell 12
input_shape = (img_width, img_height, 3)
optimizer = optimizers.Adam(learning_rate=1e-3)



## === cell 13
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    GlobalMaxPooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)
model.summary()



## === cell 14
def step_decay(epoch):
    initial_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    return initial_rate * math.pow(drop, math.floor(epoch / epochs_drop))



## === cell 15
lrate = LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, restore_best_weights=True)
callbacks = [lrate, es]



## === cell 16
epochs = 30
history = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=total_train // batch_size,
    validation_data=val_gen,
    validation_steps=total_val // batch_size,
    callbacks=callbacks,
    verbose=2
)



## === cell 17
sns.set_palette('Dark2')
fig, ax = plt.subplots(2, 1, figsize=(8,6))

plot_acc = pd.DataFrame({
    'acc': history.history['acc'],
    'val_acc': history.history['val_acc']
})
plot_loss = pd.DataFrame({
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

plot_acc.plot(ax=ax[0], title='Accuracy')
plot_loss.plot(ax=ax[1], title='Loss')
plt.tight_layout()
plt.show()



## === cell 18
def predict(model, sub_df):
    pred = np.empty((sub_df.shape[0],), dtype=np.float32)
    for n in tqdm(range(sub_df.shape[0]), desc='Predicting'):
        img_path_full = img_path(test_dir, sub_df.id[n])
        image = np.array(Image.open(img_path_full).convert('RGB'))
        image = image.reshape((1, 32, 32, 3)) / 255.0
        pred[n] = model.predict(image, verbose=0)[0][0]
    sub_df['has_cactus'] = pred
    return sub_df



## === cell 19
predictions = predict(model, sub_df)



## === cell 21
predictions.to_csv('submission.csv', header=True, index=False)
```

## --- ERROR in cell 21, traceback:
  File "/tmp/ipykernel_55/2170904989.py", line 2
    ```
    ^
SyntaxError: invalid syntax
