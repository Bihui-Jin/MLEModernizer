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

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = "../input/dog-breed-identification"
train_dir = os.path.join(input_dir, "train")
test_dir = os.path.join(input_dir, "test")
labels_path = os.path.join(input_dir, "labels.csv")

labels = pd.read_csv(labels_path)
classes = np.unique(labels.breed)
classes_num = classes.size



## === cell 2
height, width, channels_num = 512, 512, 3
learning_rate = 0.004


def dense_block(x, neurons, layer_no):
    x = Dense(neurons, kernel_initializer=HeNormal(), name=f"topDense{layer_no}")(x)
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
    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(base.output)
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()



## === cell 3
decoder_path = "/root/decoder_weights.npy"
if os.path.exists(decoder_path):
    decoder_weights = np.load(decoder_path, allow_pickle=True).item()
    for layer_name, layer_w in decoder_weights.items():
        try:
            model.get_layer(layer_name).set_weights(layer_w)
        except ValueError:
            pass



## === cell 4
temp_test_root = "/root/temp_test_dir"
temp_test_sub = os.path.join(temp_test_root, "test")
os.makedirs(temp_test_sub, exist_ok=True)

for fname in os.listdir(test_dir):
    src = os.path.join(test_dir, fname)
    dst = os.path.join(temp_test_sub, fname)
    shutil.copy(src, dst)

test_gen = ImageDataGenerator(rescale=1 / 255)
test_flow = test_gen.flow_from_directory(
    directory=temp_test_root,
    target_size=(height, width),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)  # no labels



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/2994991227.py in <cell line: 0>()
      8     src = os.path.join(test_dir, fname)
      9     dst = os.path.join(temp_test_sub, fname)
---> 10     shutil.copy(src, dst)
     11 
     12 test_gen = ImageDataGenerator(rescale=1 / 255)

/usr/lib/python3.11/shutil.py in copy(src, dst, follow_symlinks)
    429     if os.path.isdir(dst):
    430         dst = os.path.join(dst, os.path.basename(src))
--> 431     copyfile(src, dst, follow_symlinks=follow_symlinks)
    432     copymode(src, dst, follow_symlinks=follow_symlinks)
    433     return dst

/usr/lib/python3.11/shutil.py in copyfile(src, dst, follow_symlinks)
    254         os.symlink(os.readlink(src), dst)
    255     else:
--> 256         with open(src, 'rb') as fsrc:
    257             try:
    258                 with open(dst, 'wb') as fdst:

IsADirectoryError: [Errno 21] Is a directory: '../input/dog-breed-identification/test/test'

## === cell 5
steps = test_flow.n  # total number of test samples
preds = model.predict(test_flow, steps=steps, verbose=1)

ids = [os.path.splitext(os.path.basename(p))[0] for p in test_flow.filenames]
submission = pd.DataFrame(preds, columns=classes)
submission.insert(0, "id", ids)

submission = submission[["id"] + list(classes)]

submission_path = "/root/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1490581265.py in <cell line: 0>()
      1 # Predict probabilities for every test image
----> 2 steps = test_flow.n  # total number of test samples
      3 preds = model.predict(test_flow, steps=steps, verbose=1)
      4 
      5 # Build submission DataFrame

NameError: name 'test_flow' is not defined
