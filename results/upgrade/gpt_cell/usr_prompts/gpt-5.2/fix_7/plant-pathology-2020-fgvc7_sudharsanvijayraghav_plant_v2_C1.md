# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
try:
    import google.protobuf.runtime_version as _pb_runtime_version

    if hasattr(_pb_runtime_version, "ValidateProtobufRuntimeVersion"):
        _pb_runtime_version.ValidateProtobufRuntimeVersion = (
            lambda *args, **kwargs: None
        )
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras import utils
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, InputLayer
import keras
import imageio
from PIL import Image
import shutil

import sklearn as sk
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


## === cell 2
train_file = "../input/plant-pathology-2020-fgvc7/train.csv"
folder =   "../input/plant-pathology-2020-fgvc7/images/"

path = "../input/plant-pathology-2020-fgvc7/images/"
sub_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

test_file = "../input/plant-pathology-2020-fgvc7/test.csv"


## === cell 3
df = pd.read_csv(train_file)


## === cell 4
df_test = pd.read_csv(test_file)


## === cell 5
df.head()


## === cell 6
colnames = df.columns.to_list()
colnames.remove('image_id')
colnames


## === cell 7
df.describe()


## === cell 8
df.loc[:,colnames].sum(axis = 1).value_counts()


## === cell 9
def get_label(row):
    if row['healthy'] : 
        return 'healthy'
    elif row['multiple_diseases']:
        return 'multiple_diseases'
    elif row['rust']:
             return 'rust'
    elif row['scab']:
             return 'scab'


## === cell 10
df['label'] = df.apply(get_label, axis = 1)


## === cell 11
df['file_name'] = df['image_id'].astype(str)+'.jpg'
df_test['file_name'] = df_test['image_id'].astype(str)+'.jpg'


## === cell 12
df.head()


## === cell 13
df_train, df_validate = train_test_split(df, 
                                         test_size = .2, random_state = 42)


## === cell 14
print(f"Training Size : {len(df_train)}")
print(f"Validation Size : {len(df_validate)}")


## === cell 15
im = Image.open('../input/plant-pathology-2020-fgvc7/images/Train_1.jpg')
width, height = im.size
print(width, height)


## === cell 16

BATCH = 6
weidth = int(width / 1.5)
height = int(height/1.5)
train_datagen = ImageDataGenerator(
                rescale = 1.0 / 255,
                horizontal_flip = True,
                fill_mode = 'nearest' )

train_generator = train_datagen.flow_from_dataframe(
        dataframe = df_train,
        directory = path,
        x_col = 'file_name',
        y_col =  'label',    # [('healthy', 'multiple_diseases', 'rust', 'scab')],
        target_size = (height, width),
        batch_size = BATCH,
        class_mode = 'categorical',
        classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
)

validation_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = validation_datagen.flow_from_dataframe(
        dataframe = df_validate,
        directory = path,
        x_col = 'file_name',
        y_col =  'label',    # [('healthy', 'multiple_diseases', 'rust', 'scab')],
        target_size = (height, width),
        batch_size = BATCH,
        class_mode = 'categorical',
        classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
)


## === cell 17
from tensorflow.keras.callbacks import EarlyStopping

base_model = ResNet50(weights='imagenet', include_top=False) 
len(base_model.layers)


## === cell 18
x= base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(64,activation='relu')(x) 
x=Dense(32,activation='relu')(x) 
preds=Dense(4,activation="softmax")(x)


## === cell 19
model = Model(inputs = base_model.input, outputs = preds)


## === cell 20
for x, y in train_generator:
    break
x.shape, y.shape


## === cell 21
plt.imshow(x[0])


## === cell 22
n_epochs = 20
valiation_steps = len(df_validate)
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam',
              metrics=['categorical_accuracy'])


## === cell 23
history = model.fit(train_generator,
                    epochs=n_epochs,
                    steps_per_epoch= len(val_generator), #20 ,
                    validation_data=val_generator,
                    validation_steps = len(val_generator))


## === cell 24
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'Validation')
plt.legend()
plt.show()


## === cell 25
df_test.head()


## === cell 26
submit_datagen = ImageDataGenerator(rescale = 1. / 255)

submit_generator = submit_datagen.flow_from_dataframe(
            dataframe = df_test,
            directory = path,
            x_col = 'file_name',
            y_col =  None,    
            target_size = (height, width),
            batch_size = BATCH,
            class_mode = None,
            classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
)            


y_pred = model.predict(submit_generator, steps = len(df_test))


## === cell 27
submit = pd.concat([df_test, pd.DataFrame(y_pred, columns=colnames) ],  axis = 1)


## === cell 28
submit = submit[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab' ]]


## === cell 29
submit.head()


## === cell 30
submit.to_csv("/kaggle/working/submit.csv", index = False)
