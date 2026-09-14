# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.08252

# 6. Current score

0.23733

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24171) has done: 'Diagnosis: Cell 8 crashes during `import tensorflow as tf` with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between TensorFlow and the installed `protobuf==6.33.0`. TensorFlow 2.18 expects an older protobuf API where `MessageFactory.GetPrototype` exists, so the import fails before any model code runs. Since we cannot change the environment package versions, the minimal in-notebook fix is to force protobuf to use the pure-Python implementation, which avoids the missing C++ API path. This must be done before importing TensorFlow.

Patch summary: In cell 8 only, set the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and keep it deterministic) before importing TensorFlow, then proceed with the existing TensorFlow/Keras imports and session clear exactly as before.

Updated cells: (cell 8 only)

Compatibility notes for cell k+1: `tf` and `layers` are still defined with the same names and types, and `tf.keras.backend.clear_session()` still runs, so cell 9 remains unchanged and execute normally.

Assumptions: The runtime honors `os.environ` settings made before importing TensorFlow within the same process, and using the Python protobuf implementation is acceptable for this notebook (performance impact is negligible for this small dataset/model).'
- What this solution (achieved 0.23945) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 8 because the installed `protobuf==6.33.0` is incompatible with the installed `tensorflow==2.18.0` in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow’s protobuf initialization. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not resolve this incompatibility for protobuf 6.x. The minimal in-notebook fix is to force TensorFlow to use the pure-Python protobuf runtime and specifically use the pure-Python implementation version 2 before importing TensorFlow; this avoids the failing C++ API path that expects `GetPrototype`.

Patch summary: In cell 8 only, set the protobuf environment variables *before* importing TensorFlow: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`. Keep the rest of the imports and `tf.keras.backend.clear_session()` unchanged to preserve the notebook’s core logic and downstream API usage.

Updated cells / Compatibility notes for cell k+1 / Assumptions: Cell 9 expects `tf` and `layers` to be defined; this patch keeps those names and TensorFlow/Keras APIs intact. Assumption: TensorFlow 2.18 can run with the Python protobuf implementation in this environment (slower but functional), and no later cells require protobuf C++ acceleration.'
- What this solution (achieved 0.24268) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 8 due to an incompatibility between TensorFlow 2.18.0 and the installed `protobuf==6.33.0`. The workaround in the cell (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) triggers a code path where `MessageFactory.GetPrototype` is expected but no longer exists in protobuf 6, causing the `AttributeError`. The minimal fix is to remove/avoid forcing the pure-Python protobuf implementation so TensorFlow uses the faster C++ runtime that is compatible in this environment.  

Patch summary: In cell 8, stop setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` and `..._VERSION` to `"python"`/`"2"`, and instead ensure the default C++ implementation is used by clearing those env vars if they were set. Keep the rest of the TensorFlow/Keras import logic unchanged so cell 9 continues to work.  

Updated cells: Only cell 8 is modified as required.  

Compatibility notes for cell k+1: `tf` and `layers` remain defined exactly as before, so cell 9 can build the Keras model unchanged.  

Assumptions: This runtime has the standard protobuf C++ extension available (typical in Kaggle-like TF images), and the error is solely due to forcing the Python protobuf backend.'
- What this solution (achieved 0.23987) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 8 due to an incompatibility between `tensorflow==2.18.0` and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. TensorFlow 2.18 expects an older protobuf runtime where `MessageFactory.GetPrototype` exists; protobuf 6 removed/changed that API. Since we cannot change installed packages here, the minimal in-notebook fix is to force protobuf to use the pure-Python implementation (which preserves the expected API surface for TensorFlow) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow. This keeps the rest of the model/training logic unchanged.

Patch summary: Modify only cell 8 to (1) set the protobuf env var to the Python implementation before importing TensorFlow and (2) remove the lines that were explicitly unsetting those env vars, which were causing TensorFlow to load the incompatible C++ protobuf backend.

Updated cells: Only cell 8 is updated below.

Compatibility notes for cell k+1: The patch still defines `tf` and `layers` exactly as expected by cell 9, with identical interfaces; no downstream code changes are required.

Assumptions: Environment allows setting `os.environ` before importing TensorFlow in the same cell; TensorFlow has not been imported earlier in the kernel/session (or this cell is re-run after a fresh kernel).'
- What this solution (achieved 0.23733) has done: 'Diagnosis: The crash occurs while importing TensorFlow in cell 8 because the environment’s `protobuf==6.33.0` is incompatible with TensorFlow 2.18’s expected protobuf APIs, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The current attempt to force the pure-Python protobuf implementation does not fix this incompatibility for protobuf v6. The minimal safe fix is to downgrade protobuf to a TensorFlow-compatible version (protobuf 4.x) before importing TensorFlow, then proceed with the same TensorFlow imports and session reset.

Patch summary: In cell 8 only, add a small pre-import step that installs a compatible protobuf version (`protobuf==4.25.3`) at runtime, then restart the protobuf-related environment variables and import TensorFlow as originally intended. No model/training logic is changed; this only resolves the import-time crash.

Updated cells: Only cell 8 is modified below.

Compatibility notes for cell k+1: The patch still defines `tf` and `layers` exactly as cell 9 expects, so the model construction code remains unchanged and run.

Assumptions: The runtime allows `pip` installs (standard in Kaggle/hosted notebook environments), and downgrading protobuf in-process is sufficient for the subsequent TensorFlow import within the same kernel session.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import glob
import cv2
from matplotlib import pyplot as plt
%matplotlib inline


## === cell 2
X_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train/*.png")
y_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train_cleaned/*.png")
X_test_path = glob.glob("/kaggle/input/denoising-dirty-documents/test/*.png")

input_shape = (258, 540, 1)


## === cell 4
def load_images(path):
    image_list = []
    for pth in path:
        img = cv2.imread(pth, 0) # read grayscale image
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = img / 255.
        img = np.expand_dims(img, axis=-1)
        image_list.append(img)
    return image_list


## === cell 5
X_train_all = load_images(X_train_path)
y_train_all = load_images(y_train_path)
X_test = load_images(X_test_path)

X_train_all = np.array(X_train_all)
y_train_all = np.array(y_train_all)
X_test = np.array(X_test)

print(X_train_all.shape)
print(y_train_all.shape)
print(X_test.shape)


## === cell 7

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


## === cell 8
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.


## === cell 9
model = tf.keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(layers.MaxPooling2D(2, padding='same'))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'))


model.summary()


## === cell 10
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])


## === cell 11
num_epochs = 100
batch_size = 8
history = model.fit(X_train, y_train, epochs=num_epochs,
                    batch_size=batch_size, 
                    verbose=1,
                    validation_data=(X_val, y_val))


## === cell 12
score = model.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904


## === cell 13
final_predictions = model.predict(X_test)


## === cell 14
preds_0 = final_predictions[10] * 255.0
preds_0 = preds_0.reshape(258, 540)
x_test_0 = X_test[10] * 255.0
x_test_0 = x_test_0.reshape(258, 540)
plt.imshow(x_test_0, cmap='gray')


## === cell 15
plt.imshow(preds_0, cmap='gray')


## === cell 16
final_predictions = final_predictions.reshape(-1, 258, 540)

ids = []
vals = []
for i, f in enumerate(X_test_path):
    file = os.path.basename(f)
    imgid = int(file[:-4])
    test_img = cv2.imread(f, 0)
    img_shape = test_img.shape
    print('processing: {}'.format(imgid))
    print(img_shape)
    preds_reshaped = cv2.resize(final_predictions[i], (img_shape[1], img_shape[0]))
    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
            vals.append(preds_reshaped[r, c])

print('Writing to csv file')
pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)
