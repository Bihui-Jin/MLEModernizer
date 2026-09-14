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

0.92615

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.92615) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow to stop the `MessageFactory` error, removes the Jupyter‑only `%matplotlib inline` line, slightly enlarges the convolutional auto‑encoder (more filters) and trains a bit longer (200 epochs) to improve the denoising performance and move the RMSE closer to the target, while keeping the original workflow and submission format unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import glob
import cv2
from matplotlib import pyplot as plt





## === cell 2
X_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train/*.png")
y_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train_cleaned/*.png")
X_test_path = glob.glob("/kaggle/input/denoising-dirty-documents/test/*.png")

input_shape = (258, 540, 1)




## === cell 3
def load_images(path):
    """Load, resize, normalize and add channel dimension to a list of image files."""
    image_list = []
    for pth in path:
        img = cv2.imread(pth, 0)  # read as grayscale
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = img / 255.0  # normalize to [0, 1]
        img = np.expand_dims(img, axis=-1)  # add channel axis
        image_list.append(img)
    return image_list




## === cell 4
X_train_all = load_images(X_train_path)
y_train_all = load_images(y_train_path)
X_test = load_images(X_test_path)

X_train_all = np.array(X_train_all)
y_train_all = np.array(y_train_all)
X_test = np.array(X_test)

print("Train images shape:", X_train_all.shape)
print("Cleaned train images shape:", y_train_all.shape)
print("Test images shape:", X_test.shape)




## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.3, random_state=0
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)




## === cell 6
import os as _os

_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # reset any previous graph state




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
model = tf.keras.Sequential()

model.add(
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=input_shape,
    )
)
model.add(layers.MaxPooling2D(2, padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

model.summary()




## === cell 8
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
)




## === cell 9
num_epochs = 200  # train a bit longer for better denoising
batch_size = 8

history = model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1,
    validation_data=(X_val, y_val),
)




## === cell 10
val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
print("Validation MSE (loss):", val_loss)
print("Validation RMSE:", np.sqrt(val_loss))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3812300594.py in <cell line: 0>()
----> 1 val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
      2 print("Validation MSE (loss):", val_loss)
      3 print("Validation RMSE:", np.sqrt(val_loss))
      4 
      5 

TypeError: 'float' object is not subscriptable

## === cell 11
final_predictions = model.predict(X_test)  # shape: (n_test, 258, 540, 1)




## === cell 12
idx = 10
pred_img = final_predictions[idx].squeeze() * 255.0
orig_img = X_test[idx].squeeze() * 255.0

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Noisy Input")
plt.imshow(orig_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Denoised Output")
plt.imshow(pred_img, cmap="gray")
plt.show()




## === cell 13
final_predictions = final_predictions.reshape(-1, 258, 540)  # drop channel dim

ids = []
vals = []

for i, f in enumerate(X_test_path):
    file = os.path.basename(f)
    imgid = int(file[:-4])  # strip .png
    test_img = cv2.imread(f, 0)
    img_h, img_w = test_img.shape
    pred_resized = cv2.resize(final_predictions[i], (img_w, img_h))
    for r in range(img_h):
        for c in range(img_w):
            ids.append(f"{imgid}_{r+1}_{c+1}")
            vals.append(float(pred_resized[r, c]) / 255.0)  # scale back to [0,1]

print("Writing submission.csv")
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
