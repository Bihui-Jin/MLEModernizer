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
protobuf==6.33.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.05382

# 6. Current score

0.06769

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.06769) has done: 'The fix updates the imports to avoid a protobuf‑related crash, loads test images at their native resolution (so the number of predicted pixels matches the expected submission size), and builds the submission vector directly from each prediction. This removes the earlier import error and resolves the row‑count mismatch while keeping the original model and training logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau

base_path = "/kaggle/input/denoising-dirty-documents"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")
train_clean_dir = os.path.join(base_path, "train_cleaned")
sample_submission_path = os.path.join(base_path, "sampleSubmission.csv")
working_path = "/kaggle/working"  # where we will write the final CSV



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
first_train_img = os.path.join(train_dir, os.listdir(train_dir)[0])
tmp_img = load_img(first_train_img, color_mode="grayscale")
h, w = tmp_img.size[1], tmp_img.size[0]  # PIL gives (width, height)
print(f"Common image size: height={h}, width={w}")




## === cell 2
def images_to_array(data_dir, label_dir=None, img_size=(h, w)):
    """
    Load grayscale images, resize to img_size, normalize to [0,1],
    and return as NumPy arrays of shape (n, h, w, 1).
    If label_dir is provided, the same is done for the labels and both
    arrays are shuffled together.
    """
    image_names = sorted(os.listdir(data_dir))
    n = len(image_names)
    X_arr = np.zeros((n, img_size[0], img_size[1], 1), dtype=np.float32)

    for i in tqdm(range(n), desc="Loading images"):
        img_path = os.path.join(data_dir, image_names[i])
        img = load_img(img_path, color_mode="grayscale", target_size=img_size)
        arr = img_to_array(img) / 255.0  # (h, w, 1)
        X_arr[i] = arr

    if label_dir:
        label_names = sorted(os.listdir(label_dir))
        n_lbl = len(label_names)
        y_arr = np.zeros((n_lbl, img_size[0], img_size[1], 1), dtype=np.float32)

        for i in tqdm(range(n_lbl), desc="Loading labels"):
            lbl_path = os.path.join(label_dir, label_names[i])
            lbl = load_img(lbl_path, color_mode="grayscale", target_size=img_size)
            arr = img_to_array(lbl) / 255.0
            y_arr[i] = arr

        perm = np.random.permutation(n_lbl)
        X_arr = X_arr[perm]
        y_arr = y_arr[perm]

        print("Output Data Size:", X_arr.shape)
        print("Output Label Size:", y_arr.shape)
        return X_arr, y_arr

    print("Output Data Size:", X_arr.shape)
    return X_arr




## === cell 3
X, y = images_to_array(train_dir, label_dir=train_clean_dir)



## === cell 4
data_size = X.shape[0]
val_split = int(0.15 * data_size)

X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)



## === cell 5
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
f, ax = plt.subplots(2, 3, figsize=(20, 10))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()



## === cell 6
input_layer = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()



## === cell 7
lr_callback = ReduceLROnPlateau(
    monitor="val_loss", patience=4, factor=0.4, min_lr=1e-5, verbose=1
)



## === cell 8
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[lr_callback],
    verbose=2,
)



## === cell 9
val_loss = model.evaluate(X_val, y_val, verbose=0)
print("Validation MSE:", val_loss)
print("Validation RMSE:", np.sqrt(val_loss))



## === cell 10
test_names = sorted(os.listdir(test_dir))
submit_vector = []

for name in tqdm(test_names, desc="Predicting test images"):
    img_path = os.path.join(test_dir, name)
    img = load_img(img_path, color_mode="grayscale")  # keep original size
    arr = img_to_array(img) / 255.0  # shape (h_i, w_i, 1)
    pred = model.predict(arr[np.newaxis, ...], verbose=0)[0, ..., 0]  # (h_i, w_i)
    submit_vector.extend(pred.ravel().tolist())

print("Number of prediction values:", len(submit_vector))



## === cell 11
sample_csv = pd.read_csv(sample_submission_path)
print("Sample submission rows:", sample_csv.shape[0])
print(sample_csv.head())



## === cell 12
id_col = sample_csv["id"]
value_col = pd.Series(submit_vector, name="value")
submission = pd.concat([id_col, value_col], axis=1)

print("Submission preview:")
print(submission.head())



## === cell 13
submission_path = os.path.join(working_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
