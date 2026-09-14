# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd




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
        img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
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
tf_available = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers

    tf_available = True
    print("TensorFlow imported successfully.")
except Exception as e:
    print("TensorFlow import failed (fallback will be used):", e)


def baseline_denoise_batch(images):
    """Apply OpenCV fastNlMeansDenoising to a batch of normalized images."""
    denoised = []
    for img in images:
        img_uint8 = (img.squeeze() * 255).astype(np.uint8)
        out_uint8 = cv2.fastNlMeansDenoising(
            img_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        out = out_uint8.astype(np.float32) / 255.0
        out = np.expand_dims(out, -1)
        denoised.append(out)
    return np.array(denoised)




## === cell 7
if tf_available:
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            96, (3, 3), activation="relu", padding="same", input_shape=input_shape
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))

    model.add(layers.Conv2D(144, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))

    model.add(layers.Conv2D(288, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))

    model.add(layers.Conv2D(288, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(144, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(96, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

    model.add(layers.Cropping2D(cropping=((3, 3), (2, 2))))

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )
    model.summary()
else:
    model = None
    print("Using baseline denoiser; TensorFlow model will not be built.")



## === cell 8
if tf_available:
    num_epochs = 800  # increased epochs for better learning
    batch_size = 8

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr],
    )
else:
    print("Skipping training; fallback denoiser does not require fitting.")



## === cell 9
if tf_available:
    val_loss = model.evaluate(X_val, y_val, verbose=0)  # returns MSE
    print("Validation MSE (loss):", val_loss)
    print("Validation RMSE:", np.sqrt(val_loss))
else:
    val_pred = baseline_denoise_batch(X_val)
    val_mse = np.mean((val_pred - y_val) ** 2)
    print("Baseline Validation MSE:", val_mse)
    print("Baseline Validation RMSE:", np.sqrt(val_mse))



## === cell 10
if tf_available:
    final_predictions = model.predict(X_test)  # (n_test, 258, 540, 1)
else:
    final_predictions = baseline_denoise_batch(X_test)

final_predictions = final_predictions.squeeze(-1)  # (n_test, 258, 540)



## === cell 11
idx = 0
pred_img = (final_predictions[idx] * 255.0).astype(np.uint8)
orig_img = (X_test[idx].squeeze() * 255.0).astype(np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Noisy Input")
plt.imshow(orig_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Denoised Output")
plt.imshow(pred_img, cmap="gray")
plt.show()



## === cell 12

h_target, w_target = input_shape[0], input_shape[1]
num_images = len(X_test_path)
total_pixels = num_images * h_target * w_target

ids_array = np.empty(total_pixels, dtype=object)
vals_array = np.empty(total_pixels, dtype=np.float32)

pos = 0
rows_pattern = np.arange(1, h_target + 1).repeat(w_target)
cols_pattern = np.tile(np.arange(1, w_target + 1), h_target)

for i, f in enumerate(X_test_path):
    imgid = int(os.path.splitext(os.path.basename(f))[0])
    pred = final_predictions[i]

    if pred.shape[0] != h_target or pred.shape[1] != w_target:
        crop_h = min(h_target, pred.shape[0])
        crop_w = min(w_target, pred.shape[1])
        pred_resized = pred[:crop_h, :crop_w]
        if crop_h < h_target or crop_w < w_target:
            padded = np.zeros((h_target, w_target), dtype=pred_resized.dtype)
            padded[:crop_h, :crop_w] = pred_resized
            pred_resized = padded
    else:
        pred_resized = pred

    pred_resized = np.clip(pred_resized, 0.0, 1.0)

    imgid_vec = np.full(h_target * w_target, imgid, dtype=np.int32)
    ids_array[pos : pos + h_target * w_target] = np.char.mod(
        "%d_%d_%d", np.stack([imgid_vec, rows_pattern, cols_pattern], axis=1)
    )
    vals_array[pos : pos + h_target * w_target] = pred_resized.ravel()
    pos += h_target * w_target

print("Writing submission.csv")
submission_path = "submission.csv"
pd.DataFrame({"id": ids_array, "value": vals_array}).to_csv(
    submission_path, index=False
)
print(f"Submission written to {submission_path}")
