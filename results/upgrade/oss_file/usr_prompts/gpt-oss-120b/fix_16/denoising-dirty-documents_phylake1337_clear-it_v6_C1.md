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

0.03254

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'The fix updates the import of Keras to use TensorFlow’s bundled version (avoiding the protobuf error), correctly converts loaded PIL images to NumPy arrays in `images_to_array`, and corrects the pixel‑flattening order when building the submission vector so that the predicted values align with the required `id` ordering. These changes resolve the runtime crash and substantially improve the RMSE by ensuring proper data handling and submission formatting.'
- What this solution (achieved 0.22077) has done: 'I set the protobuf environment variable before importing TensorFlow to avoid the import error, and I make the submission‑building step robust by removing the strict length assertion and safely truncating or padding the prediction vector so it always matches the sample‑submission IDs. This ensures the notebook runs end‑to‑end and writes a valid *.csv* file while keeping the core model unchanged.'
- What this solution (achieved 0.22089) has done: 'The fix adds a model checkpoint that saves the weights with the lowest validation loss and reloads them before making predictions. This prevents later over‑fitting from degrading the denoising performance, lowering the RMSE toward the target while keeping the original architecture and training loop unchanged.'
- What this solution (achieved 0.28616) has done: 'The fix adds all missing imports, defines the dataset paths, and ensures the required Keras/TensorFlow objects are available before they are used. This resolves the `NameError` issues, lets the model train and predict, and guarantees that a correctly‑formatted `submission.csv` is written at the end.'
- What this solution (achieved 0.28616) has done: 'To fix the protobuf import error we set the required environment variable **before** importing TensorFlow.  
We also give the model more training data (20 % validation instead of 30 %) and allow a longer early‑stopping patience (20 epochs) so it can converge better, which should lower the RMSE toward the target while keeping the original architecture intact.'
- What this solution (achieved 0.28616) has done: 'We move the protobuf‑environment setting to the very top (before any imports) so TensorFlow loads correctly, tighten the validation split to 10 % (giving the model more data to learn), and give the training loop more room to converge (longer patience and more epochs). These fixes resolve the runtime error and let the auto‑encoder train better, moving the RMSE toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.47627) has done: 'Implemented fixes to eliminate the protobuf import error, shuffle the dataset before creating train/validation splits for better learning, and extended training capacity with a higher epoch limit and adjusted early‑stopping patience. These changes ensure the notebook runs fully, produces a correctly‑formatted CSV submission, and nudges the RMSE toward the target score.'
- What this solution (achieved 0.28616) has done: 'Implemented fixes:
- Added missing imports and defined callbacks before they’re used.
- Moved data splitting to after loading the images and removed the premature split block.
- Guarded the initial cell to avoid NameErrors.
- Created a dedicated training cell that runs after the model is built and before predictions.
- Adjusted cell ordering and variable scopes so X_train, y_train, and the model are defined before they are accessed.
- Ensured the submission vector aligns with the sample submission length and writes a proper CSV.'
- What this solution (achieved 0.28616) has done: 'The fix moves the protobuf environment variable setting before any TensorFlow import to stop the import‑time `AttributeError`, and changes the flattening order when building the submission so that pixel values align with the required `id` ordering (using column‑major order). These adjustments let the notebook run end‑to‑end and produce a correctly‑formatted CSV, while improving the RMSE toward the target score.'
- What this solution (achieved 0.28616) has done: 'The fix moves the protobuf environment variable to the very top (before any TensorFlow import) to avoid the import error, increases the training capacity (more epochs and a longer early‑stopping patience) so the auto‑encoder can converge better, and adds a small safeguard that creates the checkpoint directory if it does not exist. These changes keep the original model architecture unchanged while making the script run end‑to‑end and improving the validation loss, which should move the leaderboard RMSE toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras import callbacks, Input, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

    class DummyModel:
        def predict(self, x, batch_size=None, verbose=None):
            return x

        def load_weights(self, *_, **__):
            pass

    class DummyCallbacks:
        pass

    callbacks = DummyCallbacks()
    Input = lambda **kwargs: None
    Model = lambda **kwargs: DummyModel()
    Conv2D = MaxPooling2D = UpSampling2D = lambda *_, **__: None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
_possible_paths = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/working/denoising-dirty-documents",
    "/kaggle/working",
]
BASE_DIR = next(p for p in _possible_paths if os.path.isdir(p))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TRAIN_CLEANED_DIR = os.path.join(BASE_DIR, "train_cleaned")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_DIR, "sampleSubmission.csv")

first_train_image = os.listdir(TRAIN_DIR)[0]
sample_img = load_img(
    os.path.join(TRAIN_DIR, first_train_image), color_mode="grayscale"
)
img_width, img_height = sample_img.size  # width, height
IMG_SIZE = (img_height, img_width)  # (height, width)


def images_to_array(data_dir, label_dir=None, img_size=IMG_SIZE):
    """
    Load grayscale images from `data_dir` (and optionally matching labels).
    Returns X (and y if label_dir provided) with shape (N, H, W, 1) normalized.
    """
    image_names = sorted(os.listdir(data_dir))
    n = len(image_names)
    X = np.zeros((n, img_size[0], img_size[1]), dtype=np.uint8)

    for i, name in enumerate(tqdm(image_names, desc="Loading images")):
        img_path = os.path.join(data_dir, name)
        img = load_img(img_path, color_mode="grayscale", target_size=img_size)
        arr = np.array(img)
        if arr.ndim == 3:  # (H, W, 1) -> squeeze
            arr = arr.squeeze()
        X[i] = arr

    X = X.reshape(n, img_size[0], img_size[1], 1).astype(np.float32) / 255.0

    if label_dir:
        label_names = sorted(os.listdir(label_dir))
        y = np.zeros((len(label_names), img_size[0], img_size[1]), dtype=np.uint8)
        for i, name in enumerate(tqdm(label_names, desc="Loading labels")):
            lbl_path = os.path.join(label_dir, name)
            lbl = load_img(lbl_path, color_mode="grayscale", target_size=img_size)
            arr = np.array(lbl)
            if arr.ndim == 3:
                arr = arr.squeeze()
            y[i] = arr
        y = (
            y.reshape(len(label_names), img_size[0], img_size[1], 1).astype(np.float32)
            / 255.0
        )

        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]
        print("Output Data Shape :", X.shape)
        print("Output Label Shape:", y.shape)
        return X, y

    print("Output Data Shape :", X.shape)
    return X


X, y = images_to_array(TRAIN_DIR, label_dir=TRAIN_CLEANED_DIR)

val_split = int(0.1 * X.shape[0])
X_val, y_val = X[:val_split], y[:val_split]
X_train, y_train = X[val_split:], y[val_split:]

print("Train data shape :", X_train.shape)
print("Validation data shape :", X_val.shape)




## === cell 2
samples = np.concatenate((X_train[:3], y_train[:3]), axis=0)
f, ax = plt.subplots(2, 3, figsize=(12, 6))
for i, img in enumerate(samples):
    ax[i // 3, i % 3].imshow(img[:, :, 0], cmap="gray")
    ax[i // 3, i % 3].axis("off")
plt.show()




## === cell 3
if TF_AVAILABLE:
    tf.random.set_seed(42)
    np.random.seed(42)

    input_layer = Input(shape=(None, None, 1))

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")

    lr_callback = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=2, verbose=1, min_lr=1e-5
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=200, restore_best_weights=True, verbose=1
    )
    checkpoint_path = "best_model.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    ckpt_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
else:
    print("TensorFlow not available – model will be skipped.")
    model = None  # placeholder




## === cell 4
test_image_names = sorted(os.listdir(TEST_DIR))
X_test = []
orig_shapes = []  # keep (H, W) for each image to aid flattening later
for name in tqdm(test_image_names, desc="Preparing test set"):
    img_path = os.path.join(TEST_DIR, name)
    img = load_img(img_path, color_mode="grayscale")  # no target_size
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr.squeeze()
    h, w = arr.shape
    orig_shapes.append((h, w))
    arr = arr.reshape(h, w, 1).astype(np.float32) / 255.0
    X_test.append(arr)

X_test = np.stack(X_test, axis=0)
print("Test set shape:", X_test.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3043756350.py in <cell line: 0>()
     15     X_test.append(arr)
     16 
---> 17 X_test = np.stack(X_test, axis=0)
     18 print("Test set shape:", X_test.shape)
     19 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    447     shapes = {arr.shape for arr in arrays}
    448     if len(shapes) != 1:
--> 449         raise ValueError('all input arrays must have the same shape')
    450 
    451     result_ndim = arrays[0].ndim + 1

ValueError: all input arrays must have the same shape

## === cell 5
if TF_AVAILABLE and model is not None:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=2000,
        batch_size=16,
        callbacks=[lr_callback, early_stop, ckpt_callback],
        verbose=2,
    )
    model.load_weights(checkpoint_path)
    preds = model.predict(X_test, batch_size=16, verbose=0)
else:
    preds = X_test.copy()

submit_vector = []
for img_pred, (h, w) in zip(preds, orig_shapes):
    flat = img_pred[:h, :w, 0].ravel(order="F")
    submit_vector.extend(flat.tolist())

print("Total prediction values:", len(submit_vector))

sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

if len(submit_vector) != len(sample_df):
    print("Length mismatch detected – adjusting to fit sample submission.")
    min_len = min(len(submit_vector), len(sample_df))
    submit_vector = submit_vector[:min_len]
    sample_df = sample_df.iloc[:min_len]

submission = pd.DataFrame({"id": sample_df["id"], "value": submit_vector})




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/471685142.py in <cell line: 0>()
     11     )
     12     model.load_weights(checkpoint_path)
---> 13     preds = model.predict(X_test, batch_size=16, verbose=0)
     14 else:
     15     # Use the noisy input as a naive prediction

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/data_adapter_utils.py in check_data_cardinality(data)
    113             )
    114             msg += f"'{label}' sizes: {sizes}\n"
--> 115         raise ValueError(msg)
    116 
    117 

ValueError: Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.'x' sizes: 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 258, 258, 258, 258, 258, 258, 258, 258, 420, 420, 258, 420, 420


## === cell 6
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/701193129.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"Submission written to {submission_path}")

NameError: name 'submission' is not defined
