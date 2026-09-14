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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7333452683918757

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fixed the import errors caused by using internal TensorFlow modules, corrected the paths for saving weights and logs so they point to writable local folders, created those folders with `os.makedirs`, set the configuration to actually train a small CNN (`conv1`) for a few epochs, and ensured the submission file is written with the proper column names. These changes resolve the runtime failures and produce a valid `submission.csv` while keeping the original model architecture and training logic.'
- What this solution (achieved 0.0) has done: 'I fixed the TensorFlow import error by removing the direct `import tensorflow as tf` and only using Keras‑specific imports. The Adam optimizer calls were updated to use the correct `learning_rate` argument instead of the deprecated `lr`. To move the validation score toward the target, I switched the model from the small custom CNN (`conv1`) to the pretrained ResNet‑50 architecture, which generally yields higher quadratic weighted kappa scores. All other logic remains unchanged, and the script now writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # ratings 0‑4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
NUM_WORKERS = min(8, os.cpu_count() or 1)

MODEL_NAME = "resnet50"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

train["diagnosis"] = train["diagnosis"].astype(str)

train["filename"] = train["id_code"] + ".png"
test["filename"] = test["id_code"] + ".png"

print("Number of train samples:", train.shape[0])
print("Number of test samples :", test.shape[0])




## === cell 2
def get_image_dir(base_dir: str, subfolder: str) -> str:
    """
    Return the correct directory that contains the images.
    Some Kaggle kernels unzip archives into a nested folder
    (e.g. train_images/train_images). This helper checks for that
    pattern and falls back to the plain folder if it exists.
    """
    candidate = os.path.join(base_dir, subfolder)
    nested = os.path.join(candidate, subfolder)  # e.g. …/train_images/train_images
    if os.path.isdir(nested):
        return nested
    elif os.path.isdir(candidate):
        return candidate
    else:
        raise FileNotFoundError(f"Cannot locate image directory for {subfolder}")


BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_IMG_DIR = get_image_dir(BASE_INPUT, "train_images")
TEST_IMG_DIR = get_image_dir(BASE_INPUT, "test_images")




## === cell 3
def get_model(name, input_shape, n_classes):
    """Build a simple transfer‑learning model."""
    if name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.5)(x)
        output = Dense(n_classes, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=output)

        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-10:]:
            layer.trainable = True
        return model
    else:
        raise ValueError(f"Unsupported model name: {name}")




## === cell 4
if TF_AVAILABLE:
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=TRAIN_IMG_DIR,
        x_col="filename",
        y_col="diagnosis",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="training",
        seed=SEED,
        shuffle=True,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=train,
        directory=TRAIN_IMG_DIR,
        x_col="filename",
        y_col="diagnosis",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="validation",
        seed=SEED,
        shuffle=False,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=TEST_IMG_DIR,
        x_col="filename",
        batch_size=TEST_BATCH_SIZE,
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        seed=SEED,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )
else:
    train_gen = val_gen = test_gen = None



## === cell 5
if TF_AVAILABLE:
    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS = 5
    model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.n // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    predictions = np.argmax(preds, axis=1).astype(int)
else:
    most_common = int(train["diagnosis"].astype(int).mode()[0])
    predictions = np.full(shape=len(test), fill_value=most_common, dtype=int)

ids = test["id_code"].tolist()
submission = pd.DataFrame({"id_code": ids, "diagnosis": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission.head(10)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1863692514.py in <cell line: 0>()
      8 
      9     EPOCHS = 5
---> 10     model.fit(
     11         train_gen,
     12         steps_per_epoch=train_gen.n // BATCH_SIZE,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0
