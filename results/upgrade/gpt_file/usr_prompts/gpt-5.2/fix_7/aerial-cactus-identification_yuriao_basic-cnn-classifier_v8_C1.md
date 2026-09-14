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

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.5

# 6. Current score

0.99709

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99606) has done: 'I fix the environment-breaking import/protobuf issue by removing the unnecessary standalone `keras` import and using only `tf.keras`, which is compatible with the provided TensorFlow/Keras stack. I also correct deprecated APIs (`lr` → `learning_rate`, `fit_generator` → `fit`) so training runs end-to-end. Then I fix the dataset paths to match the actual Kaggle directory structure and make test-time inference robust by iterating only over `.jpg` files and writing the submission once at the end with the exact `id,has_cactus` schema and correct row count.'
- What this solution (achieved 0.99795) has done: 'I fix the protobuf/TensorFlow import crash by forcing the compatible pure-Python protobuf implementation before TensorFlow loads, which addresses the `MessageFactory.GetPrototype` error without changing your model/training logic. I also keep the Keras usage strictly via `tf.keras` and add a small safety fallback to pin protobuf to the pure-Python backend even if the environment variables aren’t honored initially. Finally, I ensure the submission CSV is always written with the exact `id,has_cactus` columns and the correct row order from `sample_submission.csv`, without changing prediction semantics (so the score should remain essentially the same, and still above your target).'
- What this solution (achieved 0.99745) has done: 'I fix the immediate crash by forcing protobuf’s pure-Python implementation *before* TensorFlow imports, and by adding a small compatibility shim that restores the missing `MessageFactory.GetPrototype` method expected by TF in this environment. This is a runtime-only patch (no model/training logic changes) and should allow the notebook to execute end-to-end. I keep the exact same model, data pipeline, training, and prediction semantics so the score remains essentially unchanged (still far above the 0.5 target) while producing a valid `submission.csv`.'
- What this solution (achieved 0.99865) has done: 'I fix the protobuf/TensorFlow crash by applying the `MessageFactory.GetPrototype` shim correctly at the *module level* (where TF expects it), instead of patching the class only. This is a runtime compatibility fix only and does not change your model, training loop, or prediction logic, so the score should remain essentially the same (still far above the 0.5 target). I also keep the environment variables set before importing TensorFlow to ensure the pure-Python protobuf backend is used. The rest of the pipeline (paths, generators, training, and submission writing) stays unchanged and produce a valid `submission.csv`.'
- What this solution (achieved 0.99829) has done: 'I fix the protobuf/TensorFlow crash by patching the correct object (`google.protobuf.message_factory.MessageFactory.GetPrototype`) before importing TensorFlow, instead of patching the module attribute. This is a runtime-compatibility fix only and won’t change your model, training loop, or prediction semantics (so your score should remain essentially unchanged and still far above the 0.5 target). I also keep the submission generation exactly aligned to `sample_submission.csv` order and ensure `submission.csv` is always written successfully. No model/architecture/training changes are introduced.'
- What this solution (achieved 0.99709) has done: 'I fix the immediate runtime crash by removing the fragile protobuf monkey-patch that’s currently raising `AttributeError` during import, while keeping the environment variables that force the pure-Python protobuf backend (the intended compatibility fix). This change is runtime-only and does not touch your model, training loop, data pipeline, or prediction logic, so it should keep your score essentially the same (still far above the 0.5 target). I also keep the TensorFlow/Keras imports unchanged and ensure the script always reaches submission writing. The rest of the cells remain logically identical and still output a valid `submission.csv` with `id,has_cactus` in the sample-submission order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/aerial-cactus-identification"
TRAINING_DIR = os.path.join(BASE_DIR, "train")
TESTING_DIR = os.path.join(BASE_DIR, "test")
TRAINING_LABEL_DIR = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAINING_DIR), f"Missing TRAINING_DIR: {TRAINING_DIR}"
assert os.path.exists(TESTING_DIR), f"Missing TESTING_DIR: {TESTING_DIR}"
assert os.path.exists(
    TRAINING_LABEL_DIR
), f"Missing TRAINING_LABEL_DIR: {TRAINING_LABEL_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"



## === cell 2
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["acc"],
)



## === cell 3
all_label = pd.read_csv(TRAINING_LABEL_DIR)

rng = np.random.RandomState(SEED)
msk = rng.rand(len(all_label)) < 0.8
training_label = all_label.loc[msk].reset_index(drop=True)
validation_label = all_label.loc[~msk].reset_index(drop=True)

len(training_label), len(validation_label)



## === cell 4
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 5
train_generator = train_datagen.flow_from_dataframe(
    dataframe=training_label,
    directory=TRAINING_DIR,
    batch_size=20,
    x_col="id",
    y_col="has_cactus",
    class_mode="raw",
    target_size=(150, 150),
    shuffle=True,
    seed=SEED,
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_label,
    directory=TRAINING_DIR,
    batch_size=20,
    x_col="id",
    y_col="has_cactus",
    class_mode="raw",
    target_size=(150, 150),
    shuffle=False,
)



## === cell 6
history = model.fit(
    train_generator,
    epochs=10,
    verbose=1,
    validation_data=validation_generator,
)



## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["id"].astype(str).tolist()

preds = []
for fname in test_ids:
    fpath = os.path.join(TESTING_DIR, fname)
    img = image.load_img(fpath, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    p = float(model.predict(x, verbose=0)[0][0])
    preds.append(p)

submission = pd.DataFrame({"id": test_ids, "has_cactus": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

submission.head(), submission.shape, submission_path
