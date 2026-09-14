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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.9089861751152074

# 6. Current score

0.06341

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06341) has done: 'The timeout is dominated by (1) extremely heavy per-step CPU image augmentation/decoding with a single-threaded `ImageDataGenerator`, and (2) an over-parameterized `Flatten + Dense(2048,1024,...)` head on 224×224 inputs that makes each training step expensive for 50 epochs. To keep the exact same model/training logic and accuracy semantics, the main speedups are: enable TensorFlow graph/XLA compilation for the existing model, turn on multi-worker prefetching in the Keras generators (pure I/O pipeline speedup), and remove notebook-only debug cells that iterate/visualize batches (they cost extra time but do not affect training/inference outputs). These changes reduce wall-clock time without changing architecture, loss, optimizer, epochs, or data used.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

SEED = 2021
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=5,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
)



## === cell 2
train_ds = img_datagen.flow_from_directory(
    "../input/paddy-disease-classification/train_images",
    subset="training",
    seed=SEED,
    class_mode="categorical",
    batch_size=32,
    target_size=(224, 224),
)
val_ds = img_datagen.flow_from_directory(
    "../input/paddy-disease-classification/train_images",
    subset="validation",
    seed=SEED,
    class_mode="categorical",
    batch_size=32,
    target_size=(224, 224),
)

train_ds.workers = max(2, (os.cpu_count() or 2) // 2)
train_ds.use_multiprocessing = True
train_ds.max_queue_size = 32

val_ds.workers = max(2, (os.cpu_count() or 2) // 2)
val_ds.use_multiprocessing = True
val_ds.max_queue_size = 32

num_classes = train_ds.num_classes  # ensure model output matches generator labels shape



## === cell 3
pass



## === cell 4
pass



## === cell 5
model = tf.keras.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()



## === cell 6
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## === cell 7
lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.8, patience=10, verbose=1
)
save_best = tf.keras.callbacks.ModelCheckpoint(
    "Model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
)



## === cell 8
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[save_best, lr_scheduler],
    workers=train_ds.workers,
    use_multiprocessing=True,
    max_queue_size=train_ds.max_queue_size,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_11/627278305.py in <cell line: 0>()
      1 # --- Speed: explicitly pass generator multiprocessing/prefetch settings into fit (same training loop/epochs/data) ---
----> 2 model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=50,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnimplementedError: Graph execution error:

Detected at node gradient_tape/sequential/max_pooling2d_3/MaxPool/MaxPoolGrad defined at (most recent call last):
<stack traces unavailable>
GPU MaxPool gradient ops do not yet have a deterministic XLA implementation.
	 [[{{node gradient_tape/sequential/max_pooling2d_3/MaxPool/MaxPoolGrad}}]]
	tf2xla conversion failed while converting __inference_run_step_2237[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[StatefulPartitionedCall]] [Op:__inference_train_function_2352]

## === cell 9
model = tf.keras.models.load_model("Model.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/807781106.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("Model.h5", compile=False)
      2 model.compile(
      3     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
      4     loss="categorical_crossentropy",
      5     metrics=["accuracy"],

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode, **kwargs)
    260 
    261     # Legacy case.
--> 262     return legacy_sm_saving_lib.load_model(
    263         filepath, custom_objects=custom_objects, compile=compile, **kwargs
    264     )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/legacy/save.py in load_model(filepath, custom_objects, compile, options)
    231                     if isinstance(filepath_str, str):
    232                         if not tf.io.gfile.exists(filepath_str):
--> 233                             raise IOError(
    234                                 f"No file or directory found at {filepath_str}"
    235                             )

OSError: No file or directory found at Model.h5

## === cell 10
model.evaluate(
    val_ds,
    workers=val_ds.workers,
    use_multiprocessing=True,
    max_queue_size=val_ds.max_queue_size,
)



## === cell 11
test_path = "../input/paddy-disease-classification/test_images"
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    batch_size=32,
    classes=["."],
    shuffle=False,
)

test_gen.workers = max(2, (os.cpu_count() or 2) // 2)
test_gen.use_multiprocessing = True
test_gen.max_queue_size = 32



## === cell 12
predict = model.predict(
    test_gen,
    verbose=1,
    workers=test_gen.workers,
    use_multiprocessing=True,
    max_queue_size=test_gen.max_queue_size,
)



## === cell 13
predicted_class_indices = np.argmax(predict, axis=1)
print(set(predicted_class_indices))

inv_map = {v: k for k, v in train_ds.class_indices.items()}
predictions = [inv_map[k] for k in predicted_class_indices]



## === cell 14
filenames = test_gen.filenames
image_ids = [os.path.basename(f) for f in filenames]

results = pd.DataFrame({"image_id": image_ids, "label": predictions})
results.to_csv("submission.csv", index=False)
results



## === cell 15
results["label"].value_counts()



## === cell 16
results
