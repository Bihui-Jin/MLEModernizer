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
tf_keras==2.18.0
tqdm==4.67.1

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

32.44366

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78762) has done: 'I fix the Keras/protobuf import/runtime error by using `tf.keras` consistently (avoiding mixed `keras`/`tensorflow.keras` imports that break in this environment). I update deprecated/removed imports (`keras.layers.convolutional`) and restore `ImageDataGenerator` availability, keeping your CNN architecture and training loop intact. I also fix the training/validation wiring bugs (validation was accidentally using the training set) and ensure preprocessing is consistent for both train and test (rescale), which should also improve log-loss versus the broken pipeline. Finally, I make sure the model checkpoint uses the required `.keras` suffix and that a valid submission CSV is written with columns exactly matching `sample_submission.csv`.'
- What this solution (achieved 4.78689) has done: 'The timeout is dominated by two things: (1) Python-side image loading into lists followed by large conversions/copies, and (2) `ImageDataGenerator(zca_whitening=True)` which forces an expensive ZCA computation (`datagen.fit`) over the whole training set and then applies it per-batch. To preserve the exact model/training logic while making it fast, I (a) switch image ingestion to a preallocated NumPy array (same pixels, same resize, same dtype) to remove Python list growth and extra copies, and (b) cache the ZCA statistics to disk and reuse them on subsequent runs so you don’t pay the ZCA `fit` cost every time (training behavior remains identical once the cache exists). I also add `workers`/`use_multiprocessing` to `model.fit` so the generator preprocessing runs in parallel without changing the data semantics, and I remove display-only `head()` calls that can cost time in notebook environments.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

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
def gen_graph(history, title):
    plt.plot(history.history.get("categorical_accuracy", []))
    plt.plot(history.history.get("val_categorical_accuracy", []))
    plt.title("Accuracy " + title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




## === cell 2
df_train = pd.read_csv("../input/dog-breed-identification/labels.csv")
df_test = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"




## === cell 3
pass




## === cell 4
pass




## === cell 5
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)




## === cell 6
one_hot_labels = np.asarray(one_hot).astype(np.float32)




## === cell 7
im_resize = 64  # image size
num_class = 120  # number of classes




## === cell 8
n_train = len(df_train)
n_test = len(df_test)

X_all_train = np.empty((n_train, im_resize, im_resize, 3), dtype=np.float32)
Y_all_train = one_hot_labels  # already aligned with df_train order

train_ids = df_train["id"].values
for i, f in enumerate(tqdm(train_ids, total=n_train)):
    img = cv2.imread(jpg_train.format(f), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read train image: {jpg_train.format(f)}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (im_resize, im_resize), interpolation=cv2.INTER_AREA)
    X_all_train[i] = img_resized




## === cell 9
X_test_raw = np.empty((n_test, im_resize, im_resize, 3), dtype=np.float32)

test_ids = df_test["id"].values
for i, f in enumerate(tqdm(test_ids, total=n_test)):
    img = cv2.imread(jpg_test.format(f), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read test image: {jpg_test.format(f)}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (im_resize, im_resize), interpolation=cv2.INTER_AREA)
    X_test_raw[i] = img_resized




## === cell 10
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_all_train,
    Y_all_train,
    shuffle=True,
    test_size=0.1,
    random_state=42,
    stratify=np.argmax(Y_all_train, axis=1),
)




## === cell 11
del X_all_train, Y_all_train, df_train




## === cell 12
datagen = ImageDataGenerator(
    rotation_range=15, rescale=1.0 / 255.0, horizontal_flip=True, zca_whitening=True
)

zca_cache_path = "zca_cache_im64.npy"
if os.path.exists(zca_cache_path):
    cache = np.load(zca_cache_path, allow_pickle=True).item()
    datagen.zca_mean = cache["zca_mean"]
    datagen.zca_whitening_matrix = cache["zca_whitening_matrix"]
else:
    datagen.fit(X_train)
    cache = {
        "zca_mean": getattr(datagen, "zca_mean", None),
        "zca_whitening_matrix": getattr(datagen, "zca_whitening_matrix", None),
    }
    if cache["zca_mean"] is None or cache["zca_whitening_matrix"] is None:
        raise RuntimeError(
            "ZCA whitening attributes were not created after datagen.fit; "
            "cannot proceed with zca_whitening=True."
        )
    np.save(zca_cache_path, cache, allow_pickle=True)


def _apply_rescale_zca_in_chunks(
    x_uint8ish_float32, gen: ImageDataGenerator, chunk=512
):
    x = x_uint8ish_float32
    out = np.empty_like(x, dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        out[s:e] = gen.standardize(x[s:e].copy())
    return out


X_valid_scaled = _apply_rescale_zca_in_chunks(X_valid, datagen, chunk=512)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1279291711.py in <cell line: 0>()
     21     }
     22     if cache["zca_mean"] is None or cache["zca_whitening_matrix"] is None:
---> 23         raise RuntimeError(
     24             "ZCA whitening attributes were not created after datagen.fit; "
     25             "cannot proceed with zca_whitening=True."

RuntimeError: ZCA whitening attributes were not created after datagen.fit; cannot proceed with zca_whitening=True.

## === cell 13
pass




## === cell 14
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(im_resize, im_resize, 3),
        activation="relu",
    )
)
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(num_class, activation="softmax"))




## === cell 15
print(model.summary())




## === cell 16
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_crossentropy, categorical_accuracy],
    jit_compile=True,
)




## === cell 17
batch_size = 256
train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)




## === cell 18
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    min_delta=0,
    patience=5,
    restore_best_weights=False,
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)




## === cell 19
Epochs = 100
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

history_rmsprop = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_valid_scaled, Y_valid),
    verbose=1,
)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680566346.py in <cell line: 0>()
      7     epochs=Epochs,
      8     steps_per_epoch=steps_per_epoch,
----> 9     validation_data=(X_valid_scaled, Y_valid),
     10     verbose=1,
     11 )

NameError: name 'X_valid_scaled' is not defined

## === cell 20
gen_graph(history_rmsprop, "график точности")




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1434631678.py in <cell line: 0>()
----> 1 gen_graph(history_rmsprop, "график точности")
      2 
      3 

NameError: name 'history_rmsprop' is not defined

## === cell 21
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
else:
    print(f"Warning: checkpoint {checkpoint_path} not found; using last-epoch model.")




## === cell 22
X_test = _apply_rescale_zca_in_chunks(X_test_raw, datagen, chunk=512)
del X_test_raw
preds = model.predict(X_test, verbose=1, batch_size=256)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1564359877.py in <cell line: 0>()
      2 # instead of only rescale (original code) and leaving ZCA unused at inference.
      3 # This preserves training/inference feature semantics with zca_whitening=True and speeds predict.
----> 4 X_test = _apply_rescale_zca_in_chunks(X_test_raw, datagen, chunk=512)
      5 del X_test_raw
      6 preds = model.predict(X_test, verbose=1, batch_size=256)

NameError: name '_apply_rescale_zca_in_chunks' is not defined

## === cell 23
breed_cols = [c for c in df_test.columns if c != "id"]
if len(breed_cols) != preds.shape[1]:
    raise ValueError(
        f"Prediction shape {preds.shape} does not match sample submission breed columns ({len(breed_cols)})."
    )

sub = pd.DataFrame(preds, columns=breed_cols)
sub.insert(0, "id", df_test["id"].values)

sub = sub.reindex(columns=list(df_test.columns))

if sub.isnull().values.any():
    prob_cols = [c for c in sub.columns if c != "id"]
    sub[prob_cols] = sub[prob_cols].fillna(1.0 / len(prob_cols))

out_path = "output_rmsprop_aug.csv"
sub.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")
print(sub.shape)
print(sub.head(2))

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1192057003.py in <cell line: 0>()
      1 breed_cols = [c for c in df_test.columns if c != "id"]
----> 2 if len(breed_cols) != preds.shape[1]:
      3     raise ValueError(
      4         f"Prediction shape {preds.shape} does not match sample submission breed columns ({len(breed_cols)})."
      5     )

NameError: name 'preds' is not defined
