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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

4.2321

# 6. Current score

1.98246

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.98214) has done: 'I fix the import/runtime issues caused by Keras 3 (the old `keras.preprocessing.image` and `keras.wrappers.scikit_learn` paths break) by switching to `tf_keras` utilities that exist in this environment. Then I make the label→class mapping deterministic and aligned to the sample submission’s column order, so the predicted probabilities land in the correct breed columns (crucial for log loss). I also remove the failing plotting cell dependency and ensure `x_val`/`train_test_split` are defined, and finally write `submission.csv` with exactly the required header and row order. These are correctness fixes (and should improve score from “no submission” to a valid baseline) without changing the model architecture/training semantics.'
- What this solution (achieved 1.98246) has done: 'I fix the runtime crash occurring at import-time by forcing the TensorFlow backend before importing `tf_keras`, which avoids the protobuf `MessageFactory.GetPrototype` failure seen in some Kaggle images. I also make the image-loading loop keep labels aligned to the actually-loaded images (so missing files don’t silently desynchronize `labels` vs `train`), which is a correctness fix that can significantly affect log loss without changing the model/training approach. Finally, I keep the class order exactly aligned to `sample_submission.csv` columns and ensure the written `submission.csv` has the required header/column order and row count. These changes are minimal and preserve the same architecture/training semantics while stabilizing execution and improving correctness.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tf_keras as keras
from tf_keras.utils import load_img, img_to_array

from sklearn.model_selection import train_test_split

np.random.seed(7)
try:
    keras.utils.set_random_seed(7)
except Exception:
    pass

print("KERAS_BACKEND =", os.environ.get("KERAS_BACKEND"))
print("tf_keras version:", getattr(keras, "__version__", "unknown"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels_df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
sample = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")

breed_cols = [c for c in sample.columns if c != "id"]
classes = {ix: class_name for ix, class_name in enumerate(breed_cols)}
reverse_classes = {class_name: ix for ix, class_name in classes.items()}

missing_breeds_in_sample = sorted(set(labels_df["breed"].unique()) - set(breed_cols))
if missing_breeds_in_sample:
    raise ValueError(
        f"Some train breeds are missing from sample submission columns: {missing_breeds_in_sample[:5]}..."
    )

print("num_classes =", len(breed_cols))
print("num_train_labels =", len(labels_df), "num_test =", len(sample))



## === cell 2
direcory = "/kaggle/input/dog-breed-identification/train"
print("no of images in train dataset (labels rows): {}".format(len(labels_df)))
print("no of images in test dataset (sample rows): {}".format(len(sample)))



## === cell 3
t = time.time()

train = []
train_labels = []
missing_train = 0

for img_id, breed in zip(labels_df.id.values, labels_df.breed.values):
    fp = os.path.join(direcory, img_id + ".jpg")
    if not os.path.exists(fp):
        missing_train += 1
        continue
    img = load_img(fp, target_size=(144, 144), color_mode="rgb")
    img = img_to_array(img)
    train.append(img)
    train_labels.append(breed)

train = np.array(train, dtype=np.float32) / 255.0
train_labels = np.array(train_labels)

print(
    f"runtime in seconds: {time.time() - t:.2f}, missing_train_files={missing_train}, "
    f"train_shape={train.shape}, train_labels_shape={train_labels.shape}"
)

if train.shape[0] == 0:
    raise RuntimeError(
        "No training images were loaded. Check the train directory path."
    )



## === cell 4
t = time.time()
names = sample["id"].values[:]

test = []
test_ids_loaded = []
missing_test = 0
test_dir = "/kaggle/input/dog-breed-identification/test"
for name in names:
    fp = os.path.join(test_dir, name + ".jpg")
    if not os.path.exists(fp):
        missing_test += 1
        continue
    img = load_img(fp, target_size=(144, 144), color_mode="rgb")
    img = img_to_array(img)
    test.append(img)
    test_ids_loaded.append(name)

test = np.array(test, dtype=np.float32) / 255.0
test_ids_loaded = np.array(test_ids_loaded)

print(
    f"runtime in seconds: {time.time() - t:.2f}, missing_test_files={missing_test}, "
    f"test_shape={test.shape}, test_ids_loaded={len(test_ids_loaded)}"
)

if test.shape[0] == 0:
    raise RuntimeError("No test images were loaded. Check the test directory path.")



## === cell 5
if train.shape[0] > 0:
    try:
        plt.figure(figsize=(20, 10))
        n_show = min(32, train.shape[0])
        for ix in range(n_show):
            plt.subplot(4, 8, ix + 1)
            plt.imshow(train[ix])
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(train_labels[ix])
        plt.tight_layout()
    except Exception as e:
        print("Plotting skipped due to:", repr(e))



## === cell 6
y_labels = np.array([reverse_classes[b] for b in train_labels], dtype=np.int32)

x_train, y_train = (np.array(train), y_labels)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=7, shuffle=True, stratify=y_train
)

del train, y_labels, train_labels

print("x_train:", x_train.shape, "x_val:", x_val.shape, "num_classes:", len(classes))




## === cell 7
def create_model():
    base_model = keras.applications.InceptionV3(
        input_shape=(144, 144, 3), weights="imagenet", include_top=False, pooling="avg"
    )
    base_model.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dense(4096, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(len(classes), activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )
    return model




## === cell 8
model = create_model()
model.summary()



## === cell 9
model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val), verbose=1)



## === cell 10
prediction = model.predict(test, verbose=1)

pred_df = pd.DataFrame(prediction, columns=breed_cols)

sub = sample[["id"]].copy()
if len(test_ids_loaded) != len(sub):
    full_pred = pd.DataFrame(
        np.full((len(sub), len(breed_cols)), 1.0 / len(breed_cols), dtype=np.float32),
        columns=breed_cols,
    )
    loaded_index = pd.Index(test_ids_loaded, name="id")
    full_pred.loc[sub["id"].isin(test_ids_loaded), :] = (
        pred_df.set_index(loaded_index)
        .loc[sub.loc[sub["id"].isin(test_ids_loaded), "id"].values]
        .values
    )
    pred_df = full_pred

submission = pd.concat([sub, pred_df], axis=1)
submission = submission[["id"] + breed_cols]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
