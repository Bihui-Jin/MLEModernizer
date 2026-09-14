# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

BASE_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test", "unknown")

if not os.path.exists(TEST_DIR):
    candidate_dirs = [
        os.path.join(BASE_DIR, "test", "test", "unknown"),
        os.path.join(BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "unknown"),
        os.path.join(
            BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "test", "unknown"
        ),
    ]
    for cd in candidate_dirs:
        if os.path.exists(cd):
            TEST_DIR = cd
            break

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print(
    "TRAIN_DIR subdirs:",
    os.listdir(TRAIN_DIR)[:10] if os.path.exists(TRAIN_DIR) else None,
)
print(
    "TEST_DIR sample:", os.listdir(TEST_DIR)[:10] if os.path.exists(TEST_DIR) else None
)



## === cell 1
from os import listdir

train_data = []
for cls in ["cat", "dog"]:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    label = "1" if cls == "dog" else "0"
    for file in listdir(cls_dir):
        rel_path = f"{cls}/{file}"  # relative to TRAIN_DIR
        train_data.append([rel_path, label])

df_all = pd.DataFrame(train_data, columns=["filename", "class"])
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(len(df_all) * 0.85)
train = df_all.iloc[:split_idx].copy()
test = df_all.iloc[split_idx:].copy()

print("Train size", len(train))
print("Val size", len(test))
for label in ["0", "1"]:
    print("------------")
    print("\tTrain has", len(train[train["class"] == label]), label)
    print("\tVal has", len(test[test["class"] == label]), label)



## === cell 2
import tf_keras as keras
from tf_keras.preprocessing.image import ImageDataGenerator

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32

train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    horizontal_flip=True,
)



## === cell 3
train_generator = train_image_generator.flow_from_dataframe(
    train,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="class",
    seed=42,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

validation_generator = train_image_generator.flow_from_dataframe(
    test,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="class",
    seed=42,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)



## === cell 4
from tf_keras.applications import vgg16

model = vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
    pooling="max",
)



## === cell 5
for layer in model.layers[:-5]:
    layer.trainable = False



## === cell 6
from tf_keras.layers import Dense
from tf_keras.models import Sequential

transfer_model_vgg16 = Sequential()
for layer in model.layers:
    transfer_model_vgg16.add(layer)

transfer_model_vgg16.add(Dense(512, activation="relu"))
transfer_model_vgg16.add(Dense(1, activation="sigmoid"))

transfer_model_vgg16.summary()



## === cell 7
print("Skipping model_to_dot visualization (not available in this environment).")



## === cell 8
from tf_keras import optimizers

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

transfer_model_vgg16.compile(
    optimizer=adam,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 9
steps_per_epoch = max(1, train_generator.n // BATCH_SIZE)
validation_steps = max(1, validation_generator.n // BATCH_SIZE)

vgg16_model_history = transfer_model_vgg16.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=5,
    verbose=1,
)



## === cell 10
PLOT_DEBUG = False

if PLOT_DEBUG:
    from IPython.display import Image, display

    def plot_prediction(image_path, label):
        display(Image(filename=image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT))
        prediction = "dog"
        confidence = float(label)
        if confidence < 0.5:
            prediction = "cat"
            confidence = 1.0 - confidence
        legend = (
            "The image %s above is a %s with a confidence of %.2f%% (p_dog=%.6f)"
            % (
                image_path,
                prediction,
                confidence * 100,
                float(label),
            )
        )
        print(legend)

else:

    def plot_prediction(image_path, label):
        pass




## === cell 11
import cv2


def build_batches(
    df, has_labels=True, limit=500, batch_size=BATCH_SIZE, produce="images"
):
    """
    produce: "images" -> yields (X, y) if has_labels else yields X only
             "paths"  -> yields (paths, y) if has_labels else yields paths only
    Note: For has_labels=False, expects df columns: filename (or id/filename) and uses TEST_DIR.
    """
    n_rows = len(df)
    if limit != -1:
        n_rows = min(n_rows, int(limit))

    has_filename_col = "filename" in df.columns

    Xb = np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    yb = np.empty((batch_size,), dtype=np.float32) if has_labels else None
    paths = [None] * batch_size

    b = 0
    i = 0

    for row in df.itertuples(index=False):
        if i >= n_rows:
            break

        if has_labels:
            yb[b] = float(getattr(row, "class"))
            raw_image_path = os.path.join(TRAIN_DIR, getattr(row, "filename"))
        else:
            if has_filename_col:
                fn = getattr(row, "filename")
                raw_image_path = os.path.join(TEST_DIR, fn)
            else:
                raw_image_path = os.path.join(
                    TEST_DIR, f"{int(getattr(row, 'id'))}.jpg"
                )

        img = cv2.imread(raw_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {raw_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        )

        Xb[b] = img.astype(np.float32) / 255.0
        paths[b] = raw_image_path

        b += 1
        i += 1

        if b == batch_size:
            if produce == "images":
                if has_labels:
                    yield Xb, yb.copy()
                else:
                    yield Xb
            else:
                if has_labels:
                    yield paths.copy(), yb.copy()
                else:
                    yield paths.copy()
            b = 0

    if b > 0:
        if produce == "images":
            if has_labels:
                yield Xb[:b].copy(), yb[:b].copy()
            else:
                yield Xb[:b].copy()
        else:
            if has_labels:
                yield paths[:b].copy(), yb[:b].copy()
            else:
                yield paths[:b].copy()




## === cell 12
RUN_EVAL_DEBUG = False

if RUN_EVAL_DEBUG:
    samples = 64
    eval_steps = 1
    eval_result = transfer_model_vgg16.evaluate(
        build_batches(test, limit=samples, batch_size=BATCH_SIZE),
        steps=eval_steps,
        verbose=1,
    )
    print("Eval:", eval_result)



## === cell 13
RUN_PRED_DEBUG = False

if RUN_PRED_DEBUG:
    some_predictions = transfer_model_vgg16.predict(
        build_batches(test, limit=12, batch_size=1),
        steps=12,
        verbose=1,
    )



## === cell 14
if RUN_PRED_DEBUG:
    idx = 0
    for mini_batch_files in build_batches(
        test, limit=12, batch_size=1, produce="paths", has_labels=False
    ):
        mini_batch_file = mini_batch_files[0]
        predicted_label = float(some_predictions[idx][0])
        idx += 1
        plot_prediction(mini_batch_file, predicted_label)



## === cell 15
test_files = [f for f in listdir(TEST_DIR) if f.lower().endswith(".jpg")]
output = pd.DataFrame({"filename": test_files})
output["id"] = output["filename"].str.replace(".jpg", "", regex=False).astype(int)
output = output.sort_values("id").reset_index(drop=True)

print("Num test images:", len(output))
print(output.head())



## === cell 16
pred_batch_size = 64
pred_steps = int(np.ceil(len(output) / pred_batch_size))

results = transfer_model_vgg16.predict(
    build_batches(output, limit=-1, has_labels=False, batch_size=pred_batch_size),
    steps=pred_steps,
    verbose=1,
)
results = results.reshape(-1)[: len(output)]
print("Pred shape:", results.shape)



## === cell 17
output["label"] = results.astype(float)
print(output[["id", "label"]].head(10))

if PLOT_DEBUG:
    stop = 5
    for _, row in output.head(stop).iterrows():
        path = os.path.join(TEST_DIR, f"{int(row['id'])}.jpg")
        plot_prediction(path, row["label"])

submission = output[["id", "label"]].copy()
submission["id"] = submission["id"].astype(int)
submission["label"] = submission["label"].astype(float)

submission.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", submission.shape)
print(submission.head())
