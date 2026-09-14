# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
sklearn-pandas==2.2.0
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

# 5. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
import tensorflow as tf



## === cell 2
train_data = pd.read_csv("../input/train.csv")



## === cell 3
print("Train shape:", train_data.shape)



## === cell 4
print(train_data.head())



## === cell 5
print("Unique labels:", train_data.has_cactus.unique())



## === cell 6
train_data.has_cactus.hist()
plt.show()



## === cell 7
print(train_data.has_cactus.value_counts())



## === cell 8
train_data.has_cactus.plot(kind="bar")
plt.show()



## === cell 9
positive_examples = train_data[train_data.has_cactus == 1]
negative_examples = train_data[train_data.has_cactus == 0]



## === cell 10
img = mpimg.imread("../input/train/train/" + positive_examples.id.tolist()[2])
plt.imshow(img)
plt.title("Positive example")
plt.show()



## === cell 11
img = mpimg.imread("../input/train/train/" + negative_examples.id.tolist()[2])
plt.imshow(img)
plt.title("Negative example")
plt.show()



## === cell 12
model = keras.models.Sequential(
    [
        keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (5, 5), activation="relu"),
        keras.layers.Conv2D(64, (5, 5), activation="relu"),
        keras.layers.Conv2D(64, (5, 5), activation="relu"),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 13
model.summary()



## === cell 14
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)




## === cell 15
def image_generator(batch_size=64, train=True):
    while True:
        if train:
            indexes = np.arange(train_data[:15000].shape[0])
        else:
            indexes = np.arange(train_data[15000:].shape[0])
        np.random.shuffle(indexes)
        N = len(indexes) // batch_size
        for i in range(N):
            batch_idx = indexes[i * batch_size : (i + 1) * batch_size]
            batch_input = []
            batch_output = []
            for idx in batch_idx:
                img_path = "../input/train/train/" + train_data.id.iloc[idx]
                img = mpimg.imread(img_path).astype("float32") / 255.0
                batch_input.append(img)
                batch_output.append(train_data.has_cactus.iloc[idx])
            batch_input = np.stack(batch_input)
            batch_output = np.array(batch_output).reshape(-1, 1)
            yield batch_input, batch_output




## === cell 16
steps_per_epoch = int(train_data[:15000].shape[0] / 64)
validation_steps = int(train_data[15000:].shape[0] / 64)

model.fit(
    image_generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2),
    ],
    validation_data=image_generator(train=False),
    validation_steps=validation_steps,
)



## === cell 17
val_loss, val_acc, val_auc = model.evaluate(
    image_generator(train=False), steps=validation_steps
)
print(f"Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}, AUC: {val_auc:.4f}")



## === cell 18
test_dir = "../input/test/test/"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
print(f"Number of test images: {len(test_files)}")



## === cell 19
batch = 64
preds = []
for i in range(0, len(test_files), batch):
    batch_files = test_files[i : i + batch]
    imgs = []
    for f in batch_files:
        img = mpimg.imread(os.path.join(test_dir, f)).astype("float32") / 255.0
        imgs.append(img)
    imgs_arr = np.stack(imgs)
    batch_pred = model.predict(imgs_arr, verbose=0)
    preds.extend(batch_pred.squeeze().tolist())

preds = np.array(preds)
print("Predictions shape:", preds.shape)



## === cell 20
sub_file = pd.DataFrame({"id": test_files, "has_cactus": preds})
print(sub_file.head())



## === cell 21
sub_file.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv")
