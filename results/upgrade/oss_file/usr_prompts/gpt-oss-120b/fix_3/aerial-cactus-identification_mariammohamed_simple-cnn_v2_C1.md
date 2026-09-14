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

# 5. Target score

0.6296

# 6. Current score

0.99876

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99876) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow, corrects the image folder paths (removing the extra “train/train” and “test/test” directories), and updates the generator and visualization cells to use the proper directories. These changes stop the import error, allow images to be loaded correctly for training and inference, and ensure the generated submission file contains exactly the required number of rows.'

# 9. Code solution

## === cell 0
import os, glob

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

print("input dirs:", os.listdir("../input"))
BASE_PATH = "../input/aerial-cactus-identification"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
print("train shape:", train_data.shape)




## === cell 2
train_data.head()




## === cell 3
print("Unique labels:", train_data.has_cactus.unique())




## === cell 4
train_data.has_cactus.value_counts().plot.bar()




## === cell 5
positive_examples = train_data[train_data.has_cactus == 1]
negative_examples = train_data[train_data.has_cactus == 0]




## === cell 6
img = mpimg.imread(os.path.join(BASE_PATH, "train", positive_examples.id.iloc[2]))
plt.imshow(img)




## === cell 7
img = mpimg.imread(os.path.join(BASE_PATH, "train", negative_examples.id.iloc[2]))
plt.imshow(img)




## === cell 8
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()




## === cell 9
opt = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])




## === cell 10
print("Total training samples:", train_data.shape[0])




## === cell 11
def image_generator(batch_size=64, train=True):
    """
    Yields batches of (images, labels). Images are read as float32
    and normalized to roughly [-1, 1].
    """
    while True:
        idxs = np.arange(train_data.shape[0])
        np.random.shuffle(idxs)
        steps = len(idxs) // batch_size
        for i in range(steps):
            batch_idx = idxs[i * batch_size : (i + 1) * batch_size]
            batch_x = []
            batch_y = []
            for ind in batch_idx:
                img_path = os.path.join(BASE_PATH, "train", train_data.id.iloc[ind])
                img = mpimg.imread(img_path).astype(np.float32)
                if img.shape[-1] == 4:  # handle possible alpha channel
                    img = img[..., :3]
                img = (img - 127.0) / 127.0
                batch_x.append(img)
                batch_y.append(train_data.has_cactus.iloc[ind])
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y).reshape(-1, 1)
            yield batch_x, batch_y




## === cell 12
steps_per_epoch = train_data.shape[0] // 64
model.fit(image_generator(), steps_per_epoch=steps_per_epoch, epochs=5, verbose=2)




## === cell 13
eval_steps = train_data.shape[0] // 64
model.evaluate(image_generator(train=False), steps=eval_steps)




## === cell 14
test_dir = os.path.join(BASE_PATH, "test")
all_test_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print("Number of test images:", len(all_test_files))




## === cell 15
batch = 64
preds = []
for i in range(0, len(all_test_files), batch):
    batch_files = all_test_files[i : i + batch]
    imgs = []
    for fname in batch_files:
        img_path = os.path.join(test_dir, fname)
        img = mpimg.imread(img_path).astype(np.float32)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = (img - 127.0) / 127.0
        imgs.append(img)
    imgs_np = np.array(imgs)
    batch_pred = model.predict(imgs_np, verbose=0)
    preds.extend(batch_pred.squeeze().tolist())
preds = np.array(preds)




## === cell 16
submission = pd.DataFrame({"id": all_test_files, "has_cactus": preds})
print("Submission shape:", submission.shape)




## === cell 17
submission.to_csv("sample_submission.csv", index=False)
print("Saved sample_submission.csv")
