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

No external packages required in the script and installed.

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

0.9685

# 6. Current score

0.42304

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.42304) has done: 'The script is fixed to correctly load images (skipping sub‑folders), use TensorFlow 2‑style APIs (seed, Adam optimizer, eager execution), reshape and normalize the data, train the original ConvNet model, and generate probability predictions for the test set. The final cell writes a proper `test_submission.csv` with matching row counts, fixing all earlier runtime errors and producing a valid submission file.'

# 9. Code solution

## === cell 0
import os, time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "../input/train/train"
test_dir = "../input/test/test"
train_csv = "../input/train.csv"

df = pd.read_csv(train_csv)



## === cell 2
train_images = []
for fname in os.listdir(train_dir):
    fpath = os.path.join(train_dir, fname)
    if not os.path.isfile(fpath):
        continue
    img = Image.open(fpath)
    train_images.append(np.array(img))
train_data = np.stack(train_images, axis=0)
train_labels = df["has_cactus"].values.astype(np.float32)



## === cell 3
test_filenames = []
test_images = []
for fname in os.listdir(test_dir):
    fpath = os.path.join(test_dir, fname)
    if not os.path.isfile(fpath):
        continue
    img = Image.open(fpath)
    test_images.append(np.array(img))
    test_filenames.append(fname)
test_data = np.stack(test_images, axis=0)



## === cell 4
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data.astype(np.float32).reshape([-1, 32, 32, 3])
test_data = test_data.astype(np.float32).reshape([-1, 32, 32, 3])



## === cell 5
tf.random.set_seed(219)




## === cell 6
class CactusModel(tf.keras.Model):
    def __init__(self):
        super(CactusModel, self).__init__()
        l2 = tf.keras.regularizers.l2(0.001)
        self.conv1 = layers.Conv2D(32, 5, padding="same", kernel_regularizer=l2)
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(64, 5, padding="same", kernel_regularizer=l2)
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D()
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(1024, kernel_regularizer=l2)
        self.bn3 = layers.BatchNormalization()
        self.drop = layers.Dropout(0.6)
        self.out = layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.pool1(x)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = self.pool2(x)
        x = self.flat(x)
        x = tf.nn.relu(self.bn3(self.fc1(x), training=training))
        x = self.drop(x, training=training)
        return self.out(x)


model = CactusModel()



## === cell 7
batch_size = 32
max_epochs = 20

train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_ds = train_ds.shuffle(10000).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(1e-4)
loss_fn = tf.keras.losses.BinaryCrossentropy()
accuracy = tf.keras.metrics.BinaryAccuracy()



## === cell 8
print("Start training")
for epoch in range(max_epochs):
    for step, (imgs, labs) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            preds = model(imgs, training=True)
            loss = loss_fn(tf.reshape(labs, (-1, 1)), preds)
            loss += tf.reduce_sum(model.losses)  # L2 regularization
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        accuracy.update_state(tf.reshape(labs, (-1, 1)), preds)

    print(
        f"Epoch {epoch+1}/{max_epochs} - loss: {loss.numpy():.4f} - acc: {accuracy.result().numpy()*100:.2f}%"
    )
    accuracy.reset_states()

print("Training completed")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1393387121.py in <cell line: 0>()
     14         f"Epoch {epoch+1}/{max_epochs} - loss: {loss.numpy():.4f} - acc: {accuracy.result().numpy()*100:.2f}%"
     15     )
---> 16     accuracy.reset_states()
     17 
     18 print("Training completed")

AttributeError: 'BinaryAccuracy' object has no attribute 'reset_states'

## === cell 9
test_probs = model(test_data, training=False).numpy().flatten()



## === cell 10
submission = pd.DataFrame({"id": test_filenames, "has_cactus": test_probs})
submission.to_csv("test_submission.csv", index=False)
print("Submission saved to test_submission.csv, rows:", len(submission))
