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
import os
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

BASE_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "../input",
]
BASE_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(c) and os.path.isdir(c):
        if os.path.exists(os.path.join(c, "train.csv")) or os.path.exists(
            os.path.join(c, "train")
        ):
            BASE_DIR = c
            break
if BASE_DIR is None:
    BASE_DIR = "../input"

print("Using BASE_DIR:", BASE_DIR)
print("BASE_DIR contents (partial):", sorted(os.listdir(BASE_DIR))[:20])


def resolve_dir(base, name):
    candidates = [
        os.path.join(base, name),
        os.path.join(base, name, name),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return None


TRAIN_DIR = resolve_dir(BASE_DIR, "train")
TEST_DIR = resolve_dir(BASE_DIR, "test")

if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not resolve TRAIN_DIR/TEST_DIR under BASE_DIR={BASE_DIR}"
    )

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.exists(TRAIN_CSV):
    if os.path.exists("../input/train.csv"):
        TRAIN_CSV = "../input/train.csv"
if not os.path.exists(SAMPLE_SUB):
    if os.path.exists("../input/sample_submission.csv"):
        SAMPLE_SUB = "../input/sample_submission.csv"

print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)



## === cell 1
df = pd.read_csv(TRAIN_CSV)
df.head()



## === cell 2
test_files_preview = sorted(
    [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
)[:10]
print(
    "Num test jpg:",
    len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]),
)
print("Preview:", test_files_preview)



## === cell 3
train_files_preview = sorted(
    [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]
)[:10]
print(
    "Num train jpg:",
    len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]),
)
print("Preview:", train_files_preview)



## === cell 4
data_dir = TRAIN_DIR
filename = df["id"].iloc[0]
path = os.path.join(data_dir, filename)
path

test_dir = TEST_DIR



## === cell 5
image_pil = Image.open(path)
image_pil



## === cell 6
image = np.array(image_pil)
plt.imshow(image)
plt.show()



## === cell 7
has_cactus = df["has_cactus"].iloc[0]
has_cactus



## === cell 8
image = np.array(image_pil)
plt.title(has_cactus)
plt.imshow(image)
plt.show()



## === cell 9
np.mean(df["has_cactus"])



## === cell 10
np.sum(df["has_cactus"]), len(df["has_cactus"])



## === cell 11
image.shape



## === cell 12
np.min(image), np.max(image)



## === cell 13
heights = []
widths = []
train_arr = []
test_arr = []

for filename in df["id"]:
    path = os.path.join(data_dir, filename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    train_arr.append(image)
    h, w, c = image.shape
    if h not in heights:
        heights.append(h)
    if w not in widths:
        widths.append(w)

train_data = np.array(train_arr)

train_labels_list = []
for label in df["has_cactus"]:
    train_labels_list.append(label)
train_labels = np.array(train_labels_list)

test_filenames_sorted = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)
for testfilename in test_filenames_sorted:
    path = os.path.join(test_dir, testfilename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    test_arr.append(image)

test_data = np.array(test_arr)

print(
    "train_data:",
    train_data.shape,
    "train_labels:",
    train_labels.shape,
    "test_data:",
    test_data.shape,
)



## === cell 14
from tqdm.auto import tqdm



## === cell 15
for filename in tqdm(df["id"].iloc[:50], desc="Quick read check (first 50)"):
    path = os.path.join(data_dir, filename)
    _ = Image.open(path)



## === cell 16
plt.imshow(train_data[12])
plt.show()



## === cell 17
train_data.shape



## === cell 18
train_labels[:10]



## === cell 19
heights



## === cell 20
widths



## === cell 21
import time
import tensorflow as tf
from tensorflow.keras import layers
from IPython.display import clear_output

print("TensorFlow version:", tf.__version__)

np.random.seed(219)
tf.random.set_seed(219)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 22
print("TensorFlow version: {}".format(tf.__version__))



## === cell 23
train_data = train_data / 255.0
train_data = train_data.reshape([-1, 32, 32, 3]).astype(np.float32)

train_labels = train_labels.astype(np.int32).astype(np.float64)

test_data = test_data / 255.0
test_data = test_data.reshape([-1, 32, 32, 3]).astype(np.float32)

print("train_data dtype/shape:", train_data.dtype, train_data.shape)
print("test_data dtype/shape :", test_data.dtype, test_data.shape)



## === cell 24
batch_size = 32
max_epochs = 20

N = len(train_data)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(
    buffer_size=10000, seed=219, reshuffle_each_iteration=True
)
train_dataset = train_dataset.batch(batch_size=batch_size)

print(train_dataset)




## === cell 25
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l2_decay = 0.001
        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay),
        )
        self.conv1_bn = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay),
        )
        self.conv2_bn = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(
            units=1024, kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay)
        )
        self.dense1_bn = layers.BatchNormalization()
        self.drop1 = layers.Dropout(rate=0.6)
        self.dense2 = layers.Dense(
            units=1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay),
        )

    def call(self, inputs, training=False):
        self.conv1_ = self.conv1(inputs)
        self.conv1_bn_ = self.conv1_bn(self.conv1_, training=training)
        self.conv1_ = tf.nn.relu(self.conv1_bn_)
        self.pool1_ = self.pool1(self.conv1_)

        self.conv2_ = self.conv2(self.pool1_)
        self.conv2_bn_ = self.conv2_bn(self.conv2_, training=training)
        self.conv2_ = tf.nn.relu(self.conv2_bn_)
        self.pool2_ = self.pool2(self.conv2_)

        self.flatten_ = self.flatten(self.pool2_)
        self.dense1_ = self.dense1(self.flatten_)
        self.dense1_bn_ = self.dense1_bn(self.dense1_, training=training)
        self.dense1_ = tf.nn.relu(self.dense1_bn_)
        self.drop1_ = self.drop1(self.dense1_, training=training)

        self.predictions_ = self.dense2(self.drop1_)
        return self.predictions_




## === cell 26
model = MNISTModel()



## === cell 27
for images, labels in train_dataset.take(1):
    predictions = model(images[0:1])
    print("Predictions: ", predictions.numpy())
    print(labels[0:1].numpy())



## === cell 28
_ = model(tf.zeros([1, 32, 32, 3], dtype=tf.float32))
model.summary()



## === cell 29
loss_object = tf.keras.losses.BinaryCrossentropy()
acc_object = tf.keras.metrics.BinaryAccuracy()



## === cell 30
mean_help = tf.keras.metrics.Mean
print("tf.keras.metrics.Mean available:", mean_help)



## === cell 31
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

mean_ce = tf.keras.metrics.Mean(name="binary_cross_entropy")
mean_l2 = tf.keras.metrics.Mean(name="l2_loss")
mean_total_loss = tf.keras.metrics.Mean(name="total_loss")

cross_entropy_history = []
l2_loss_history = []
total_loss_history = []
accuracy_history = [(0, 0.0)]



## === cell 32
print("start training!")
global_step = 0
num_batches_per_epoch = int(N / batch_size)

for epoch in range(max_epochs):
    for step, (images, labels) in enumerate(train_dataset):
        start_time = time.time()

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            labels = tf.reshape(labels, [tf.shape(labels)[0], 1])
            labels = tf.cast(labels, tf.float32)

            binary_cross_entropy = loss_object(labels, predictions)
            l2_loss = tf.reduce_sum(model.losses)
            total_loss = binary_cross_entropy + l2_loss
            acc_value = acc_object(labels, predictions)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_step += 1

        mean_ce(binary_cross_entropy)
        mean_l2(l2_loss)
        mean_total_loss(total_loss)

        cross_entropy_history.append((global_step, float(mean_ce.result().numpy())))
        l2_loss_history.append((global_step, float(mean_l2.result().numpy())))
        total_loss_history.append(
            (global_step, float(mean_total_loss.result().numpy()))
        )

        if global_step % 200 == 0:
            clear_output(wait=True)
            epochs = epoch + step / float(num_batches_per_epoch)
            duration = time.time() - start_time
            examples_per_sec = batch_size / float(duration)
            print(
                "epochs: {:.2f}, step: {}, loss: {:.4g}, accuracy: {:.4g}% ({:.2f} examples/sec; {:.4f} sec/batch)".format(
                    epochs,
                    global_step,
                    mean_total_loss.result().numpy(),
                    acc_value.numpy() * 100,
                    examples_per_sec,
                    duration,
                )
            )

    accuracy_history.append((global_step, float(acc_value.numpy() * 100)))

    mean_ce.reset_state()
    mean_l2.reset_state()
    mean_total_loss.reset_state()
    acc_object.reset_state()

print("training done!")



## === cell 33
df.to_csv("train_debug_dump.csv", index=False)
print("Wrote train_debug_dump.csv")



## === cell 34
test_data[0].shape



## === cell 35
test_predictions = (
    model(test_data, training=False).numpy().reshape(-1).astype(np.float64)
)
print(
    "Pred stats:",
    float(test_predictions.min()),
    float(test_predictions.max()),
    float(test_predictions.mean()),
)



## === cell 36
sub = pd.read_csv(SAMPLE_SUB)

pred_map = dict(zip(test_filenames_sorted, test_predictions))

sub["has_cactus"] = sub["id"].map(pred_map)
if sub["has_cactus"].isna().any():
    missing = sub.loc[sub["has_cactus"].isna(), "id"].head(10).tolist()
    raise ValueError(f"Missing predictions for some ids (showing up to 10): {missing}")

sub.to_csv("submission.csv", index=False)
sub.to_csv("test_submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
