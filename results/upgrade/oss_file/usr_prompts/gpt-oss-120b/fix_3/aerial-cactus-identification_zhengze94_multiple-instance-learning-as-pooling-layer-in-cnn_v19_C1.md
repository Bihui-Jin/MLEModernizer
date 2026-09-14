# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.6222

# 6. Current score

0.99859

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99859) has done: 'I fix the protobuf import error by setting the environment variable before importing TensorFlow, and correct the model architecture so the NoisyAnd layer’s output shape matches the next layer (remove the incompatible GlobalAveragePooling2D). These minimal changes let the code run end‑to‑end and produce a proper sample_submission.csv.'

# 9. Code solution

## === cell 0
import os, glob, cv2, numpy as np, pandas as pd
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Layer,
)
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

BASE_INPUT = "../input/aerial-cactus-identification"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUBMIT = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Folders:", os.listdir(BASE_INPUT)[:5])



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # force 3‑channel BGR
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = img.astype(np.float32) / 255.0  # normalize
    return img


X_train_list = []
Y_train_list = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading train"):
    img_path = os.path.join(TRAIN_IMG_DIR, row["id"])
    X_train_list.append(load_image(img_path))
    Y_train_list.append(row["has_cactus"])
X_train = np.stack(X_train_list)  # shape (N,32,32,3)
Y_train = np.array(Y_train_list).astype(np.float32)

test_df = pd.read_csv(SAMPLE_SUBMIT)  # same structure as sample submission
X_test_list = []
for img_id in tqdm(test_df["id"], desc="Loading test"):
    img_path = os.path.join(TEST_IMG_DIR, img_id)
    X_test_list.append(load_image(img_path))
X_test = np.stack(X_test_list)

print("Train shape:", X_train.shape, "Labels shape:", Y_train.shape)
print("Test shape:", X_test.shape)



## === cell 2
x_tr, x_val, y_tr, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)




## === cell 3
class NoisyAnd(Layer):
    def __init__(self, a=20, **kwargs):
        super(NoisyAnd, self).__init__(**kwargs)
        self.a = max(1, a)

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.b = self.add_weight(
            name="b", shape=(1, channel_dim), initializer="uniform", trainable=True
        )
        super(NoisyAnd, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])  # (batch, C)
        term1 = tf.nn.sigmoid(self.a * (mean - self.b))
        term2 = tf.nn.sigmoid(-self.a * self.b)
        numerator = term1 - term2
        denominator = tf.nn.sigmoid(self.a * (1 - self.b)) - term2
        return numerator / denominator

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




## === cell 4
def define_model(input_shape=(32, 32, 3), num_classes=1):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (1, 1), activation="relu"))

    model.add(NoisyAnd())
    model.add(Dense(num_classes, activation="sigmoid"))
    return model


model = define_model()
model.summary()



## === cell 5
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

EPOCHS = 5
history = model.fit(
    x_tr, y_tr, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=32, verbose=1
)



## === cell 6
preds = model.predict(X_test, batch_size=32).reshape(-1)

submission = pd.read_csv(SAMPLE_SUBMIT)
submission["has_cactus"] = preds
submission_path = "sample_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
