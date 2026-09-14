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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

22.84361

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm



## === cell 1
base_path = "../input/petfinder-pawpularity-score/"
train_df = pd.read_csv(base_path + "train.csv")
test_df = pd.read_csv(base_path + "test.csv")

train_jpg = glob(base_path + "train/*.jpg")
test_jpg = glob(base_path + "test/*.jpg")



## === cell 2
print(train_df.shape, len(train_jpg))



## === cell 3
train_df.head(5)



## === cell 4
sns.histplot(train_df["Pawpularity"])
plt.axvline(train_df["Pawpularity"].mean(), color="red")
plt.axvline(train_df["Pawpularity"].median(), color="green")
plt.show()



## === cell 5
train_df[["Pawpularity"]].describe().T



## === cell 6
train_df.columns



## === cell 7
cols = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]

for col in cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.violinplot(data=train_df, y="Pawpularity", x=col, ax=ax[0])
    sns.histplot(data=train_df, x="Pawpularity", hue=col, kde=True, fill=True, ax=ax[1])
    plt.suptitle(str(col), fontsize=18, fontweight="bold")
    plt.show()



## === cell 8
img_path = base_path + "train/"
ext = ".jpg"
nums = len(cols)

for count, col in enumerate(cols):
    sample_ids = train_df.loc[train_df[col] == 1, "Id"].values
    if len(sample_ids) == 0:
        continue
    sample = np.random.choice(sample_ids)
    pawpularity = train_df.loc[train_df["Id"] == sample, "Pawpularity"].values[0]
    image_loc = img_path + sample + ext
    image_array = plt.imread(image_loc)
    plt.imshow(image_array)
    plt.title(f"Image of pet with {col}\nPawpularity Score: {pawpularity}")
    plt.axis("off")
    plt.show()
    del sample, image_loc, image_array
    gc.collect()




## === cell 9
def train_id_to_path(x):
    return base_path + "train/" + x + ".jpg"


def test_id_to_path(x):
    return base_path + "test/" + x + ".jpg"




## === cell 10
train = pd.read_csv(base_path + "train.csv")
train = train.drop(
    columns=[
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
)

test = pd.read_csv(base_path + "test.csv")
test = test.drop(
    columns=[
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
)



## === cell 11
train["img_path"] = train["Id"].apply(train_id_to_path)
test["img_path"] = test["Id"].apply(test_id_to_path)



## === cell 12
train["two_bin_pawp"] = pd.qcut(train["Pawpularity"], q=2, labels=False).astype(str)
train["four_bin_pawp"] = pd.qcut(train["Pawpularity"], q=4, labels=False).astype(str)
train["ten_bin_pawp"] = pd.qcut(train["Pawpularity"], q=10, labels=False).astype(str)



## === cell 13
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_height = 128
image_width = 128


def path_to_eagertensor(image_path):
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (image_height, image_width))
    return image




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
X = []
for img_path in tqdm(train["img_path"].values, desc="Loading train images"):
    X.append(path_to_eagertensor(img_path).numpy())
X = np.array(X, dtype=np.float32)
print("Train images array:", X.shape, X.dtype)



## === cell 15
X_submission = []
for img_path in tqdm(test["img_path"].values, desc="Loading test images"):
    X_submission.append(path_to_eagertensor(img_path).numpy())
X_submission = np.array(X_submission, dtype=np.float32)
print("Test images array:", X_submission.shape, X_submission.dtype)



## === cell 16
y = train["Pawpularity"].astype(np.float32).values
print("Target vector shape:", y.shape)



## === cell 17
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=2021
)

print("Split shapes ->", x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 18
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 128
patch_size = 12
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 6
mlp_head_units = [2048, 1024]
num_classes = 1
input_shape = (image_size, image_size, 3)



## === cell 19
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(x_train)




## === cell 20
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x




## === cell 21
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches




## === cell 22
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded




## === cell 23
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    patches = Patches(patch_size)(x)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model




## === cell 24
def run_experiment(model):
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        verbose=2,
    )
    return history




## === cell 25
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3248664556.py in <cell line: 0>()
      1 vit_classifier = create_vit_classifier()
----> 2 history = run_experiment(vit_classifier)
      3 

/tmp/ipykernel_55/1315195198.py in run_experiment(model)
      1 def run_experiment(model):
----> 2     optimizer = tf.keras.optimizers.experimental.AdamW(
      3         learning_rate=learning_rate, weight_decay=weight_decay
      4     )
      5     model.compile(

AttributeError: module 'keras.api.optimizers' has no attribute 'experimental'

## === cell 26
cnn_pred = vit_classifier.predict(X_submission, batch_size=batch_size)
cnn_pred = cnn_pred.squeeze()
print("Predictions shape:", cnn_pred.shape)



## === cell 27
submission = pd.DataFrame({"Id": test["Id"], "Pawpularity": cnn_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
