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

20.50497

# 6. Current score

22.69145

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 65.01253) has done: 'I add a protobuf compatibility fix before importing TensorFlow, correct the RMSE metric to cast inputs to the same dtype, and make the prediction step robust by handling the case where no models were trained (using the training mean as a fallback). These minimal changes resolve the import error, the datatype mismatch during training, and the index error when creating the submission, allowing the pipeline to run end‑to‑end and output a valid submission.csv.'
- What this solution (achieved 22.69145) has done: 'I fix the protobuf import error, ensure the best‑weights are restored after early stopping, train on all folds instead of just one, and average predictions from all trained models (instead of picking a single fold). These changes keep the original architecture and training pipeline while improving model performance and guaranteeing a valid submission.csv.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class Config:
    image_size = 128
    input_shape = [image_size, image_size, 3]
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 128
    num_classes = 1
    num_epochs = 30
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    scale_factor = 100.0
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    tabular_columns = [
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




## === cell 2
def display_images(images, row_count, column_count):
    fig, axs = plt.subplots(row_count, column_count, figsize=(10, 10))
    for i in range(row_count):
        for j in range(column_count):
            axs[i, j].imshow(images[i * column_count + j])
            axs[i, j].axis("off")
    plt.show()




## === cell 3
def preprocess_image(image_url):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (Config.image_size, Config.image_size))
    return image




## === cell 4
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x




## === cell 5
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = (y_true - y_pred) * Config.scale_factor
    return tf.sqrt(tf.reduce_mean(tf.square(diff)))




## === cell 6
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)




## === cell 7
train["file_path"] = train["Id"].apply(
    lambda identifier: f"../input/petfinder-pawpularity-score/train/{identifier}.jpg"
)
test["file_path"] = test["Id"].apply(
    lambda identifier: f"../input/petfinder-pawpularity-score/test/{identifier}.jpg"
)




## === cell 8
train["Pawpularity"] /= Config.scale_factor




## === cell 9
def preprocess(image_url, tabular):
    image = preprocess_image(image_url)
    return (image, tabular[1:]), tabular[0]




## === cell 10
augmentation_layer = keras.Sequential(
    [
        layers.Input(shape=Config.input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]
)




## === cell 11
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
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




## === cell 12
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded




## === cell 13
def create_vision_transformer(use_tabular_inputs=False):
    image_inputs = layers.Input(shape=Config.input_shape, name="image_input")
    tabular_inputs = layers.Input(
        shape=(len(Config.tabular_columns),), name="tabular_input"
    )

    augmented = augmentation_layer(image_inputs)
    patches = Patches(Config.patch_size)(augmented)
    encoded_patches = PatchEncoder(Config.num_patches, Config.projection_dim)(patches)

    x = encoded_patches
    for _ in range(Config.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=Config.num_heads, key_dim=Config.projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=Config.transformer_units, dropout_rate=0.1)
        x = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=Config.mlp_head_units, dropout_rate=0.5)

    if use_tabular_inputs:
        image_feat = layers.Dense(128, activation=tf.nn.gelu)(features)
        tabular_feat = mlp(tabular_inputs, hidden_units=[16] * 10, dropout_rate=0.5)
        combined = layers.Concatenate(axis=1)([image_feat, tabular_feat])
    else:
        combined = features

    outputs = layers.Dense(1, activation="sigmoid")(combined)

    if use_tabular_inputs:
        model = keras.Model(inputs=[image_inputs, tabular_inputs], outputs=outputs)
    else:
        model = keras.Model(inputs=image_inputs, outputs=outputs)
    return model




## === cell 14
tf.keras.utils.plot_model(create_vision_transformer(True), show_shapes=True)




## === cell 15
tf.keras.backend.clear_session()
models = []
historys = []
kfold = KFold(n_splits=5, shuffle=True, random_state=997)
train_best_fold = False

for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_best_fold and index != best_fold:
        continue

    x_train = train.loc[train_indices, "file_path"].values
    tabular_train = train.loc[
        train_indices, ["Pawpularity"] + Config.tabular_columns
    ].values
    x_val = train.loc[val_indices, "file_path"].values
    tabular_val = train.loc[
        val_indices, ["Pawpularity"] + Config.tabular_columns
    ].values

    checkpoint_path = f"model_{index}.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        min_delta=1e-4, patience=20, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=10, min_lr=1e-7, monitor="val_loss", mode="min"
    )
    callbacks = [checkpoint, reduce_lr, early_stop]

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=Config.learning_rate, weight_decay=Config.weight_decay
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, tabular_train))
        .map(preprocess)
        .shuffle(512)
        .batch(Config.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, tabular_val))
        .map(preprocess)
        .batch(Config.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = create_vision_transformer(use_tabular_inputs=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=[rmse, "mae", "mape"])

    history = model.fit(
        train_ds,
        epochs=Config.num_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )

    model.load_weights(checkpoint_path)
    models.append(model)
    historys.append(history)




## === cell 16
def preprocess_test_data(image_url, tabular):
    image = preprocess_image(image_url)
    return (image, tabular), 0.0  # dummy label (unused)




## === cell 17
test_ds = (
    tf.data.Dataset.from_tensor_slices(
        (test["file_path"].values, test[Config.tabular_columns].values)
    )
    .map(preprocess_test_data)
    .batch(Config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 18
use_best_result = False
if use_best_result and models:
    best_model = models[0]  # retained for compatibility but not used
    predictions = best_model.predict(test_ds, verbose=0)
    sample_submission["Pawpularity"] = Config.scale_factor * predictions.reshape(-1)
else:
    if models:
        total = [m.predict(test_ds, verbose=0).reshape(-1) for m in models]
        avg_pred = np.mean(total, axis=0)
        sample_submission["Pawpularity"] = Config.scale_factor * avg_pred
    else:
        mean_pred = train["Pawpularity"].mean() * Config.scale_factor
        sample_submission["Pawpularity"] = np.full(len(test), mean_pred)

sample_submission.to_csv("submission.csv", index=False)
