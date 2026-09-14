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

20.74227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

try:
    import tensorflow_addons as tfa

    _has_tfa = True
except Exception:
    _has_tfa = False
    from tensorflow.keras.optimizers import experimental as tf_optim_experimental

    tfa = None




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
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]




## === cell 2
def sample_images(images, row_count, column_count):
    fig, axs = plt.subplots(row_count, column_count, figsize=(10, 10))
    for i in range(row_count):
        for j in range(column_count):
            axs[i, j].imshow(images[i * column_count + j])
            axs[i, j].axis("off")
    plt.show()




## === cell 3
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)




## === cell 4
train.head()




## === cell 5
train["file_path"] = train["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/train/"
    + identifier
    + ".jpg"
)
test["file_path"] = test["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/test/"
    + identifier
    + ".jpg"
)




## === cell 6
train.head()




## === cell 7
train["Pawpularity"].hist()




## === cell 8
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




## === cell 9
def preprocess_image(image_url):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (Config.image_size, Config.image_size))
    return image




## === cell 10
def preprocess(image_url, tabular):
    image = preprocess_image(image_url)
    return (image, tabular[1:]), tabular[0]




## === cell 11
augmentation_layer = keras.Sequential(
    [
        keras.layers.Input(Config.input_shape),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]
)




## === cell 12
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x




## === cell 13
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




## === cell 14
index = np.random.choice(train.shape[0])
plt.figure(figsize=(4, 4))
image = preprocess_image(tf.constant(train.iloc[index]["file_path"], dtype=tf.string))
print(image.shape)
plt.imshow(np.squeeze(image))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(Config.image_size, Config.image_size)
)
print(resized_image.shape)
patches = Patches(Config.patch_size)(resized_image)
print(f"Image size: {Config.image_size} X {Config.image_size}")
print(f"Patch size: {Config.patch_size} X {Config.patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (Config.patch_size, Config.patch_size, 3))
    plt.imshow(patch_img.numpy())
    plt.axis("off")




## === cell 15
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




## === cell 16
def create_vision_transformer(use_tabular_inputs=False):

    tabular_inputs = tf.keras.Input(shape=(len(tabular_columns),))
    inputs = layers.Input(shape=Config.input_shape)
    augmented = augmentation_layer(inputs)
    patches = Patches(Config.patch_size)(augmented)
    encoder_patches = PatchEncoder(Config.num_patches, Config.projection_dim)(patches)

    for _ in range(Config.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=Config.num_heads, key_dim=Config.projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoder_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=Config.transformer_units, dropout_rate=0.1)
        encoder_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=Config.mlp_head_units, dropout_rate=0.5)

    if use_tabular_inputs:
        image_x = layers.Dense(128, activation=tf.nn.gelu)(features)
        tabular_x = mlp(tabular_inputs, hidden_units=[16] * 10, dropout_rate=0.5)
        x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    else:
        x = features

    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=[inputs, tabular_inputs], outputs=outputs)
    return model




## === cell 17
model = create_vision_transformer(True)
tf.keras.utils.plot_model(model, show_shapes=True)




## === cell 18
image = np.random.normal(size=(1, Config.image_size, Config.image_size, 3)).astype(
    np.float32
)
tabular = np.random.normal(size=(1, len(tabular_columns))).astype(np.float32)
print(image.shape, tabular.shape)
print(model((image, tabular)).shape)




## === cell 19
tf.keras.backend.clear_session()
models = []
historys = []
kfold = KFold(n_splits=5, shuffle=True, random_state=997)
train_best_fold = True
best_fold = 4
for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_best_fold and index != best_fold:
        continue
    x_train = train.loc[train_indices, "file_path"]
    tabular_train = train.loc[train_indices, ["Pawpularity"] + tabular_columns]
    x_val = train.loc[val_indices, "file_path"]
    tabular_val = train.loc[val_indices, ["Pawpularity"] + tabular_columns]
    checkpoint_path = f"model_{index}.weights.h5"  # must end with .weights.h5
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(min_delta=1e-4, patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=2, min_lr=1e-7
    )
    callbacks = [early_stop, checkpoint, reduce_lr]

    loss = tf.keras.losses.MeanSquaredError()

    if _has_tfa:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=Config.learning_rate, weight_decay=Config.weight_decay
        )
    else:
        optimizer = tf_optim_experimental.AdamW(
            learning_rate=Config.learning_rate, weight_decay=Config.weight_decay
        )
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, tabular_train))
        .map(preprocess)
        .shuffle(512)
        .batch(Config.batch_size)
        .cache()
        .prefetch(1)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, tabular_val))
        .map(preprocess)
        .batch(Config.batch_size)
        .cache()
        .prefetch(1)
    )
    model = create_vision_transformer(use_tabular_inputs=True)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"],
    )
    history = model.fit(
        train_ds,
        epochs=Config.num_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )
    model.load_weights(checkpoint_path)
    historys.append(history)
    models.append(model)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2669571613.py in <cell line: 0>()
     30         )
     31     else:
---> 32         optimizer = tf_optim_experimental.AdamW(
     33             learning_rate=Config.learning_rate, weight_decay=Config.weight_decay
     34         )

NameError: name 'tf_optim_experimental' is not defined

## === cell 20
def preprocess_test_data(image_url, tabular):
    image = preprocess_image(image_url)
    return (image, tabular), 0




## === cell 21
test_ds = (
    tf.data.Dataset.from_tensor_slices((test["file_path"], test[tabular_columns]))
    .map(preprocess_test_data)
    .batch(Config.batch_size)
    .cache()
    .prefetch(1)
)




## === cell 22
use_best_result = True
if use_best_result:
    if train_best_fold:
        best_model = models[0]  # only the selected best fold was trained
    else:
        best_score = float("inf")
        best_idx = 0
        for fold, history in enumerate(historys):
            for val_rmse in history.history["val_rmse"]:
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_idx = fold
        best_model = models[best_idx]
    preds = best_model.predict(test_ds).reshape(-1)
    sample_submission["Pawpularity"] = preds
    sample_submission.to_csv("submission.csv", index=False)
else:
    total_results = []
    for m in models:
        total_results.append(m.predict(test_ds).reshape(-1))
    results = np.mean(total_results, axis=0)
    sample_submission["Pawpularity"] = results
    sample_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/2497041215.py in <cell line: 0>()
      2 if use_best_result:
      3     if train_best_fold:
----> 4         best_model = models[0]  # only the selected best fold was trained
      5     else:
      6         best_score = float("inf")

IndexError: list index out of range
