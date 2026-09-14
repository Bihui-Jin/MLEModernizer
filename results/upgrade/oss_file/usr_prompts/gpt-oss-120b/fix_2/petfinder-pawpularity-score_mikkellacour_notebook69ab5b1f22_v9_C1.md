# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input, ResNet101
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping



## === cell 1
train_img_dir = r"/kaggle/input/petfinder-pawpularity-score/train"
test_img_dir = r"/kaggle/input/petfinder-pawpularity-score/test"
train_csv_path = r"/kaggle/input/petfinder-pawpularity-score/train.csv"
test_csv_path = r"/kaggle/input/petfinder-pawpularity-score/test.csv"

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)




## === cell 2
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img)
    return preprocess_input(arr)




## === cell 3
feature_extractor = ResNet101(weights="imagenet", include_top=False, pooling="avg")




## === cell 4
def extract_image_features(df, img_dir):
    feats = []
    ids = []
    batch = []
    batch_paths = []
    batch_size = 64

    for _, row in df.iterrows():
        img_id = row["Id"]
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            feats.append(np.zeros(2048, dtype=np.float32))
            ids.append(img_id)
            continue
        img = load_and_preprocess_image(img_path)
        batch.append(img)
        batch_paths.append(img_id)

        if len(batch) == batch_size:
            batch_arr = np.stack(batch, axis=0)
            batch_feats = feature_extractor.predict(batch_arr, verbose=0)
            feats.extend(batch_feats)
            ids.extend(batch_paths)
            batch = []
            batch_paths = []

    if batch:
        batch_arr = np.stack(batch, axis=0)
        batch_feats = feature_extractor.predict(batch_arr, verbose=0)
        feats.extend(batch_feats)
        ids.extend(batch_paths)

    return np.array(feats, dtype=np.float32), np.array(ids)


print("Extracting train image features …")
train_img_feats, _ = extract_image_features(df_train, train_img_dir)
print("Extracting test image features …")
test_img_feats, _ = extract_image_features(df_test, test_img_dir)



## === cell 5
metadata_cols = [c for c in df_train.columns if c not in ["Id", "Pawpularity"]]
X_meta_train = df_train[metadata_cols].values.astype(np.float32)
X_meta_test = df_test[metadata_cols].values.astype(np.float32)

y = df_train["Pawpularity"].values.astype(np.float32)



## === cell 6
X_img_tr, X_img_val, X_meta_tr, X_meta_val, y_tr, y_val = train_test_split(
    train_img_feats, X_meta_train, y, test_size=0.2, random_state=42
)

image_input = Input(shape=(train_img_feats.shape[1],))
x = Dense(256, activation="relu")(image_input)

feature_input = Input(shape=(X_meta_train.shape[1],))
y_feat = Dense(64, activation="relu")(feature_input)

combined = Concatenate()([x, y_feat])
z = Dense(256, activation="relu")(combined)
z = Dropout(0.2)(z)
z = Dense(512, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(128, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(48, activation="relu")(z)
z = Dropout(0.2)(z)
z = Dense(10, activation="relu")(z)
z = Dropout(0.3)(z)
output = Dense(1, activation="linear")(z)  # regression output

model = Model(inputs=[image_input, feature_input], outputs=output)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-7,
    patience=20,
    verbose=1,
    restore_best_weights=True,
)

history = model.fit(
    [X_img_tr, X_meta_tr],
    y_tr,
    validation_data=([X_img_val, X_meta_val], y_val),
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    shuffle=True,
    verbose=2,
)



## === cell 7
loss, mae = model.evaluate([X_img_val, X_meta_val], y_val, verbose=0)
rmse = np.sqrt(loss)
print(f"Validation RMSE: {rmse:.4f}")



## === cell 8
test_preds = model.predict([test_img_feats, X_meta_test], verbose=0).flatten()
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({"Id": df_test["Id"], "Pawpularity": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")
