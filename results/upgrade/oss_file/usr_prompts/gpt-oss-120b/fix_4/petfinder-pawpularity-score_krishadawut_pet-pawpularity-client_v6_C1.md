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

# 5. Target score

20.60746535109428

# 6. Current score

22.9013

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 23.38155) has done: 'The script now pre‑computes EfficientNetB0 features once for all images, then trains only the lightweight dense head on those fixed features. This preserves the exact architecture (EfficientNetB0 → BatchNorm → Dense → Dense → Dropout → Dense) while removing the costly per‑epoch forward passes through EfficientNet, cutting runtime well below the 600‑second limit. The rest of the workflow (data loading, evaluation, submission) remains unchanged.'
- What this solution (achieved 22.9013) has done: 'I avoid the TensorFlow import error by loading it only inside a try‑except block; if TensorFlow cannot be imported the script falls back to a lightweight scikit‑learn GradientBoostingRegressor that uses only the provided metadata features. This keeps the original EfficientNet‑based approach when possible, fixes the runtime crash, and adds a simple yet effective model that should bring the RMSE closer to the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob, cv2, numpy as np, pandas as pd

np.random.seed(42)

BASE_PATH = "/kaggle/input/petfinder-pawpularity-score"

train_csv = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_csv = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

train_data = train_csv.drop(columns=["Id", "Pawpularity"]).values.astype(np.float32)
train_label = train_csv["Pawpularity"].values.astype(np.float32)

test_data = test_csv.drop(columns=["Id"]).values.astype(np.float32)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    train_imgs_path = os.path.join(BASE_PATH, "train")
    train_imgs_files = sorted(glob.glob(os.path.join(train_imgs_path, "*.jpg")))
    train_imgs = np.stack(
        [cv2.resize(cv2.imread(f), (224, 224)) for f in train_imgs_files]
    )

    test_imgs_path = os.path.join(BASE_PATH, "test")
    test_imgs_files = sorted(glob.glob(os.path.join(test_imgs_path, "*.jpg")))
    test_imgs = np.stack(
        [cv2.resize(cv2.imread(f), (224, 224)) for f in test_imgs_files]
    )

    eff = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    eff.trainable = False

    train_features = eff.predict(train_imgs, batch_size=32, verbose=0)
    test_features = eff.predict(test_imgs, batch_size=32, verbose=0)

    inputs = keras.Input(shape=(train_features.shape[1],), name="features")
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, name="Pawpularity")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PawpularityRegressor")

    def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(
        optimizer="rmsprop",
        loss=root_mean_squared_error,
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    model.fit(
        train_features[:7930], train_label[:7930], epochs=10, batch_size=32, verbose=2
    )

    val_results = model.evaluate(train_features[7930:], train_label[7930:], verbose=0)
    print("Validation loss, validation RMSE:", val_results)

    predictions = model.predict(test_features, batch_size=32).flatten()

except Exception as e:
    print("TensorFlow import or execution failed, falling back to scikit-learn model.")
    print("Error details:", e)

    from sklearn.ensemble import GradientBoostingRegressor

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(train_data[:7930], train_label[:7930])

    val_pred = gbr.predict(train_data[7930:])
    val_rmse = np.sqrt(np.mean((val_pred - train_label[7930:]) ** 2))
    print("Validation RMSE (GradientBoosting):", val_rmse)

    predictions = gbr.predict(test_data)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
submission = pd.DataFrame({"Id": test_csv["Id"], "Pawpularity": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, header=True)
print(f"Submission file written to {submission_path}")
submission.head()
