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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.4282875439153069

# 6. Current score

0.6781

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.36629) has done: 'The fix updates the imports to use `tensorflow.keras` (avoiding the failing plain `keras` import), corrects the dataset paths, creates proper ImageDataGenerators for the train and test folders, builds a ResNet50‑based model (matching the original architecture intent), trains it for a few epochs, predicts the dog‑class probability for each test image, and writes a correctly formatted `submission.csv` with the required **id** and **label** columns. These changes resolve the runtime errors and ensure a valid submission file is produced, moving the solution toward the target log‑loss.'
- What this solution (achieved 0.77049) has done: 'I fix the import warning, update the Adam optimizer argument, and ensure the model is properly compiled. After the initial training I unfreeze the last few convolutional layers of ResNet50 and fine‑tune them for a couple more epochs to improve validation loss, which should bring the log‑loss closer to the target. Finally, I keep the same submission‑writing logic so a valid `submission.csv` is produced.'
- What this solution (achieved 0.72468) has done: 'Implemented targeted speed‑ups while keeping the model architecture, training loops, and evaluation unchanged.  
1. Removed costly real‑time augmentations (shear, zoom, flip) from the `ImageDataGenerator` to speed image preprocessing.  
2. Enabled parallel data loading in both training phases by adding `workers=4` and `use_multiprocessing=True` to `model.fit`.  
3. Used `math.ceil` for step calculations to ensure full dataset coverage without extra loops.  
These changes reduce I/O and preprocessing overhead, allowing the script to complete well within the 600‑second limit while preserving the original learning logic and prediction output.'
- What this solution (achieved 0.84119) has done: 'I fix the protobuf import error by setting the appropriate environment variable before loading TensorFlow, remove the illegal intra‑op threading configuration (which caused a RuntimeError), and drop unsupported `workers`/`use_multiprocessing` arguments from the prediction call. These minimal changes eliminate the runtime crashes while preserving the original model and training logic, allowing a valid `submission.csv` to be produced and moving the log‑loss toward the target.'
- What this solution (achieved 0.69315) has done: 'I replace the failing TensorFlow‑based pipeline with a safe fallback that computes the overall dog‑class prevalence from the training folders and uses this constant probability for every test image. This removes the protobuf import error and the unsupported `workers` argument, while still producing a correctly formatted `submission.csv`. The constant prediction (≈0.5) improves the log‑loss from 0.84 to about 0.69, moving the score closer to the target without altering the core modeling intent.'
- What this solution (achieved 0.67414) has done: 'I add a lightweight TensorFlow Keras model (MobileNetV2) that trains briefly on the cat/dog folders and uses it to generate per‑image probabilities instead of the constant class‑prior prediction. The new code is wrapped in a try/except so that if TensorFlow is unavailable the original constant‑baseline logic still runs, keeping the script safe while likely reducing the log‑loss toward the target.'
- What this solution (achieved 0.6781) has done: 'I keep the original TensorFlow pipeline but catch its import error. When TensorFlow fails, I replace it with a lightweight fallback that trains a logistic‑regression model on the mean RGB values of each image (a fast yet more informative feature than a constant prior). This fixes the protobuf `AttributeError`, ensures a valid `submission.csv` is written, and provides better predictions than the constant baseline, moving the log‑loss closer to the target.'

# 9. Code solution

## === cell 0
import os
import math
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

default_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
if os.path.isdir(default_path):
    base_path = default_path
else:
    base_path = "../input/dogs-vs-cats-redux-kernels-edition/"

train_data_dir = os.path.join(base_path, "train")
test_data_dir = os.path.join(
    base_path, "test"
)  # contains subfolders 'test' and 'unknown'

print("Train dir exists:", os.path.isdir(train_data_dir))
print("Test dir exists :", os.path.isdir(test_data_dir))



## === cell 1
cat_dir = os.path.join(train_data_dir, "cat")
dog_dir = os.path.join(train_data_dir, "dog")

num_cat = len([f for f in os.listdir(cat_dir) if f.lower().endswith(".jpg")])
num_dog = len([f for f in os.listdir(dog_dir) if f.lower().endswith(".jpg")])

total = num_cat + num_dog
dog_ratio = num_dog / total if total > 0 else 0.5
print(f"Training set: {num_cat} cats, {num_dog} dogs → dog_ratio = {dog_ratio:.5f}")

test_ids = []
test_path_map = {}
for root, _, files in os.walk(test_data_dir):
    for f in files:
        if f.lower().endswith(".jpg"):
            img_id = os.path.splitext(f)[0]
            test_ids.append(img_id)
            test_path_map[img_id] = os.path.join(root, f)

test_ids = sorted(test_ids, key=lambda x: int(x) if x.isdigit() else x)
print(f"Found {len(test_ids)} test images.")



## === cell 2
try:
    import tensorflow as tf

    tf.random.set_seed(42)

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 2  # short training to stay within runtime limits

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.1,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False  # freeze base

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=2,
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )

    preds = model.predict(test_generator, verbose=0).flatten()
    pred_probs = np.clip(preds, 1e-7, 1 - 1e-7)

    ids_from_gen = [
        os.path.splitext(os.path.basename(fp))[0] for fp in test_generator.filenames
    ]
    submission = pd.DataFrame({"id": ids_from_gen, "label": pred_probs})
    submission = submission.set_index("id").loc[test_ids].reset_index()

except Exception as e:
    print("TensorFlow model failed or not available:", e)

    from sklearn.linear_model import LogisticRegression
    from PIL import Image

    IMG_SMALL = (64, 64)  # modest size for fast feature extraction

    def load_features(dir_path, label):
        feats = []
        labs = []
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith(".jpg"):
                continue
            fp = os.path.join(dir_path, fname)
            try:
                img = Image.open(fp).convert("RGB").resize(IMG_SMALL)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                mean_rgb = arr.mean(axis=(0, 1))  # shape (3,)
                feats.append(mean_rgb)
                labs.append(label)
            except Exception:
                continue
        return np.array(feats), np.array(labs)

    cat_features, cat_labels = load_features(cat_dir, 0)
    dog_features, dog_labels = load_features(dog_dir, 1)

    X_train = np.vstack([cat_features, dog_features])
    y_train = np.concatenate([cat_labels, dog_labels])

    lr = LogisticRegression(solver="lbfgs", max_iter=200)
    lr.fit(X_train, y_train)

    test_feats = []
    for img_id in test_ids:
        fp = test_path_map.get(img_id)
        if fp and os.path.exists(fp):
            try:
                img = Image.open(fp).convert("RGB").resize(IMG_SMALL)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                mean_rgb = arr.mean(axis=(0, 1))
                test_feats.append(mean_rgb)
            except Exception:
                test_feats.append([0.5, 0.5, 0.5])
        else:
            test_feats.append([0.5, 0.5, 0.5])

    test_feats = np.array(test_feats)
    pred_probs = lr.predict_proba(test_feats)[:, 1]
    pred_probs = np.clip(pred_probs, 1e-7, 1 - 1e-7)

    submission = pd.DataFrame({"id": test_ids, "label": pred_probs})



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, rows:", submission.shape[0])
