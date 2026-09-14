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
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.32756

# 6. Current score

0.05716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02663) has done: 'I fix the environment-breaking Keras import error by switching to `tensorflow.keras`, which is the compatible API in this Kaggle image environment. Then I repair small pandas and encoder bugs (`iteritems`→`items`, wrong attribute names, wrong inverse methods) and update deprecated training/inference calls (`fit_generator`→`fit`, `predict_proba`→`predict`). Finally, I ensure the submission is written to `working/submission.csv` using the exact images and row order from `sample_submission.csv`, which fixes the “wrong number of rows” error and guarantees a valid `.csv` output.'
- What this solution (achieved 0.00057) has done: 'I fix the crash happening at import time by forcing TensorFlow to use the pure-Python protobuf implementation and by avoiding the problematic standalone `keras` import path, which is what triggers the `MessageFactory.GetPrototype` error in some Kaggle TF/protobuf builds. Then I keep your model/training loop intact and only make inference consistent with training normalization by reusing the same `ImageDataGenerator` and ensuring test images are standardized the same way (no logic change, just making it actually execute). Finally, I keep the submission row order exactly as `sample_submission.csv` and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.00027) has done: 'The crash happens before any training because importing TensorFlow triggers an incompatible protobuf API (`MessageFactory.GetPrototype`) in this Kaggle environment; the most reliable minimal fix is to force-install a protobuf version compatible with TF and then import TensorFlow. After that, the rest of your pipeline can run as-is, but your current MAP@5 is near-zero because the label decoding is wrong: you’re passing class indices into the `LabelEncoder` without going back through the `OneHotEncoder` category mapping. I fix `inverse_labels` to correctly map predicted class indices → original string Ids, which should move the score substantially toward your target without changing the model/training setup. I also keep the submission row order exactly matching `sample_submission.csv` and write `submission.csv`.'
- What this solution (achieved 0.11452) has done: 'Your current near-zero MAP@5 is mostly because the submission never includes `new_whale`, which the metric strongly rewards when the model is unsure (and with this small CNN + few epochs, it be unsure often). I keep your model/training exactly the same, but change only the prediction post-processing to (1) guarantee `new_whale` is present in the top-5 list and (2) avoid writing duplicate labels in the 5 predictions. This is a minimal semantic change to align predictions with the competition’s submission rules and typically moves the score substantially upward toward your target without touching architecture or training. I also keep the exact sample_submission row order and still write `submission.csv`.'
- What this solution (achieved 0.11454) has done: 'We keep your CNN, training loop, preprocessing, and loss exactly as-is, and only adjust the MAP@5-oriented post-processing. Right now you always force `new_whale` into rank-1, which can suppress correct known-IDs; instead we include `new_whale` only when the model is sufficiently uncertain, and otherwise return the model’s top-5 unique IDs. This is a minimal semantic change confined to submission construction, and it should move your score upward from 0.1145 toward the 0.3276 target without touching the model. We also keep strict row alignment with `sample_submission.csv` and ensure exactly 5 unique predictions per row.'
- What this solution (achieved 0.1143) has done: 'The timeout is dominated by two hotspots: (1) loading all train images into RAM via a slow Python/PIL loop, and (2) per-test-image TTA using `ImageDataGenerator.flow(..., batch_size=1)` inside nested loops (2600 images × 3 passes), which adds large Python overhead. I keep the exact same model and training loop semantics, but speed up data loading with a cached, faster image reader and vectorized preallocation, and I make test-time prediction batched per TTA pass (still 3 passes, same generator transforms) to remove the per-image generator/predict overhead. I also avoid repeated `toarray()` conversions and repeated CSV reads, which are pure overhead and don’t affect results. The submission-row mismatch error is fixed by writing exactly one line per `sample_submission` row without re-reading the CSV multiple times.'
- What this solution (achieved 0.11443) has done: 'Your current score (0.1143) is far below the target (0.32756), so we should cautiously increase MAP@5 without changing the model/training core. The biggest minimal win here is improving the `new_whale` insertion rule: using a fixed top1 probability threshold (0.55) is poorly calibrated for this small CNN and tends to misplace `new_whale`; instead we use a per-image uncertainty rule based on the *margin* between top-1 and top-2 probabilities, and we place `new_whale` at rank-1 only when uncertain, otherwise at rank-2 (so it doesn’t suppress a confident known-ID). We also filter out any accidental `new_whale` coming from the label set (it shouldn’t, but this prevents duplicates and makes ranking behavior consistent), while still outputting exactly 5 unique predictions per image in sample submission order. Everything else (image loading, preprocessing, architecture, epochs, optimizer, TTA passes) stays the same.'
- What this solution (achieved 0.11454) has done: 'Your current MAP@5 is far below the target, so we should increase it with the smallest change that affects ranking quality without touching the CNN, training loop, or augmentation/training semantics. The most direct lever is your `new_whale` placement rule: using a fixed margin threshold is usually miscalibrated, so we replace it with an adaptive, per-image rule based on the model’s own confidence distribution (normalized entropy), which tends to insert `new_whale` only when the prediction is genuinely diffuse. We keep exactly the same top-5 construction logic (unique labels, always include `new_whale` once) but decide whether `new_whale` is rank-1 or rank-2 using this more reliable uncertainty signal. Everything else (data loading, ImageDataGenerator usage, TTA passes, model, epochs, optimizer, file paths, and submission formatting) stays the same.'
- What this solution (achieved 0.11467) has done: 'We keep your CNN, training loop, augmentation, and TTA exactly as-is, and only adjust the MAP@5 post-processing because that’s the smallest lever that can move the score upward from 0.1145 toward 0.3276. The current entropy-threshold rule is likely inserting `new_whale` too rarely (ENTROPY_THRESH=0.90), which hurts MAP@5 on this dataset where many test images truly are “unknown” given the weak classifier. I switch the uncertainty gate from normalized entropy to a simpler and usually better-calibrated confidence rule using `top1_prob` and `top1-top2` margin, and I tune it to include `new_whale` more often but still not always at rank-1. Everything else (paths, preprocessing, label decoding, submission ordering/format) stays unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.11438) has done: 'We keep your CNN, training loop, augmentation, and 3-pass TTA unchanged, and only adjust the MAP@5 post-processing because that’s the smallest lever to raise the score from 0.11467 toward the 0.32756 target. The main issue is that `new_whale` placement is still too conservative for this weak classifier; increasing the frequency of placing `new_whale` at rank-1 when the model is uncertain typically boosts MAP@5. Concretely, we slightly relax the uncertainty gates by increasing `TOP1_PROB_THRESH` and `MARGIN_THRESH` so more images get `new_whale` earlier, while preserving unique top-5 outputs and sample_submission row order. Everything else remains identical and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.11446) has done: 'We keep your CNN, training loop, preprocessing, and 3-pass TTA exactly the same, and only adjust the MAP@5 post-processing because that’s the smallest lever to move your score upward toward the 0.32756 target. Your current rule likely still places `new_whale` too aggressively (hurting cases where the true ID is known), so we make the “uncertain” gate slightly stricter and also add a tiny, calibration-style safeguard: only insert `new_whale` at rank-1 when the model is both low-confidence and not clearly separated from rank-2. This preserves the same evaluation semantics (top-5 ranking) while improving the ordering quality without changing the model. We also keep unique top-5 labels, stable row order matching `sample_submission.csv`, and a valid `submission.csv`.'
- What this solution (achieved 0.05716) has done: 'Your current gap to target is large (0.11446 → 0.32756), but we should still make the smallest changes that can materially improve MAP@5 without touching your CNN/training. The biggest low-risk lever is the `new_whale` placement rule: right now it often suppresses correct known-IDs by inserting `new_whale` too early/too often; we replace it with a simple, more robust calibration rule based on `new_whale` prior + renormalization (a standard trick for this competition) while keeping exactly 5 unique predictions. Concretely, we estimate a global `p_new_whale` from the train.csv label frequency, then at inference time inject that probability into every test prediction and re-rank; this tends to boost MAP@5 for weak classifiers without changing the model. Everything else (paths, preprocessing, augmentation/TTA count, architecture, epochs, optimizer, submission order/format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

print("Using BASE_INPUT:", BASE_INPUT)
print("Listing input dir:")
print("\n".join(sorted(os.listdir(BASE_INPUT))[:50]))




## === cell 1
def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_csv_path = resolve_path(
    os.path.join(BASE_INPUT, "train.csv"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "train.csv"),
)
sample_sub_path = resolve_path(
    os.path.join(BASE_INPUT, "sample_submission.csv"),
    os.path.join(
        BASE_INPUT, "whale-categorization-playground", "sample_submission.csv"
    ),
)
train_dir = resolve_path(
    os.path.join(BASE_INPUT, "train"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "train"),
)
test_dir = resolve_path(
    os.path.join(BASE_INPUT, "test"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "test"),
)

print("train_csv_path:", train_csv_path)
print("sample_sub_path:", sample_sub_path)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

train_images = sorted(glob(os.path.join(train_dir, "*jpg")))
test_images = sorted(glob(os.path.join(test_dir, "*jpg")))

df = pd.read_csv(train_csv_path)
df["ImagePath"] = df["Image"].map(lambda x: os.path.join(train_dir, x))
ImageToLabelDict = dict(zip(df["ImagePath"], df["Id"]))

print("Train images found:", len(train_images))
print("Test images found:", len(test_images))
print("Train CSV rows:", len(df))




## === cell 2
SIZE = 64

from functools import lru_cache


@lru_cache(maxsize=None)
def ImportImage(filename):
    img = Image.open(filename).convert("LA").resize((SIZE, SIZE))
    return np.array(img, dtype=np.uint8)[:, :, 0]


n_train = len(train_images)
train_img = np.empty((n_train, SIZE, SIZE), dtype=np.uint8)
for i, p in enumerate(train_images):
    train_img[i] = ImportImage(p)
x = train_img




## === cell 3
print("%d training images" % x.shape[0])

print("Nbr of samples/class\tNbr of classes")
for index, val in df["Id"].value_counts().value_counts().sort_index().items():
    print("%d\t\t\t%d" % (index, val))




## === cell 4
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class LabelOneHotEncoder:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")
        self.le = LabelEncoder()

    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1, 1))

    def transform(self, x):
        features = self.le.transform(x)
        return self.ohe.transform(features.reshape(-1, 1))

    def inverse_labels(self, class_indices):
        class_indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
        ohe_categories = self.ohe.categories_[0]
        int_labels = ohe_categories[class_indices]
        return self.le.inverse_transform(int_labels.astype(np.int64))


y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)

y_dense = y_cat.toarray().astype(np.float32, copy=False)




## === cell 5
def plotImages(images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12, 12))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        if img.ndim != 2:
            img = img.reshape((SIZE, SIZE))
        ax.imshow(img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()


plotImages(x)




## === cell 6
x = x.reshape((-1, SIZE, SIZE, 1))
input_shape = x[0].shape
x_train = x.astype("float32", copy=False)
y_train = y_cat

image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
)

image_gen.fit(x_train, augment=True)

augmented_images, _ = next(
    image_gen.flow(x_train, y_dense, batch_size=4 * 4, shuffle=True)
)
plotImages(augmented_images)




## === cell 7
batch_size = 128
num_classes = y_cat.shape[1]
epochs = 5  # keep original

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print("num_classes:", num_classes)

model = Sequential()
model.add(
    tf.keras.layers.Conv2D(
        48, kernel_size=(3, 3), activation="relu", input_shape=input_shape
    )
)
model.add(tf.keras.layers.Conv2D(48, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.33))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(24, activation="relu"))
model.add(tf.keras.layers.Dropout(0.33))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)
model.summary()

model.fit(
    image_gen.flow(x_train, y_dense, batch_size=batch_size, shuffle=True),
    steps_per_epoch=25,
    epochs=epochs,
    verbose=1,
)




## === cell 8
sub_df = pd.read_csv(sample_sub_path)
assert "Image" in sub_df.columns and "Id" in sub_df.columns

test_path_map = {os.path.basename(p): p for p in test_images}

NEW_WHALE_TOKEN = "new_whale"
TTA_PASSES = 3

label_names = lohe.inverse_labels(np.arange(num_classes))
label_to_index = {lbl: i for i, lbl in enumerate(label_names)}
new_whale_idx = label_to_index.get(NEW_WHALE_TOKEN, None)

p_new_from_train = float((df["Id"] == NEW_WHALE_TOKEN).mean()) if "Id" in df else 0.0
P_NEW_WHALE = float(np.clip(p_new_from_train, 0.05, 0.35))
print("Estimated P_NEW_WHALE from train:", p_new_from_train, "-> using:", P_NEW_WHALE)

test_names = sub_df["Image"].tolist()
n_test = len(test_names)

x_test = np.empty((n_test, SIZE, SIZE, 1), dtype=np.float32)
for i, image_name in enumerate(test_names):
    image_path = test_path_map.get(image_name, os.path.join(test_dir, image_name))
    img = ImportImage(image_path).astype("float32", copy=False)
    x_test[i, :, :, 0] = img

probs_sum = np.zeros((n_test, num_classes), dtype=np.float32)

for _ in range(TTA_PASSES):
    gen = image_gen.flow(x_test, batch_size=256, shuffle=False)
    p = model.predict(gen, steps=int(np.ceil(n_test / 256.0)), verbose=0).astype(
        np.float32
    )
    p = p[:n_test]
    probs_sum += p

y_pred_all = probs_sum / float(TTA_PASSES)

out_path = "submission.csv"  # must end with .csv for Kaggle
with open(out_path, "w") as f:
    f.write("Image,Id\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for i, image_name in enumerate(test_names):
            y_pred = y_pred_all[i].astype(np.float32, copy=False)

            if new_whale_idx is not None:
                y_adj = (1.0 - P_NEW_WHALE) * y_pred
                y_adj[new_whale_idx] = P_NEW_WHALE
                s = float(y_adj.sum())
                if s > 0:
                    y_adj /= s
            else:
                y_adj = y_pred

            order = np.argsort(y_adj)[::-1]

            predicted_tags = []
            for j in order:
                lbl = label_names[j]
                if lbl not in predicted_tags:
                    predicted_tags.append(lbl)
                if len(predicted_tags) >= 20:
                    break

            final = []
            for lbl in predicted_tags:
                if lbl == NEW_WHALE_TOKEN:
                    continue
                if lbl not in final:
                    final.append(lbl)
                if len(final) == 5:
                    break

            if NEW_WHALE_TOKEN not in final:
                if len(final) == 0:
                    final = [NEW_WHALE_TOKEN]
                else:
                    final = [final[0], NEW_WHALE_TOKEN] + final[1:]
                final = final[:5]

            if len(final) < 5:
                for lbl in predicted_tags:
                    if lbl not in final and lbl != NEW_WHALE_TOKEN:
                        final.append(lbl)
                    if len(final) == 5:
                        break
            if len(final) < 5:
                if NEW_WHALE_TOKEN not in final:
                    final.append(NEW_WHALE_TOKEN)
            final = final[:5]

            f.write("{},{}\n".format(image_name, " ".join(final)))

print("Wrote submission:", out_path)
sub_preview = pd.read_csv(out_path)
print("Submission preview:")
print(sub_preview.head())
print("Submission rows:", len(sub_preview))
assert len(sub_preview) == len(sub_df)
assert list(sub_preview.columns) == ["Image", "Id"]
