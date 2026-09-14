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

20.4855

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys, subprocess, os


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure protobuf compatibility: {e}")


_ensure_protobuf_compat()



## === cell 1
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/kaggle/input/petfinder-pawpularity-score"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(train_csv.shape, test_csv.shape, submission.shape)
train_csv.head()



## === cell 2
train_csv.isnull().sum()



## === cell 3
train_csv = train_csv.drop_duplicates()
train_csv.shape



## === cell 4
for col in train_csv.drop(["Id", "Pawpularity"], axis=1).columns:
    plt.figure(figsize=(4, 2))
    sns.countplot(x=train_csv[col])
    plt.title(col)
    plt.tight_layout()
    plt.show()



## === cell 5
plt.figure(figsize=(5, 3))
sns.histplot(train_csv["Pawpularity"], kde=True, bins=30)
plt.tight_layout()
plt.show()



## === cell 6
test_csv.isnull().sum()



## === cell 7
submission.head()



## === cell 8
train_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])
sizes = []
for fname in train_files:
    path = os.path.join(TRAIN_DIR, fname)
    imgg = cv2.imread(path)
    if imgg is None:
        continue
    w, h, c = imgg.shape
    sizes.append((w, h, c, imgg.size / 3.0))
size_data = pd.DataFrame(sizes, columns=["w", "h", "c", "pixels"])
size_data.head()



## === cell 9
size_data[size_data["pixels"] == size_data["pixels"].min()].head()



## === cell 10
size_data["pixels"].value_counts().head()



## === cell 11
size_data[size_data["pixels"] == 691200].head()



## === cell 12
train_img = []
train_img_name = []
for fname in train_files:
    path = os.path.join(TRAIN_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    train_img.append(img / 255.0)
    train_img_name.append(fname)
train_img[:1], len(train_img), len(train_img_name)



## === cell 13
train_img_name[:5]



## === cell 14
bad = [name for name in train_img_name if not name.lower().endswith(".jpg")]
print("Non-jpg in list:", bad[:5], "count:", len(bad))



## === cell 15
train_ids = [n[:-4] for n in train_img_name]
train_csv_data = train_csv[train_csv["Id"].isin(train_ids)].copy()

order = pd.Categorical(train_csv_data["Id"], categories=train_ids, ordered=True)
train_csv_data = (
    train_csv_data.assign(_order=order)
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

id_set = set(train_csv_data["Id"].tolist())
filtered_imgs, filtered_names = [], []
for img, name in zip(train_img, train_img_name):
    if name[:-4] in id_set:
        filtered_imgs.append(img)
        filtered_names.append(name)
train_img = filtered_imgs
train_img_name = filtered_names

train_ids = [n[:-4] for n in train_img_name]
order2 = pd.Categorical(train_csv_data["Id"], categories=train_ids, ordered=True)
train_csv_data = (
    train_csv_data.assign(_order=order2)
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

print("Aligned:", len(train_img), len(train_csv_data))
train_csv_data.head()



## === cell 16
train_csv_data = train_csv_data.reset_index(drop=True)
train_csv_data.head()



## === cell 17
image_1 = cv2.imread(os.path.join(TRAIN_DIR, train_csv_data.loc[0, "Id"] + ".jpg"))
plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 18
plt.imshow(train_img[0])
plt.axis("off")
plt.show()



## === cell 19
image_2 = cv2.imread(os.path.join(TRAIN_DIR, train_csv_data.loc[1, "Id"] + ".jpg"))
plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 20
plt.imshow(train_img[1])
plt.axis("off")
plt.show()



## === cell 21
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
for i, fname in enumerate(test_files[:5]):
    file = cv2.imread(os.path.join(TEST_DIR, fname))
    print(fname, None if file is None else file.shape)



## === cell 22
test_img = []
test_img_name = []
for fname in test_files:
    img = cv2.imread(os.path.join(TEST_DIR, fname))
    if img is None:
        continue
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    test_img.append(img / 255.0)
    test_img_name.append(fname)
test_img[:1], len(test_img), len(test_img_name)



## === cell 23
test_img_name[:5]



## === cell 24
test_ids = [n[:-4] for n in test_img_name]
test_csv_data = test_csv[test_csv["Id"].isin(test_ids)].copy()
order = pd.Categorical(test_csv_data["Id"], categories=test_ids, ordered=True)
test_csv_data = (
    test_csv_data.assign(_order=order)
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

id_set = set(test_csv_data["Id"].tolist())
filtered_imgs, filtered_names = [], []
for img, name in zip(test_img, test_img_name):
    if name[:-4] in id_set:
        filtered_imgs.append(img)
        filtered_names.append(name)
test_img = filtered_imgs
test_img_name = filtered_names

test_ids = [n[:-4] for n in test_img_name]
order2 = pd.Categorical(test_csv_data["Id"], categories=test_ids, ordered=True)
test_csv_data = (
    test_csv_data.assign(_order=order2)
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

print("Aligned:", len(test_img), len(test_csv_data))
test_csv_data.head()



## === cell 25
test_1 = cv2.imread(os.path.join(TEST_DIR, test_csv_data.loc[0, "Id"] + ".jpg"))
plt.imshow(cv2.cvtColor(test_1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 26
plt.imshow(test_img[0])
plt.axis("off")
plt.show()



## === cell 27
train_csv_x = train_csv_data.drop(["Id", "Pawpularity"], axis=1)
train_y = train_csv_data["Pawpularity"].astype(np.float32)

test_csv_x = test_csv_data.drop(["Id"], axis=1)

train_csv_x = train_csv_x.astype(np.float32)
test_csv_x = test_csv_x.astype(np.float32)

train_img_arr = np.array(train_img, dtype=np.float32)
test_img_arr = np.array(test_img, dtype=np.float32)

print(
    train_csv_x.shape,
    train_img_arr.shape,
    train_y.shape,
    test_csv_x.shape,
    test_img_arr.shape,
)



## === cell 28
tf.keras.utils.set_random_seed(42)
np.random.seed(42)



## === cell 29
csv_input = tf.keras.Input(shape=train_csv_x.shape[1:], name="CSV_Input")
img_input = tf.keras.Input(shape=train_img_arr.shape[1:], name="IMG_Input")

csv_hidden1 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden1"
)(csv_input)
csv_hidden2 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden2"
)(csv_hidden1)
csv_hidden3 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden3"
)(csv_hidden2)
csv_hidden4 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden4"
)(csv_hidden3)
csv_hidden5 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden5"
)(csv_hidden4)
csv_hidden6 = tf.keras.layers.Dense(
    200, activation="elu", kernel_initializer="he_normal", name="CSV_Hidden6"
)(csv_hidden5)
csv_dropout = tf.keras.layers.Dropout(0.5, name="CSV_Dropout")(csv_hidden6)

img_conv1 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv1",
)(img_input)
img_conv2 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv2",
)(img_conv1)
img_pooling1 = tf.keras.layers.MaxPooling2D(4, name="IMG_Max1")(img_conv2)

img_conv3 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv3",
)(img_pooling1)
img_conv4 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv4",
)(img_conv3)
img_pooling2 = tf.keras.layers.MaxPooling2D(4, name="IMG_Max2")(img_conv4)

img_conv5 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv5",
)(img_pooling2)
img_conv6 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv6",
)(img_conv5)
img_pooling3 = tf.keras.layers.MaxPooling2D(3, name="IMG_Max3")(img_conv6)

img_dropout = tf.keras.layers.Dropout(0.5, name="IMG_Dropout")(img_pooling3)
img_conv7 = tf.keras.layers.Conv2D(
    120,
    4,
    padding="same",
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_Conv7",
)(img_dropout)

img_hidden1 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden1",
    use_bias=False,
)(img_conv7)
img_dropout1 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout1")(img_hidden1)
img_hidden2 = tf.keras.layers.Dense(
    300,
    activation="elu",
    kernel_initializer="he_normal",
    name="IMG_hidden2",
    use_bias=False,
)(img_dropout1)

img_gpool = tf.keras.layers.GlobalAvgPool2D(name="IMG_Gpool")(img_hidden2)
img_dropout2 = tf.keras.layers.Dropout(0.5, name="IMG_Dropout2")(img_gpool)

csv_output = tf.keras.layers.Dense(1, name="CSV_Output")(csv_dropout)
img_output = tf.keras.layers.Dense(1, name="IMG_Output")(img_dropout2)

model = tf.keras.Model(
    inputs=[csv_input, img_input],
    outputs=[csv_output, img_output],
    name="Pythonash_model",
)
model.summary()



## === cell 30
try:
    tf.keras.utils.plot_model(
        model,
        to_file="/kaggle/working/model.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
    )
    print("Saved model plot to /kaggle/working/model.png")
except Exception as e:
    print("plot_model skipped:", e)



## === cell 31
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=10000, decay_rate=0.97
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

model.compile(
    loss=["mse", "mse"],
    loss_weights=[0.5, 0.5],
    optimizer=opt,
    metrics=tf.keras.metrics.RootMeanSquaredError(),
)

epoch_number = 20
check_1 = tf.keras.callbacks.ModelCheckpoint(
    "/kaggle/working/pythonash_model.keras", save_best_only=True, verbose=2
)



## === cell 32
history = model.fit(
    x=[train_csv_x.values, train_img_arr],
    y=[train_y.values, train_y.values],
    epochs=epoch_number,
    validation_split=0.2,
    verbose=2,
    batch_size=100,
    validation_batch_size=100,
    callbacks=[check_1],
)



## === cell 33
best_model = tf.keras.models.load_model(
    "/kaggle/working/pythonash_model.keras", compile=False
)
csv_result, img_result = best_model.predict(
    [test_csv_x.values, test_img_arr], batch_size=100, verbose=1
)

final_pred = 0.5 * csv_result.reshape(-1) + 0.5 * img_result.reshape(-1)

final_pred = np.clip(final_pred, 0, 100)

final_result = pd.DataFrame({"Pawpularity": final_pred.astype(np.float32)})
final_result.head()



## === cell 34
sub = pd.DataFrame(
    {
        "Id": test_csv_data["Id"].values,
        "Pawpularity": final_result["Pawpularity"].values,
    }
)

submission_out = submission[["Id"]].merge(sub, on="Id", how="left")
fallback = (
    float(np.nanmean(submission_out["Pawpularity"]))
    if submission_out["Pawpularity"].notna().any()
    else 50.0
)
submission_out["Pawpularity"] = submission_out["Pawpularity"].fillna(fallback)

out_path = "/kaggle/working/submission.csv"
submission_out.to_csv(out_path, index=False)
print("Wrote:", out_path, submission_out.shape)
submission_out.head()
