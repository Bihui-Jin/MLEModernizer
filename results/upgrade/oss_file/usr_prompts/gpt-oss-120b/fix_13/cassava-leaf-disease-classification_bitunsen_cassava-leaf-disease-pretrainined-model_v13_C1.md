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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8458748866727108

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'The changes increase data‑loading parallelism and double the batch size, which cuts the number of training steps roughly in half and uses multiple CPU workers for image preprocessing. TensorFlow thread settings are also tuned to keep all cores busy. These adjustments keep the model architecture, training schedule, and augmentation logic unchanged, so the predictions remain the same while fitting comfortably inside the 600 s limit.'
- What this solution (achieved 0.05531) has done: 'The changes increase the batch size and enable multiprocessing data loading, which cuts the number of training steps and speeds up image preprocessing without altering the model architecture, training schedule, or evaluation logic. Adding `workers` and `use_multiprocessing` to the generators and `fit` calls lets TensorFlow load and augment images in parallel, preserving identical results while fitting within the 600‑second limit.'

# 9. Code solution

## === cell 0
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1426506867.py in <cell line: 0>()
      1 BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
----> 2 TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
      3 TEST_DIR = os.path.join(BASE_DIR, "test_images")
      4 

NameError: name 'os' is not defined

## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    label_map = json.load(f)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(str)

sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_filenames = sample_sub["image_id"].tolist()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2627091322.py in <cell line: 0>()
----> 1 with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
      2     label_map = json.load(f)
      3 
      4 train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
      5 train_df["label"] = train_df["label"].astype(str)

NameError: name 'os' is not defined

## === cell 2
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 512  # larger batch halves the number of steps per epoch
NUM_CLASSES = 5
SEED = 42



## === cell 3
np.random.seed(SEED)
perm = np.random.permutation(len(train_df))
split_idx = int(0.2 * len(train_df))
val_idx, train_idx = perm[:split_idx], perm[split_idx:]

train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split = train_df.iloc[val_idx].reset_index(drop=True)

train_datagen_aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator()  # no augmentation, no rescaling needed

from tensorflow.keras.utils import Sequence, to_categorical


class CachedImageSequence(Sequence):
    """Loads each image once, caches it, and applies optional augmentation."""

    def __init__(self, df, directory, batch_size, augmentor=None, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.directory = directory
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.cached = {}  # filename -> numpy array (float32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idxs = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = self.df.iloc[batch_idxs]["image_id"].values
        batch_labels = self.df.iloc[batch_idxs]["label"].values.astype(int)

        imgs = []
        for fname in batch_paths:
            if fname in self.cached:
                img = self.cached[fname]
            else:
                img_path = os.path.join(self.directory, fname)
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img = img_to_array(img).astype(np.float32) / 255.0  # manual rescale
                self.cached[fname] = img
            imgs.append(img)

        x = np.stack(imgs, axis=0)
        y = to_categorical(batch_labels, NUM_CLASSES)

        if self.augmentor is not None:
            aug_iter = self.augmentor.flow(
                x, y, batch_size=self.batch_size, shuffle=False
            )
            x, y = next(aug_iter)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


train_sequence = CachedImageSequence(
    df=train_df_split,
    directory=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    augmentor=train_datagen_aug,
    shuffle=True,
)

val_sequence = CachedImageSequence(
    df=val_df_split,
    directory=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    augmentor=None,
    shuffle=False,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1670204939.py in <cell line: 0>()
      1 # ----- Optimize data loading by caching images in memory -----
----> 2 np.random.seed(SEED)
      3 perm = np.random.permutation(len(train_df))
      4 split_idx = int(0.2 * len(train_df))
      5 val_idx, train_idx = perm[:split_idx], perm[split_idx:]

NameError: name 'np' is not defined

## === cell 4
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False  # freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3272045781.py in <cell line: 0>()
----> 1 base_model = EfficientNetB0(
      2     weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
      3 )
      4 base_model.trainable = False  # freeze base
      5 

NameError: name 'EfficientNetB0' is not defined

## === cell 5
callbacks = [
    EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=0
    ),
]

model.fit(
    train_sequence,
    epochs=20,
    validation_data=val_sequence,
    callbacks=callbacks,
    verbose=2,
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_sequence,
    epochs=10,
    validation_data=val_sequence,
    callbacks=callbacks,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/264473065.py in <cell line: 0>()
      1 callbacks = [
----> 2     EarlyStopping(
      3         monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
      4     ),
      5     ModelCheckpoint(

NameError: name 'EarlyStopping' is not defined

## === cell 6
class CachedTestSequence(Sequence):
    def __init__(self, filenames, directory, batch_size):
        self.filenames = filenames
        self.directory = directory
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.filenames))
        self.cached = {}

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_idxs = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = [self.filenames[i] for i in batch_idxs]

        imgs = []
        for fname in batch_paths:
            if fname in self.cached:
                img = self.cached[fname]
            else:
                img_path = os.path.join(self.directory, fname)
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img = img_to_array(img).astype(np.float32) / 255.0
                self.cached[fname] = img
            imgs.append(img)

        x = np.stack(imgs, axis=0)
        return x


test_sequence = CachedTestSequence(test_filenames, TEST_DIR, BATCH_SIZE)

pred_probs = model.predict(test_sequence, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1).astype(str)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1242353376.py in <cell line: 0>()
      1 # ----- Cache test images to avoid repeated disk reads -----
----> 2 class CachedTestSequence(Sequence):
      3     def __init__(self, filenames, directory, batch_size):
      4         self.filenames = filenames
      5         self.directory = directory

NameError: name 'Sequence' is not defined

## === cell 7
submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2731632357.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 
      5 print(f"Submission file written to {submission_path}, shape: {submission.shape}")

NameError: name 'pd' is not defined
